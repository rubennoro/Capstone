// sim_deploy/sim_pc (E4): the io thread of SimWorld: the state export (ZMQ PUB, sim.state/v1, DESIGN.md section 4)
// and the scene control channel (ZMQ REP, JSON, DESIGN.md section 5). Requests that read or change the physics run
// on the physics thread between two steps (SimWorld::runOnPhysics). P1: {"cmd": "key", "key": "h|p|g|s"} pushes the
// key into franka_control's KeyQueue (the keyboard thread's path) and replies, without blocking the state export,
// once the supervisor has logged its one line about that key.
#include <cmath>
#include <cstdio>
#include <sstream>

#include "franka_control/keyboard_reader.h"

#include <yaml-cpp/yaml.h>
#include <zmq.hpp>

#include "franka_control/clock.h"
#include "franka_control/log.h"
#include "sim_pc/sim_world.h"

namespace sim_pc {

using franka_control::Log;
using franka_control::monotonicNowNs;

namespace {

std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

std::string str(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string arr(const double* v, int n) {
  std::string out = "[";
  for (int i = 0; i < n; ++i) out += (i ? ", " : "") + num(v[i]);
  return out + "]";
}

std::string fail(const std::string& error) { return "{\"ok\": false, \"error\": " + str(error) + "}"; }

constexpr double kKeyAnswerS = 1.0;   // the messaging thread drains the KeyQueue every <= 10 ms

std::string fmt(double v, int prec) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return buf;
}

}  // namespace

void SimWorld::ioLoop() {
  zmq::context_t ctx(1);
  zmq::socket_t pub(ctx, zmq::socket_type::pub);
  zmq::socket_t rep(ctx, zmq::socket_type::rep);
  pub.set(zmq::sockopt::linger, 0);
  pub.set(zmq::sockopt::sndhwm, 1000);
  rep.set(zmq::sockopt::linger, 0);
  const std::string state_ep = "tcp://" + sim_.bind_address + ":" + std::to_string(sim_.state_port);
  const std::string scene_ep = "tcp://" + sim_.bind_address + ":" + std::to_string(sim_.scene_port);
  try {
    pub.bind(state_ep);
  } catch (const zmq::error_t& e) {
    io_error_ = "cannot bind the sim state endpoint " + state_ep + " (" + e.what() + "): another sim running?";
    io_ready_.store(true);
    return;
  }
  try {
    rep.bind(scene_ep);
  } catch (const zmq::error_t& e) {
    io_error_ = "cannot bind the sim scene endpoint " + scene_ep + " (" + e.what() + "): another sim running?";
    io_ready_.store(true);
    return;
  }
  Log::info("sim: state PUB on " + state_ep + " (sim.state/v1 every " + std::to_string(sim_.state_period_steps) +
            " ms), scene REP on " + scene_ep);
  io_ready_.store(true);

  const std::string head = "{\"schema\": \"sim.state/v1\", \"t_ns\": ";
  uint64_t sent = 0, requests = 0, sent_window = 0;
  SimStats prev = stats();
  int64_t last_log = monotonicNowNs();
  const int64_t log_period = franka_control::secondsToNs(sim_.stats_period_s);
  auto logStats = [&](const char* label) {
    const SimStats s = stats();
    const double dwall = s.wall_s - prev.wall_s, dphys = s.physics_s - prev.physics_s;
    Log::info(std::string("sim: ") + label + ": rtf " + fmt(dwall > 0 ? dphys / dwall : 0.0, 4) + " (" +
              fmt(dphys, 2) + " s physics / " + fmt(dwall, 2) + " s wall), slow steps +" +
              std::to_string(s.slow_steps - prev.slow_steps) + " (total " + std::to_string(s.slow_steps) +
              ", > " + fmt(sim_.slow_lag_ms, 0) + " ms behind), resyncs " + std::to_string(s.resyncs) +
              ", max lag " + fmt(s.max_lag_ms, 2) + " ms, step cost mean " + fmt(s.step_cost_mean_us, 0) +
              " us max " + fmt(s.step_cost_max_us, 0) + " us, state msgs " +
              fmt(dwall > 0 ? static_cast<double>(sent_window) / dwall : 0.0, 1) + "/s (total " +
              std::to_string(sent) + "), scene requests " + std::to_string(requests) + ", torque saturations " +
              std::to_string(s.torque_saturations) + ", reflexes " + std::to_string(s.reflexes));
    prev = s;
    sent_window = 0;
  };

  std::optional<int64_t> key_deadline;   // a key request waits for its line (REP: no recv until it is answered)
  while (running_.load()) {
    zmq::pollitem_t items[] = {{static_cast<void*>(rep), 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, std::chrono::milliseconds(1));
    if (key_deadline) {
      if (auto reply = keyReply(monotonicNowNs() > *key_deadline)) {
        (void)rep.send(zmq::buffer(*reply), zmq::send_flags::none);
        key_deadline.reset();
      }
    } else if (items[0].revents & ZMQ_POLLIN) {
      zmq::message_t msg;
      if (rep.recv(msg, zmq::recv_flags::dontwait)) {
        ++requests;
        std::string reply;
        try {
          reply = handleSceneRequest(msg.to_string());
        } catch (const std::exception& e) {
          reply = fail(e.what());
        }
        if (reply.empty()) {
          key_deadline = monotonicNowNs() + franka_control::secondsToNs(kKeyAnswerS);
        } else {
          (void)rep.send(zmq::buffer(reply), zmq::send_flags::none);
        }
      }
    }
    std::deque<StateMessage> q;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      q.swap(state_queue_);
    }
    for (const auto& s : q) {
      std::string out = head + std::to_string(s.t_ns) + ", \"step\": " + std::to_string(s.step) +
                        ", \"model_sha256\": \"" + model_sha_ + "\", \"qpos\": " +
                        arr(s.qpos.data(), static_cast<int>(s.qpos.size())) + "}";
      (void)pub.send(zmq::buffer(out), zmq::send_flags::dontwait);
      ++sent;
      ++sent_window;
    }
    if (monotonicNowNs() - last_log >= log_period) {
      last_log = monotonicNowNs();
      logStats("stats");
    }
  }
  logStats("final stats (last window)");
  const SimStats s = stats();
  Log::info("sim: run totals: " + std::to_string(s.steps) + " steps, " + fmt(s.physics_s, 2) + " s physics in " +
            fmt(s.wall_s, 2) + " s wall (rtf " + fmt(s.wall_s > 0 ? s.physics_s / s.wall_s : 0.0, 4) +
            "), slow steps " + std::to_string(s.slow_steps) + ", resyncs " + std::to_string(s.resyncs) +
            ", max lag " + fmt(s.max_lag_ms, 2) + " ms, reflexes " + std::to_string(s.reflexes));
}

std::string SimWorld::handleSceneRequest(const std::string& request) {
  YAML::Node req;
  try {
    req = YAML::Load(request);
  } catch (const YAML::Exception& e) {
    return fail(std::string("not JSON: ") + e.what());
  }
  if (!req.IsMap() || !req["cmd"]) return fail("expected a JSON object with \"cmd\"");
  const std::string cmd = req["cmd"].as<std::string>();

  if (cmd == "hello") {
    std::string joints = "{";
    for (int j = 0; j < m_->njnt; ++j) {
      const char* n = mj_id2name(m_, mjOBJ_JOINT, j);
      joints += std::string(j ? ", " : "") + str(n ? n : "") + ": {\"qposadr\": " +
                std::to_string(m_->jnt_qposadr[j]) + ", \"dofadr\": " + std::to_string(m_->jnt_dofadr[j]) + "}";
    }
    joints += "}";
    const SimStats s = stats();
    return "{\"ok\": true, \"schema\": \"sim.scene.hello/v1\", \"model_sha256\": " + str(model_sha_) +
           ", \"model\": " + str(sim_.model.string()) + ", \"nq\": " + std::to_string(m_->nq) +
           ", \"nv\": " + std::to_string(m_->nv) + ", \"timestep_s\": " + num(timestep_s_) +
           ", \"names\": {\"arm_joints\": [\"joint1\", \"joint2\", \"joint3\", \"joint4\", \"joint5\", \"joint6\", "
           "\"joint7\"], \"finger_joints\": [\"finger_joint1\", \"finger_joint2\"], \"sites\": [\"flange\", \"tcp\"], "
           "\"objects\": [\"eraser\"], \"joints\": " + joints + "}, \"gripper_width_offset_m\": " +
           num(cfg_.gripper.width_offset_m) + ", \"stats\": {\"rtf\": " +
           num(s.wall_s > 0 ? s.physics_s / s.wall_s : 0.0) + ", \"slow_steps\": " + std::to_string(s.slow_steps) +
           ", \"resyncs\": " + std::to_string(s.resyncs) + ", \"max_lag_ms\": " + num(s.max_lag_ms) +
           ", \"step_cost_mean_us\": " + num(s.step_cost_mean_us) + ", \"step_cost_max_us\": " +
           num(s.step_cost_max_us) + ", \"reflexes\": " + std::to_string(s.reflexes) + "}, \"t_ns\": " +
           std::to_string(monotonicNowNs()) + "}";
  }

  if (cmd == "set_object") {
    const std::string name = req["name"] ? req["name"].as<std::string>() : "";
    const int body = mj_name2id(m_, mjOBJ_BODY, name.c_str());
    if (body < 0 || m_->body_jntnum[body] != 1 || m_->jnt_type[m_->body_jntadr[body]] != mjJNT_FREE) {
      return fail("set_object: no body with a free joint named \"" + name + "\"");
    }
    const YAML::Node p = req["pose"];
    if (!p || !p.IsSequence() || p.size() != 7) return fail("set_object: pose must be [x, y, z, qw, qx, qy, qz]");
    double pose[7];
    for (size_t i = 0; i < 7; ++i) pose[i] = p[i].as<double>();
    const double qn = std::sqrt(pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] + pose[6] * pose[6]);
    for (double v : pose) {
      if (!std::isfinite(v)) return fail("set_object: pose not finite");
    }
    if (qn < 1e-9) return fail("set_object: zero quaternion");
    for (int i = 3; i < 7; ++i) pose[i] /= qn;
    const int qadr = m_->jnt_qposadr[m_->body_jntadr[body]];
    const int dadr = m_->jnt_dofadr[m_->body_jntadr[body]];
    return runOnPhysics([this, qadr, dadr, pose](int64_t t_ns) {
      for (int i = 0; i < 7; ++i) d_->qpos[qadr + i] = pose[i];
      for (int i = 0; i < 6; ++i) d_->qvel[dadr + i] = 0.0;
      return "{\"ok\": true, \"t_ns\": " + std::to_string(t_ns) + "}";
    });
  }

  if (cmd == "ground_truth") {
    return runOnPhysics([this](int64_t t_ns) { return groundTruthJson(t_ns); });
  }

  if (cmd == "set_arm") {
    const YAML::Node qn = req["q"];
    if (!qn || !qn.IsSequence() || qn.size() != 7) return fail("set_arm: q must be 7 joint angles (rad)");
    std::array<double, 7> q{};
    for (size_t i = 0; i < 7; ++i) {
      q[i] = qn[i].as<double>();
      if (!std::isfinite(q[i])) return fail("set_arm: q not finite");
    }
    return runOnPhysics([this, q](int64_t t_ns) -> std::string {
      {
        std::lock_guard<std::mutex> lock(call_mutex_);
        if (active_) {
          return fail(std::string("set_arm refused: a ") +
                      (active_->kind == ArmCall::Kind::kTorque ? "torque session" : "homing move") +
                      " is running (franka_control holds the arm in a torque session from start-up to shutdown, "
                      "also in idle: reset the arm with C3 home or the scene key h)");
        }
      }
      for (int i = 0; i < 7; ++i) {
        d_->qpos[arm_qadr_[i]] = q[static_cast<size_t>(i)];
        d_->qvel[arm_dadr_[i]] = 0.0;
        q_hold_(i) = q[static_cast<size_t>(i)];
      }
      return "{\"ok\": true, \"t_ns\": " + std::to_string(t_ns) + "}";
    });
  }

  if (cmd == "key") {
    return pressKey(req["key"] && req["key"].IsScalar() ? req["key"].as<std::string>() : std::string());
  }

  if (cmd == "stats") {
    const SimStats s = stats();
    return "{\"ok\": true, \"steps\": " + std::to_string(s.steps) + ", \"physics_s\": " + num(s.physics_s) +
           ", \"wall_s\": " + num(s.wall_s) + ", \"rtf\": " + num(s.wall_s > 0 ? s.physics_s / s.wall_s : 0.0) +
           ", \"slow_steps\": " + std::to_string(s.slow_steps) + ", \"resyncs\": " + std::to_string(s.resyncs) +
           ", \"max_lag_ms\": " + num(s.max_lag_ms) + ", \"step_cost_mean_us\": " + num(s.step_cost_mean_us) +
           ", \"step_cost_max_us\": " + num(s.step_cost_max_us) + ", \"torque_saturations\": " +
           std::to_string(s.torque_saturations) + ", \"reflexes\": " + std::to_string(s.reflexes) + "}";
  }
  return fail("unknown cmd \"" + cmd + "\" (hello, set_object, ground_truth, set_arm, key, stats)");
}

// ---- {"cmd": "key"} (P1) ----

void SimWorld::attachKeys(franka_control::KeyQueue* keys) {
  std::lock_guard<std::mutex> lock(keys_mutex_);
  if (keys == keys_) return;
  keys_ = keys;
  if (keys) {
    Log::setTap([this](Log::Level, const std::string& message) { onLogLine(message); });
  } else {
    Log::setTap(nullptr);
  }
}

std::string SimWorld::pressKey(const std::string& key) {
  const std::optional<franka_control::Key> k =
      key.size() == 1 ? franka_control::KeyboardReader::map(key[0]) : std::nullopt;
  if ((key != "h" && key != "p" && key != "g" && key != "s") || !k) {
    return fail("key: \"key\" must be one of h, p, g, s (q is not queued: stop the sim PC with SIGTERM)");
  }
  std::lock_guard<std::mutex> lock(keys_mutex_);
  if (!keys_) return fail("key: franka_control's key queue is not attached (not ready yet, or shutting down)");
  {
    std::lock_guard<std::mutex> line_lock(key_line_mutex_);
    key_wait_ = "key " + key;
    key_line_.reset();
  }
  keys_->push(*k);   // the keyboard thread's call: the messaging thread pops it and calls Supervisor::onKey
  return "";
}

void SimWorld::onLogLine(const std::string& message) {
  std::lock_guard<std::mutex> lock(key_line_mutex_);
  if (key_wait_.empty() || key_line_) return;   // the first line about the key (p logs a second one)
  if (message.rfind(key_wait_ + ":", 0) == 0 || message.rfind(key_wait_ + " refused:", 0) == 0) {
    key_line_ = message;
  }
}

std::optional<std::string> SimWorld::keyReply(bool timed_out) {
  std::lock_guard<std::mutex> lock(key_line_mutex_);
  const std::string key = key_wait_.substr(4);
  if (key_line_) {
    const bool refused = key_line_->rfind(key_wait_ + " refused:", 0) == 0;
    std::string reply = refused ? "{\"ok\": false, \"key\": " + str(key) + ", \"error\": " + str(*key_line_) + "}"
                                : "{\"ok\": true, \"key\": " + str(key) + ", \"log\": " + str(*key_line_) + "}";
    key_wait_.clear();
    key_line_.reset();
    return reply;
  }
  if (!timed_out) return std::nullopt;
  key_wait_.clear();
  // pushed, but no line within kKeyAnswerS (e.g. s while homing, or before ready): C2 and the log show the result
  return "{\"ok\": true, \"key\": " + str(key) + ", \"log\": null}";
}

std::string SimWorld::groundTruthJson(int64_t t_ns) {
  auto sitePose = [&](int site) {
    double q[4];
    mju_mat2Quat(q, d_->site_xmat + 9 * site);
    const double* p = d_->site_xpos + 3 * site;
    const double v[7] = {p[0], p[1], p[2], q[0], q[1], q[2], q[3]};
    return arr(v, 7);
  };
  std::string objects = "{";
  bool first = true;
  for (int b = 1; b < m_->nbody; ++b) {
    if (m_->body_jntnum[b] != 1 || m_->jnt_type[m_->body_jntadr[b]] != mjJNT_FREE) continue;
    const int qadr = m_->jnt_qposadr[m_->body_jntadr[b]];
    const int dadr = m_->jnt_dofadr[m_->body_jntadr[b]];
    const char* n = mj_id2name(m_, mjOBJ_BODY, b);
    objects += std::string(first ? "" : ", ") + str(n ? n : "") + ": {\"pose\": " + arr(d_->qpos + qadr, 7) +
               ", \"lin_vel\": " + arr(d_->qvel + dadr, 3) + "}";
    first = false;
  }
  objects += "}";
  std::string contacts = "[";
  first = true;
  const int table = mj_name2id(m_, mjOBJ_GEOM, "table");
  for (int c = 0; c < d_->ncon; ++c) {
    const mjContact& con = d_->contact[c];
    if (con.efc_address < 0 || con.geom[0] < 0 || con.geom[1] < 0) continue;
    const int b0 = m_->geom_bodyid[con.geom[0]], b1 = m_->geom_bodyid[con.geom[1]];
    const bool eraser_table = (b0 == eraser_body_ && con.geom[1] == table) || (b1 == eraser_body_ && con.geom[0] == table);
    if (eraser_table) continue;
    double f6[6];
    mj_contactForce(m_, d_, c, f6);
    contacts += std::string(first ? "" : ", ") + "{\"a\": " + str(geomLabel(con.geom[0])) + ", \"b\": " +
                str(geomLabel(con.geom[1])) + ", \"force_n\": " + num(std::abs(f6[0])) + "}";
    first = false;
  }
  contacts += "]";
  const double gap = d_->qpos[fin_qadr_[0]] + d_->qpos[fin_qadr_[1]];
  double q[7];
  for (int i = 0; i < 7; ++i) q[i] = d_->qpos[arm_qadr_[i]];
  std::string call = "none";
  {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (active_) call = active_->kind == ArmCall::Kind::kTorque ? "torque" : "joints";
  }
  return "{\"ok\": true, \"t_ns\": " + std::to_string(t_ns) + ", \"objects\": " + objects + ", \"tcp\": " +
         (tcp_site_ >= 0 ? sitePose(tcp_site_) : std::string("null")) + ", \"flange\": " + sitePose(flange_site_) +
         ", \"finger_gap_m\": " + num(gap) + ", \"gripper_width_m\": " + num(gap + cfg_.gripper.width_offset_m) +
         ", \"arm\": {\"q\": " + arr(q, 7) + ", \"call\": " + str(call) + ", \"reflex\": " +
         (reflex_ ? "true" : "false") + ", \"tau_ext_nm\": " + arr(tau_ext_f_.data(), 7) + "}, \"contacts\": " +
         contacts + "}";
}

}  // namespace sim_pc
