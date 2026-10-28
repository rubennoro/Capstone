// SPDX-License-Identifier: MIT
// T3: rig.yaml loader / validator / canonical YAML / content hash.
#include <gtest/gtest.h>

#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "rig/config/rig_config.hpp"
#include "rig/package/sha256.hpp"

namespace rig {
namespace {

const std::string kExample =
    std::string(RIG_SOURCE_DIR) + "/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml";

std::string read_file(const std::string& p) {
  std::ifstream in(p);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

TEST(RigConfig, ExampleLoadsWithEveryValue) {
  auto r = load_rig_config(kExample);
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  const RigConfig& c = *r;
  EXPECT_EQ(c.source_path, kExample);
  EXPECT_EQ(c.schema_version, 1);
  EXPECT_EQ(c.rig_name, "jetson-imx519-imx219w160-lsm6dsox");
  ASSERT_EQ(c.cameras.size(), 2u);
  const RigCameraConfig& main = c.cameras.at("main");
  EXPECT_EQ(main.role, "main");
  EXPECT_EQ(main.driver.id, "main");
  EXPECT_EQ(main.driver.backend, CameraBackendKind::kArgus);
  EXPECT_EQ(main.driver.width, 1920);
  EXPECT_DOUBLE_EQ(main.driver.fps, 60.0);
  EXPECT_EQ(main.driver.exposure_ns, 8000000);
  EXPECT_EQ(main.driver.focus.mode, FocusMode::kFixed);
  EXPECT_EQ(main.driver.focus.i2c_addr, 0x0C);  // hex in the file
  EXPECT_EQ(main.model.kind, CameraModelKind::kPinholeRadtan);
  EXPECT_EQ(main.storage.width, 1280);
  EXPECT_EQ(main.thread.cpus, std::vector<int>{1});
  const RigCameraConfig& uw = c.cameras.at("uw");
  EXPECT_EQ(uw.role, "ultrawide");
  EXPECT_EQ(uw.model.kind, CameraModelKind::kKannalaBrandt4);
  EXPECT_DOUBLE_EQ(uw.model.fov_max_deg, 160.0);
  EXPECT_EQ(uw.driver.mock.seed, 2u);
  EXPECT_EQ(c.imu.driver.i2c_addr, 0x6A);
  EXPECT_EQ(c.imu.driver.expected_who_am_i, 0x6C);
  EXPECT_EQ(c.imu.driver.clock.method, ClockFitMethod::kLowerEnvelope);
  EXPECT_DOUBLE_EQ(c.imu.noise.accel_noise_density, 7.845e-4);  // T1 F9: 80 ug/sqrt(Hz) at +-8 g
  EXPECT_EQ(c.imu.thread.cpus, std::vector<int>{5});  // own core (R1 F18)
  EXPECT_EQ(c.imu.queue_samples, 16384u);
  ASSERT_EQ(c.virtual_cameras.count("uw"), 1u);
  EXPECT_EQ(c.virtual_cameras.at("uw").model.kind, CameraModelKind::kVirtualPinhole);
  EXPECT_EQ(c.virtual_cameras.at("uw").model.provenance, Provenance::kMeasured);
  EXPECT_DOUBLE_EQ(c.live_checks.main_fps_min, 31.0);
  EXPECT_EQ(c.slam_live.thread.cpus, (std::vector<int>{3, 4}));
  EXPECT_EQ(c.viz.mode, VizMode::kBoth);
  EXPECT_EQ(c.shm.prefix, "/rig");
  EXPECT_DOUBLE_EQ(c.calibration.T_uw_tcp.translation_m.z(), 0.15);
}

TEST(RigConfig, CanonicalYamlRoundTripsAndHashIsStable) {
  auto r = load_rig_config(kExample);
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  const std::string y = to_yaml(*r);
  auto r2 = parse_rig_config(y, "canonical");
  ASSERT_TRUE(r2.ok()) << r2.status().to_string() << "\n" << y;
  EXPECT_EQ(to_yaml(*r2), y);
  EXPECT_EQ(content_hash(*r2), content_hash(*r));
  EXPECT_EQ(content_hash(*r), sha256_hex(y));
  EXPECT_EQ(content_hash(*r).size(), 64u);

  // Comments, key order and number spelling do not change the hash; a value does.
  YAML::Node n = YAML::Load(read_file(kExample));
  YAML::Node reordered;
  std::vector<std::string> keys;
  for (auto it = n.begin(); it != n.end(); ++it) keys.push_back(it->first.as<std::string>());
  for (auto it = keys.rbegin(); it != keys.rend(); ++it) reordered[*it] = n[*it];
  reordered["imu"]["driver"]["i2c_addr"] = YAML::Load("106");  // 0x6A in decimal
  auto r3 = parse_rig_config(YAML::Dump(reordered), "reordered");
  ASSERT_TRUE(r3.ok()) << r3.status().to_string();
  EXPECT_EQ(content_hash(*r3), content_hash(*r));
  r3.value().cameras.at("uw").model.fx += 1e-9;
  EXPECT_NE(content_hash(*r3), content_hash(*r));
  // source_path is not part of the content.
  r3.value().cameras.at("uw").model.fx = r->cameras.at("uw").model.fx;
  r3.value().source_path = "elsewhere";
  EXPECT_EQ(content_hash(*r3), content_hash(*r));
}

TEST(RigConfig, StringEscapesRoundTrip) {
  auto r = load_rig_config(kExample);
  ASSERT_TRUE(r.ok());
  RigConfig c = *r;
  c.notes = "quote \" backslash \\ newline\n tab\t colon: # hash, unicode \xC3\xA9 ctrl \x01";
  auto r2 = parse_rig_config(to_yaml(c));
  ASSERT_TRUE(r2.ok()) << r2.status().to_string();
  EXPECT_EQ(r2->notes, c.notes);
}

TEST(RigConfig, MissingFileAndSyntaxError) {
  auto r = load_rig_config("/nonexistent/rig.yaml");
  EXPECT_EQ(r.status().code(), StatusCode::kNotFound);
  auto s = parse_rig_config("a: [1, 2\nb: 3\n", "bad.yaml");
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.status().code(), StatusCode::kInvalidArgument);
  EXPECT_NE(s.status().message().find("YAML syntax error"), std::string::npos) << s.status().message();
  EXPECT_NE(s.status().message().find("bad.yaml:"), std::string::npos);
  auto e = parse_rig_config("", "empty.yaml");
  ASSERT_FALSE(e.ok());
  EXPECT_NE(e.status().message().find("expected a mapping"), std::string::npos) << e.status().message();
}

// ---------------------------------------------------------------------------------------------------------------
// Error paths: mutate the example and check the first error's path and text.
// ---------------------------------------------------------------------------------------------------------------

struct Mutation {
  std::string path;   ///< dotted path; last element is the key to set/remove
  std::string value;  ///< YAML text to set; "<remove>" removes the key
  std::string expect_prefix;  ///< error message must start with this (the YAML path)
  std::string expect_text;    ///< ... and contain this
};

std::string apply(const Mutation& m) {
  YAML::Node root = YAML::Load(read_file(kExample));
  std::vector<std::string> parts;
  std::stringstream ss(m.path);
  for (std::string p; std::getline(ss, p, '.');) parts.push_back(p);
  YAML::Node parent = root;
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) parent.reset(parent[parts[i]]);
  if (m.value == "<remove>") {
    parent.remove(parts.back());
  } else {
    parent[parts.back()] = YAML::Load(m.value);
  }
  return YAML::Dump(root);
}

void PrintTo(const Mutation& m, std::ostream* os) { *os << m.path << " = " << m.value; }

class RigConfigErrors : public ::testing::TestWithParam<Mutation> {};

TEST_P(RigConfigErrors, ReportsPathAndReason) {
  const Mutation& m = GetParam();
  const std::string text = apply(m);
  auto r = parse_rig_config(text, "mut.yaml");
  ASSERT_FALSE(r.ok()) << "accepted: " << m.path << " = " << m.value;
  EXPECT_EQ(r.status().code(), StatusCode::kInvalidArgument);
  const std::string& msg = r.status().message();
  EXPECT_EQ(msg.rfind(m.expect_prefix, 0), 0u) << msg;
  EXPECT_NE(msg.find(m.expect_text), std::string::npos) << msg;
  EXPECT_NE(msg.find("(mut.yaml:"), std::string::npos) << msg;  // line reference
}

const Mutation kMutations[] = {
    // Top level / structure
    {"schema_version", "2", "schema_version:", "not supported"},
    {"schema_version", "<remove>", "schema_version:", "missing required key"},
    {"schema_version", "1.5", "schema_version:", "expected an integer"},
    {"rig_name", "\"\"", "rig_name:", "must not be empty"},
    {"rig_name", "[a]", "rig_name:", "expected a scalar"},
    {"bogus_key", "1", "bogus_key:", "unknown key"},
    {"cameras.main.driver.bogus", "1", "cameras.main.driver.bogus:", "unknown key"},
    {"cameras", "<remove>", "cameras:", "missing required key"},
    {"cameras", "[1, 2]", "cameras:", "expected a mapping"},
    {"imu", "<remove>", "imu:", "missing required key"},
    {"cameras.Main", "{}", "cameras.Main:", "[a-z0-9_]+"},
    {"cameras.averyveryverylongid", "{}", "cameras.averyveryverylongid:", "<= 15"},
    // Roles
    {"cameras.uw.role", "main", "cameras:", "exactly one camera with role main"},
    {"cameras.uw.role", "wide", "cameras.uw.role:", "not one of main | ultrawide"},
    {"cameras.uw.role", "<remove>", "cameras.uw.role:", "missing required key"},
    // Driver
    {"cameras.uw.driver.backend", "v4l2", "cameras.uw.driver.backend:", "not one of argus | gst | mock | replay"},
    {"cameras.uw.driver.sensor", "imx477", "cameras.uw.driver.sensor:", "not one of imx519 | imx219"},
    {"cameras.uw.driver.width", "1280", "cameras.uw.driver:", "not a mode of imx219"},
    {"cameras.uw.driver.fps", "60", "cameras.uw.driver:", "runs at 1.46"},
    {"cameras.uw.driver.fps", "0", "cameras.uw.driver.fps:", "out of range (0, 240]"},
    {"cameras.uw.driver.fps", "abc", "cameras.uw.driver.fps:", "expected a finite number"},
    {"cameras.uw.driver.fps", ".nan", "cameras.uw.driver.fps:", "expected a finite number"},
    {"cameras.main.driver.width", "4656", "cameras.main.driver:", "4656x1080@60 is not a mode of imx519"},
    {"cameras.uw.driver.pixel_format", "jpeg", "cameras.uw.driver.pixel_format:", "not one of nv12"},
    {"cameras.uw.driver.exposure_ns", "40000000", "cameras.uw.driver.exposure_ns:", "longer than the frame duration"},
    {"cameras.uw.driver.exposure_ns", "-1", "cameras.uw.driver.exposure_ns:", "out of range"},
    {"cameras.uw.driver.analog_gain", "0.5", "cameras.uw.driver.analog_gain:", "0 (free) or >= 1"},
    {"cameras.uw.driver.isp_digital_gain", "0.5", "cameras.uw.driver.isp_digital_gain:", "out of range"},
    {"cameras.uw.driver.ae_lock", "yes", "cameras.uw.driver.ae_lock:", "expected true or false"},
    {"cameras.uw.driver.awb_mode", "sunny", "cameras.uw.driver.awb_mode:", "not one of off | auto"},
    {"cameras.uw.driver.buffer_count", "1", "cameras.uw.driver.buffer_count:", "out of range"},
    {"cameras.uw.driver.drop_gap_factor", "1.0", "cameras.uw.driver.drop_gap_factor:", "out of range (1, 10]"},
    {"cameras.uw.driver.sensor_id", "0", "cameras.uw.driver.sensor_id:", "also used by camera 'main'"},
    {"cameras.main.driver.focus.position", "-1", "cameras.main.driver.focus.position:", "needs a lens position"},
    {"cameras.main.driver.focus.mode", "auto", "cameras.main.driver.focus.mode:", "not one of none | fixed"},
    {"cameras.main.driver.focus.i2c_addr", "0x80", "cameras.main.driver.focus.i2c_addr:", "out of range"},
    {"cameras.main.driver.focus.v4l2_subdev", "~", "cameras.main.driver.focus.v4l2_subdev:", "got null"},
    {"cameras.uw.driver.mock.pattern", "stripes", "cameras.uw.driver.mock.pattern:", "not one of checker"},
    {"cameras.uw.driver.mock.jitter_ns", "20000000", "cameras.uw.driver.mock.jitter_ns:", "half the frame period"},
    {"cameras.uw.driver.mock.drop_every_n", "1", "cameras.uw.driver.mock.drop_every_n:", "drop every frame"},
    {"cameras.uw.driver.mock.seed", "-3", "cameras.uw.driver.mock.seed:", "non-negative integer"},
    {"cameras.uw.driver.replay.loop", "1", "cameras.uw.driver.replay.loop:", "expected true or false"},
    {"cameras.uw.driver.backend", "replay", "cameras.uw.driver.replay:", "needs replay.path"},
    // Model
    {"cameras.uw.model.width", "1600", "cameras.uw.model:", "!= driver capture size"},
    {"cameras.uw.model.kind", "fisheye", "cameras.uw.model.kind:", "not one of pinhole-radtan"},
    {"cameras.uw.model.kind", "pinhole", "cameras.uw.model.distortion:", "pinhole needs 0 parameters"},
    {"cameras.uw.model.distortion", "[0.0, 0.0]", "cameras.uw.model.distortion:", "needs 4 parameters, got 2"},
    {"cameras.uw.model.distortion", "<remove>", "cameras.uw.model.distortion:", "missing required key"},
    {"cameras.uw.model.fx", "-1", "cameras.uw.model.fx:", "out of range"},
    {"cameras.uw.model.cx", "5000", "cameras.uw.model.cx:", "outside the image"},
    {"cameras.uw.model.fov_max_deg", "360", "cameras.uw.model.fov_max_deg:", "out of range [0, 360)"},
    {"cameras.uw.model.provenance", "guess", "cameras.uw.model.provenance:", "not one of nominal | cad"},
    {"cameras.uw.model", "<remove>", "cameras.uw.model:", "missing required key"},
    // Extrinsics
    {"cameras.uw.T_imu_cam.rotation", "[[1, 0, 0], [0, 1, 0], [0, 0, 2]]", "cameras.uw.T_imu_cam.rotation:",
     "not orthonormal"},
    {"cameras.uw.T_imu_cam.rotation", "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]", "cameras.uw.T_imu_cam.rotation:",
     "determinant"},
    {"cameras.uw.T_imu_cam.rotation", "[[1, 0, 0], [0, 1, 0]]", "cameras.uw.T_imu_cam.rotation:", "3 rows"},
    {"cameras.uw.T_imu_cam.translation_m", "[0, 0]", "cameras.uw.T_imu_cam.translation_m:", "list of 3 numbers"},
    {"cameras.uw.T_imu_cam.translation_m", "[0, 0, 50]", "cameras.uw.T_imu_cam.translation_m:", "wrong unit"},
    {"cameras.uw.time.offset_ns", "2000000000", "cameras.uw.time.offset_ns:", "out of range"},
    // Storage
    {"cameras.main.storage.width", "1000", "cameras.main.storage:", "aspect ratio"},
    {"cameras.main.storage.height", "0", "cameras.main.storage:", "both be 0"},
    {"cameras.main.storage.width", "3840", "cameras.main.storage:", "larger than the capture"},
    {"cameras.main.storage.jpeg_quality", "101", "cameras.main.storage.jpeg_quality:", "out of range [1, 100]"},
    {"cameras.main.storage.queue_frames", "0", "cameras.main.storage.queue_frames:", "out of range"},
    // Threads
    {"cameras.main.thread.cpus", "[1, 1]", "cameras.main.thread.cpus:", "duplicate CPU id"},
    {"cameras.main.thread.cpus", "[2000]", "cameras.main.thread.cpus:", "out of range"},
    {"cameras.main.thread.priority", "100", "cameras.main.thread.priority:", "out of range [0, 99]"},
    {"cameras.main.thread.nice", "5", "cameras.main.thread.nice:", "SCHED_OTHER only"},
    {"imu.thread.nice", "-21", "imu.thread.nice:", "out of range [-20, 19]"},
    {"imu.thread.cpus", "[1]", "imu.thread.cpus:", "CPU 1 is shared with cameras.main.thread.cpus"},
    {"imu.thread.cpus", "[5, 2]", "imu.thread.cpus:", "CPU 2 is shared with cameras.uw.thread.cpus"},
    // IMU
    {"imu.driver.backend", "iio", "imu.driver.backend:", "not one of lsm6dsox_i2c | sim | replay"},
    {"imu.driver.odr_hz", "800", "imu.driver.odr_hz:", "not one of 12.5"},
    {"imu.driver.accel_fs_g", "3", "imu.driver.accel_fs_g:", "not one of 2, 4, 8, 16"},
    {"imu.driver.gyro_fs_dps", "300", "imu.driver.gyro_fs_dps:", "not one of 125"},
    {"imu.driver.fifo_watermark_words", "512", "imu.driver.fifo_watermark_words:", "out of range [1, 511]"},
    {"imu.driver.poll_period_ms", "200", "imu.driver.poll_period_ms:", "FIFO words per poll"},
    {"imu.driver.i2c_addr", "0x6G", "imu.driver.i2c_addr:", "expected an integer"},
    {"imu.driver.id", "IMU", "imu.driver.id:", "[a-z0-9_]+"},
    {"imu.driver.gap_factor", "1", "imu.driver.gap_factor:", "out of range"},
    {"imu.driver.timestamp_decimation", "4", "imu.driver.timestamp_decimation:", "not one of 1, 8, 32"},
    {"imu.driver.temperature_batch_hz", "10", "imu.driver.temperature_batch_hz:", "not one of 0, 1.6, 12.5, 52"},
    {"imu.driver.fifo_burst_words", "0", "imu.driver.fifo_burst_words:", "out of range [1, 256]"},
    {"imu.driver.fifo_burst_words", "257", "imu.driver.fifo_burst_words:", "out of range [1, 256]"},
    {"imu.driver.pair_by_tag_counter", "1", "imu.driver.pair_by_tag_counter:", "expected true or false"},
    {"imu.driver.clock.outlier_gate_ns", "-1", "imu.driver.clock.outlier_gate_ns:", "out of range"},
    {"imu.driver.clock.method", "kalman", "imu.driver.clock.method:", "not one of lower_envelope"},
    {"imu.driver.clock.max_slope_deviation", "0", "imu.driver.clock.max_slope_deviation:", "out of range"},
    {"imu.driver.clock.window_brackets", "1", "imu.driver.clock.window_brackets:", "out of range"},
    {"imu.driver.sim.i2c_error_rate", "1.5", "imu.driver.sim.i2c_error_rate:", "out of range [0, 1]"},
    {"imu.driver.backend", "replay", "imu.driver.replay:", "needs replay.path"},
    {"imu.driver", "<remove>", "imu.driver:", "missing required key"},
    {"imu.noise.gyro_noise_density", "0", "imu.noise.gyro_noise_density:", "out of range"},
    {"imu.noise.gyro_noise_density", "<remove>", "imu.noise.gyro_noise_density:", "missing required key"},
    {"imu.noise", "<remove>", "imu.noise:", "missing required key"},
    {"imu.queue_samples", "0", "imu.queue_samples:", "out of range"},
    // Virtual cameras
    {"virtual_cameras.uw.model.kind", "kannala-brandt4", "virtual_cameras.uw.model.distortion:", "needs 4"},
    {"virtual_cameras.side", "{model: {kind: pinhole, width: 10, height: 10, fx: 1, fy: 1, cx: 5, cy: 5, distortion: []}}",
     "virtual_cameras.side:", "no physical camera 'side'"},
    {"virtual_cameras.uw.min_coverage", "1.2", "virtual_cameras.uw.min_coverage:", "out of range [0, 1]"},
    // Other sections
    {"imageprep.out_size", "8", "imageprep.out_size:", "out of range"},
    {"capture.steady_state.rate_tolerance", "0", "capture.steady_state.rate_tolerance:", "out of range (0, 1)"},
    {"capture.steady_state.min_frames", "0", "capture.steady_state.min_frames:", "out of range"},
    {"capture.read_timeout_ms", "0", "capture.read_timeout_ms:", "out of range"},
    {"live_checks.main_camera", "front", "live_checks.main_camera:", "no camera 'front'"},
    {"live_checks.main_fps_min", "61", "live_checks.main_fps_min:", "every frame would abort"},
    {"live_checks.uw_fps_min", "30", "live_checks.uw_fps_min:", "every window would abort"},
    {"live_checks.imu_gap_max_ms", "0.5", "live_checks.imu_gap_max_ms:", "shorter than one IMU sample period"},
    {"live_checks.imu_rate_min_ratio", "1.5", "live_checks.imu_rate_min_ratio:", "out of range"},
    {"live_checks.thermal_zones", "[\"\"]", "live_checks.thermal_zones:", "empty zone name"},
    {"live_checks.thermal_zones", "cpu", "live_checks.thermal_zones:", "expected a list of strings"},
    {"record.arming_timeout_s", "0", "record.arming_timeout_s:", "out of range"},
    {"slam_live.backend", "openvins", "slam_live.backend:", "not one of orbslam3 | mock"},
    {"slam_live.camera", "side", "slam_live.camera:", "no camera 'side'"},
    {"slam_live.input_width", "2000", "slam_live.input_width:", "larger than the camera"},
    {"slam_live.vocabulary_path", "\"\"", "slam_live.vocabulary_path:", "needs the ORB vocabulary"},
    {"slam_live.frame_decimation", "0", "slam_live.frame_decimation:", "out of range"},
    {"viz.mode", "all", "viz.mode:", "not one of none | cameras | slam | both"},
    {"shm.prefix", "rig", "shm.prefix:", "must be '/'"},
    {"shm.prefix", "/rig/x", "shm.prefix:", "must be '/'"},
    {"shm.frame_ring_slots", "1", "shm.frame_ring_slots:", "out of range"},
    {"shm.viz_frame_width", "641", "shm.viz_frame_width:", "even width"},
    {"storage.data_root", "\"\"", "storage.data_root:", "must not be empty"},
    {"calibration.date", "2026/10/01", "calibration.date:", "YYYY-MM-DD"},
    {"calibration.T_uw_tcp.provenance", "x", "calibration.T_uw_tcp.provenance:", "not one of"},
};

INSTANTIATE_TEST_SUITE_P(All, RigConfigErrors, ::testing::ValuesIn(kMutations));

TEST(RigConfig, CroppedUltrawideModeMessage) {
  // PLAN T3: "cameras.uw...: 1280x720@60 is a cropped mode on imx219; full-FOV modes are ...".
  YAML::Node root = YAML::Load(read_file(kExample));
  root["cameras"]["uw"]["driver"]["width"] = 1280;
  root["cameras"]["uw"]["driver"]["height"] = 720;
  root["cameras"]["uw"]["driver"]["fps"] = 60.0;
  root["cameras"]["uw"]["model"]["width"] = 1280;
  root["cameras"]["uw"]["model"]["height"] = 720;
  root["cameras"]["uw"]["model"]["cx"] = 639.5;
  root["cameras"]["uw"]["model"]["cy"] = 359.5;
  root["cameras"]["uw"]["storage"]["width"] = 0;
  root["live_checks"]["uw_fps_min"] = 27.0;
  auto r = parse_rig_config(YAML::Dump(root), "crop.yaml");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().message().rfind("cameras.uw.driver: 1280x720@60 is a cropped mode on imx219; full-FOV modes "
                                       "are 3280x2464@21, 1640x1232@30",
                                       0),
            0u)
      << r.status().message();
}

TEST(RigConfig, MainMustCoverTheFull169FieldUnlessForced) {
  // R1 F15: IMX519 1280x720 is a 2/3 crop of the 1080p field (T2 sensor_modes.hpp).
  YAML::Node root = YAML::Load(read_file(kExample));
  auto d = root["cameras"]["main"]["driver"];
  d["width"] = 1280;
  d["height"] = 720;
  auto mdl = root["cameras"]["main"]["model"];
  mdl["width"] = 1280;
  mdl["height"] = 720;
  mdl["cx"] = 639.5;
  mdl["cy"] = 359.5;
  root["cameras"]["main"]["storage"]["width"] = 0;
  root["cameras"]["main"]["storage"]["height"] = 0;
  auto r = parse_rig_config(YAML::Dump(root), "main720.yaml");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().message().rfind("cameras.main.driver: 1280x720@60 is a crop of the full field on imx519", 0), 0u)
      << r.status().message();
  EXPECT_NE(r.status().message().find("set sensor_mode to force it"), std::string::npos);
  d["sensor_mode"] = 3;
  auto forced = parse_rig_config(YAML::Dump(root), "main720.yaml");
  EXPECT_TRUE(forced.ok()) << forced.status().to_string();
  // 1920x1080 (2x2 binned 16:9 field) and the full 4656x3496 array are fine for the main camera.
  EXPECT_TRUE(load_rig_config(kExample).ok());
}

TEST(RigConfig, NewImuKeysAreReadHashedAndBudgeted) {
  auto base = load_rig_config(kExample).value();
  EXPECT_EQ(base.imu.driver.timestamp_decimation, 1);
  EXPECT_DOUBLE_EQ(base.imu.driver.temperature_batch_hz, 12.5);
  EXPECT_EQ(base.imu.driver.fifo_burst_words, 32);
  EXPECT_TRUE(base.imu.driver.pair_by_tag_counter);
  EXPECT_EQ(base.imu.driver.clock.outlier_gate_ns, 1000000);
  for (auto mut : std::vector<std::function<void(RigConfig&)>>{
           [](RigConfig& c) { c.imu.driver.timestamp_decimation = 8; },
           [](RigConfig& c) { c.imu.driver.temperature_batch_hz = 52; },
           [](RigConfig& c) { c.imu.driver.fifo_burst_words = 16; },
           [](RigConfig& c) { c.imu.driver.pair_by_tag_counter = false; },
           [](RigConfig& c) { c.imu.driver.clock.outlier_gate_ns = 0; }}) {
    RigConfig c = base;
    mut(c);
    EXPECT_NE(content_hash(c), content_hash(base));
    auto back = parse_rig_config(to_yaml(c));
    ASSERT_TRUE(back.ok()) << back.status().to_string();
    EXPECT_EQ(content_hash(*back), content_hash(c));
  }
  // R1 F8: 3332 Hz x 30 ms = 200 words with 2 words/ODR (passes the old rule) but 300 with the timestamp word.
  YAML::Node root = YAML::Load(read_file(kExample));
  root["imu"]["driver"]["odr_hz"] = 3332;
  root["imu"]["driver"]["poll_period_ms"] = 30;
  root["live_checks"]["imu_gap_max_ms"] = 20.0;
  auto r = parse_rig_config(YAML::Dump(root), "budget.yaml");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.status().message().find("imu.driver.poll_period_ms"), std::string::npos) << r.status().message();
  root["imu"]["driver"]["timestamp_decimation"] = 32;
  root["imu"]["driver"]["temperature_batch_hz"] = 0;
  auto ok = parse_rig_config(YAML::Dump(root), "budget.yaml");  // 203 words: within budget
  EXPECT_TRUE(ok.ok()) << ok.status().to_string();
}

TEST(RigConfig, QuotedNumbersAndBooleansAreTypeErrors) {
  std::string text = read_file(kExample);
  const std::string from = "schema_version: 1 ";
  ASSERT_NE(text.find(from), std::string::npos);
  text.replace(text.find(from), from.size(), "schema_version: \"1\" ");
  auto r = parse_rig_config(text, "q.yaml");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().message().rfind("schema_version: expected an integer (decimal or 0x hex), got the quoted string '1'", 0), 0u)
      << r.status().message();
  text = read_file(kExample);
  const std::string f2 = "fsync_on_stop: true";
  text.replace(text.find(f2), f2.size(), "fsync_on_stop: 'true'");
  auto r2 = parse_rig_config(text, "q.yaml");
  ASSERT_FALSE(r2.ok());
  EXPECT_EQ(r2.status().message().rfind("storage.fsync_on_stop: expected true or false", 0), 0u) << r2.status().message();
}

TEST(RigConfig, CollectsAllErrorsAndRejectsDuplicates) {
  std::string text = read_file(kExample);
  text += "\nrig_name: again\n";  // duplicate top-level key
  auto r = parse_rig_config(text, "dup.yaml");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.status().message().find("rig_name: duplicate key"), std::string::npos) << r.status().message();

  YAML::Node root = YAML::Load(read_file(kExample));
  root["imageprep"]["out_size"] = 1;
  root["viz"]["refresh_hz"] = -1;
  auto r2 = parse_rig_config(YAML::Dump(root), "two.yaml");
  ASSERT_FALSE(r2.ok());
  EXPECT_NE(r2.status().message().find("1 more: "), std::string::npos) << r2.status().message();
  EXPECT_NE(r2.status().message().find("viz.refresh_hz"), std::string::npos);
}

TEST(RigConfig, MockBackendsSkipSensorModeTables) {
  YAML::Node root = YAML::Load(read_file(kExample));
  for (const char* id : {"main", "uw"}) {
    auto d = root["cameras"][id]["driver"];
    d["backend"] = "mock";
    d["width"] = 320;
    d["height"] = 240;
    auto mdl = root["cameras"][id]["model"];
    mdl["width"] = 320;
    mdl["height"] = 240;
    mdl["cx"] = 159.5;
    mdl["cy"] = 119.5;
    root["cameras"][id]["storage"]["width"] = 0;
    root["cameras"][id]["storage"]["height"] = 0;
  }
  root["slam_live"]["input_width"] = 320;
  root["slam_live"]["input_height"] = 240;
  auto r = parse_rig_config(YAML::Dump(root), "mock.yaml");
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  EXPECT_EQ(r->cameras.at("uw").driver.backend, CameraBackendKind::kMock);
}

TEST(RigConfig, DefaultsApplyToOmittedOptionalSections) {
  YAML::Node root = YAML::Load(read_file(kExample));
  for (const char* k : {"notes", "imageprep", "capture", "live_checks", "record", "viz", "shm", "storage", "calibration",
                        "virtual_cameras"})
    root.remove(k);
  root["slam_live"].remove("thread");
  auto r = parse_rig_config(YAML::Dump(root), "min.yaml");
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  EXPECT_EQ(r->imageprep.out_size, 224);
  EXPECT_DOUBLE_EQ(r->live_checks.pose_speed_max_mps, 7.0);
  EXPECT_EQ(r->shm.imu_ring_slots, 8192u);
  EXPECT_TRUE(r->virtual_cameras.empty());
}

}  // namespace
}  // namespace rig
