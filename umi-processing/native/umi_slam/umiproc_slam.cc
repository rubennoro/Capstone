// umiproc_slam: run UMI's ORB-SLAM3 fork (github.com/cheng-chi/ORB_SLAM3 at b741dca + umi-processing/native/patches)
// on a umiproc EuRoC folder.
//
// The fork's own front-end (Examples/Monocular-Inertial/gopro_slam.cc) reads a GoPro MP4 + IMU JSON and assumes a
// constant frame rate. This one reads the EuRoC folder that umiproc's copied write_euroc writes:
//   SEQ/times.txt            one camera time in ns per line (the frames to feed, in order)
//   SEQ/mav0/cam0/data/<t>.png  gray frames at the settings size
//   SEQ/mav0/imu0/data.csv   t_ns,gx,gy,gz,ax,ay,az (rad/s, m/s^2), same time base as the camera
// and calls the fork's System::LocalizeMonocular like gopro_slam does, so both fork workflows are available:
//   * mapping:      --save_map MAP.osa  (normal SLAM; the atlas is written at shutdown)
//   * localization: --load_map MAP.osa  (the fork relocalizes against the loaded map instead of initialising a new
//                   one, and relocalizes again instead of resetting when tracking fails)
//
// Times: the fork's map loading shifts the map's keyframe times to end just before t = 0, so demo times must start
// at 0 (gopro_slam feeds frame_idx / fps). Every time here is (t_ns - t0_ns) * 1e-9 with t0_ns = the first line of
// times.txt; t0_ns is written to OUT_CSV + ".t0_ns" so the CSV's seconds can be mapped back.
//
// IMU feeding follows upstream Examples/Monocular-Inertial/mono_inertial_euroc.cc: start one sample before the first
// frame; for every frame after the first, pass the samples with t <= the frame time.
//
// Gripper mask (UMI's mechanism, gopro_slam.cc): --mask_img PNG, nonzero = masked; those pixels are set to 0 in every
// frame fed to ORB-SLAM3 (im.setTo(0, mask)). Use it for mapping and for every localization run.
//
// Pacing: frames are fed at the recording's own rate by default (as upstream mono_inertial_euroc), because ORB-SLAM3
// is built for real-time input: Tracking skips keyframes while LocalMapping is busy, so feeding faster than real time
// changes what the map gets. --no_realtime feeds as fast as possible (faster, fewer keyframes, less repeatable).
// Whether the machine keeps up (the Jetson) is in the summary's "pacing" object:
//   track_ms {median, p95, max}  per-frame tracking time: the IMU batch + LocalizeMonocular, in ms (p95 = nearest
//                                rank); the frame read and the mask are not in it
//   frames_over_period           frames whose tracking time exceeded their frame period (the time to the next
//                                frame; for the last frame, the time since the previous one): tracking fell behind
//   feed_rate_hz                 frames fed / wall time from the first frame read to the last LocalizeMonocular
//                                return (includes the frame reads and the waits)
//   recording_rate_hz            the fed frames' own rate: (frames fed - 1) / (their last - first time)
//   feed_wall_s                  that wall time
// "pacing" is null when no frame was fed.
//
// Input checks (exit 1 instead of a silently wrong run): camera and IMU times strictly increasing, no camera gap
// > 1 s (ORB-SLAM3 treats that as a new sequence and resets or starts a new map), the IMU spans every frame, and
// frames / mask with the settings' aspect ratio.
//
// Output: SaveTrajectoryCSV (frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w): per-frame
// T_W_cam, W = the first keyframe of the biggest map. Row k is line k of times.txt: join on frame_idx, not on the
// timestamp column (garbage before the first tracked frame, repeated on lost frames, 6 decimals). state 2 = OK
// (visually tracked), 3 = RECENTLY_LOST (is_lost=false, but an IMU-only prediction); a map reset marks its rows
// is_lost=true without changing their state, so "tracked" is state 2 and is_lost false.
//
// Last line on stdout: "UMIPROC_SUMMARY {json}" (frames fed, lost frames, whether the run stopped early, the maps from
// System::MapSummary(), t0_ns, the pacing, the options used); umiproc/slam/log.py parses it. Two lost counts:
//   lost_frames        state != OK (2) after the frame: not visually tracked (umiproc's definition of lost)
//   not_tracking_frames LocalizeMonocular returned false (state neither OK nor RECENTLY_LOST): what --max_lost_frames
//                      counts, as gopro_slam does
// Map export (--save_points CSV, for umiproc slam-3d): after Shutdown, the map SaveTrajectoryCSV picks (the first
// with the most keyframes) in the trajectory CSV's world W (the camera frame of that map's first keyframe):
//   CSV                       id,x,y,z,observations                         every map point that is not bad
//   <CSV stem>_keyframes.csv  id,frame_id,timestamp,x,y,z,q_x,q_y,q_z,q_w   T_W_cam of every keyframe not bad
//   <CSV stem>_meta.json      the map id, the counts, imu_initialized, up_W (ORB-SLAM3's gravity-aligned +z in W
//                             once the IMU is initialised, else null) and T_W_orbworld (row-major 4x4)
// No map with a keyframe: the two CSVs are header-only and "map_id" is null. The library does not expose the atlas,
// so it is read through System::mpAtlas with the explicit-instantiation access idiom (no library change).
// --max_frames N feeds only the first N frames (0 with --load_map and --save_points: export a saved map as it is).
//
// Exit 0; 3 when --max_lost_frames was reached (the trajectory is still saved, unlike gopro_slam); 1 on bad input or
// an unreadable frame. On a non-zero exit the --save_map file is removed, so a failed mapping run never leaves a map
// behind.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI11.hpp>
#include <System.h>

namespace {

// System::mpAtlas is private and the library has no accessor; an explicit instantiation may name a private member
// ([temp.explicit]), which this uses to read it without changing (and rebuilding) libORB_SLAM3.
template <typename Tag, typename Tag::type M>
struct PrivateAccess {
    friend typename Tag::type Get(Tag) { return M; }
};
struct SystemAtlas {
    using type = ORB_SLAM3::Atlas *ORB_SLAM3::System::*;
    friend type Get(SystemAtlas);
};
template struct PrivateAccess<SystemAtlas, &ORB_SLAM3::System::mpAtlas>;

// "path/x.csv" + "_keyframes.csv" -> "path/x_keyframes.csv" (a path without ".csv" keeps its whole name)
std::string Sibling(const std::string &csv, const std::string &suffix) {
    const std::string ext = ".csv";
    const bool has_ext = csv.size() > ext.size() && csv.compare(csv.size() - ext.size(), ext.size(), ext) == 0;
    return (has_ext ? csv.substr(0, csv.size() - ext.size()) : csv) + suffix;
}

bool LoadTimes(const std::string &path, std::vector<int64_t> &t_ns) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        t_ns.push_back(std::stoll(line));
    }
    return !t_ns.empty();
}

bool LoadImu(const std::string &path, std::vector<int64_t> &t_ns, std::vector<cv::Point3f> &acc,
             std::vector<cv::Point3f> &gyr) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> v;
        int64_t t = 0;
        for (int k = 0; std::getline(ss, item, ','); ++k) {
            if (k == 0) t = std::stoll(item);
            else v.push_back(std::stod(item));
        }
        if (v.size() != 6) {
            std::cerr << "bad IMU line (want t,gx,gy,gz,ax,ay,az): " << line << std::endl;
            return false;
        }
        t_ns.push_back(t);
        gyr.emplace_back(v[0], v[1], v[2]);
        acc.emplace_back(v[3], v[4], v[5]);
    }
    return !t_ns.empty();
}

// "" when ok, else what is wrong (see the header comment).
std::string CheckTimes(const std::vector<int64_t> &t_cam, const std::vector<int64_t> &t_imu) {
    const int64_t kMaxCamGapNs = 1000000000LL;     // Tracking.cc: a > 1 s jump is treated as a new sequence
    for (size_t i = 1; i < t_cam.size(); ++i) {
        if (t_cam[i] <= t_cam[i - 1]) return "camera times not strictly increasing at line " + std::to_string(i + 1);
        if (t_cam[i] - t_cam[i - 1] > kMaxCamGapNs) return "camera gap > 1 s at line " + std::to_string(i + 1);
    }
    for (size_t i = 1; i < t_imu.size(); ++i)
        if (t_imu[i] <= t_imu[i - 1]) return "IMU times not strictly increasing at sample " + std::to_string(i + 1);
    if (t_imu.front() > t_cam.front() || t_imu.back() < t_cam.back())
        return "the IMU does not span every frame (IMU " + std::to_string(t_imu.front()) + ".." +
               std::to_string(t_imu.back()) + " ns, frames " + std::to_string(t_cam.front()) + ".." +
               std::to_string(t_cam.back()) + " ns)";
    return "";
}

bool SameAspect(const cv::Size &a, const cv::Size &b) {
    return static_cast<int64_t>(a.width) * b.height == static_cast<int64_t>(b.width) * a.height;
}

std::string JsonString(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (const char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
                else
                    o << c;
        }
    }
    o << '"';
    return o.str();
}

const char *JsonBool(bool b) { return b ? "true" : "false"; }

// What the run did, printed as the last stdout line (see the header comment).
struct Summary {
    int exit_code = 0;
    std::string error;                      // "" when none
    size_t frames_total = 0;                // lines in times.txt
    size_t frames_fed = 0;
    int lost_frames = 0;                    // state != OK
    int not_tracking_frames = 0;            // LocalizeMonocular false (what --max_lost_frames counts)
    bool stopped_early = false;             // not every frame was fed
    std::string stop_reason;                // "max_lost_frames" | "unreadable_frame" | ""
    std::map<int, int> state_counts;        // tracking state after each fed frame -> frames
    bool have_maps = false;                 // the SLAM system ran (else "maps" is null)
    std::vector<ORB_SLAM3::System::MapInfo> maps;
    bool have_t0 = false;
    int64_t t0_ns = 0;
    bool map_removed = false;               // --save_map removed because of a non-zero exit
    std::vector<double> track_ms;           // per fed frame: IMU batch + LocalizeMonocular (see the header comment)
    int frames_over_period = 0;             // tracking took longer than the frame period
    double feed_wall_s = 0.0;               // first frame read .. last LocalizeMonocular return
    double recording_span_s = 0.0;          // last - first fed frame time
};

// Nearest-rank percentile (q in (0, 1]) of a non-empty sorted vector.
double Percentile(const std::vector<double> &sorted, double q) {
    size_t rank = static_cast<size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    if (rank < 1) rank = 1;
    return sorted[std::min(rank, sorted.size()) - 1];
}

double Median(const std::vector<double> &sorted) {
    const size_t n = sorted.size();
    return n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

struct Options {
    std::string vocabulary, setting, sequence, out_csv, load_map, save_map, mask_img, save_points;
    int max_lost_frames = -1;
    long max_frames = -1;            // < 0: every frame
    int aruco_dict_id = cv::aruco::DICT_4X4_50;
    int init_tag_id = -1;            // -1: never found, so the fork's tag-based initialisation is off
    float init_tag_size = 0.16f;
    bool no_realtime = false;
};

// The map export (see the header comment). "" when written, else what went wrong.
std::string SaveMapPoints(ORB_SLAM3::System &slam, const std::string &csv, size_t &n_points, size_t &n_keyframes) {
    n_points = n_keyframes = 0;
    ORB_SLAM3::Atlas *atlas = slam.*Get(SystemAtlas());
    ORB_SLAM3::Map *best = nullptr;               // as System::SaveTrajectoryCSV: the first with the most keyframes
    size_t best_kfs = 0;
    for (ORB_SLAM3::Map *m : atlas->GetAllMaps()) {
        const size_t k = m->GetAllKeyFrames().size();
        if (k > best_kfs) {
            best_kfs = k;
            best = m;
        }
    }
    const std::string kf_csv = Sibling(csv, "_keyframes.csv"), meta_json = Sibling(csv, "_meta.json");
    std::ofstream fp(csv), fk(kf_csv), fm(meta_json);
    if (!fp || !fk || !fm) return "cannot write " + csv + " / " + kf_csv + " / " + meta_json;
    fp << "id,x,y,z,observations\n";
    fk << "id,frame_id,timestamp,x,y,z,q_x,q_y,q_z,q_w\n";
    fp << std::setprecision(9);
    fk << std::setprecision(9);
    fm << std::setprecision(9);
    fm << "{\"format\":\"umiproc.map_points.v1\"";
    if (best == nullptr) {
        fm << ",\"map_id\":null,\"keyframes\":0,\"map_points\":0,\"bad_map_points\":0,\"imu_initialized\":false"
           << ",\"up_W\":null,\"T_W_orbworld\":null}\n";
    } else {
        std::vector<ORB_SLAM3::KeyFrame *> kfs = best->GetAllKeyFrames();
        std::sort(kfs.begin(), kfs.end(), ORB_SLAM3::KeyFrame::lId);
        const Sophus::SE3f T_W_orb = kfs[0]->GetPose();     // W = first keyframe's camera; Tcw maps ORB world -> it
        for (ORB_SLAM3::KeyFrame *kf : kfs) {
            if (kf == nullptr || kf->isBad()) continue;
            const Sophus::SE3f T = T_W_orb * kf->GetPoseInverse();
            const Eigen::Vector3f t = T.translation();
            const Eigen::Quaternionf q = T.unit_quaternion();
            fk << kf->mnId << ',' << kf->mnFrameId << ',' << kf->mTimeStamp << ',' << t.x() << ',' << t.y() << ','
               << t.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
            n_keyframes++;
        }
        size_t bad = 0;
        for (ORB_SLAM3::MapPoint *mp : best->GetAllMapPoints()) {
            if (mp == nullptr || mp->isBad()) {
                bad++;
                continue;
            }
            const Eigen::Vector3f p = T_W_orb * mp->GetWorldPos();
            fp << mp->mnId << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << mp->Observations() << '\n';
            n_points++;
        }
        const bool imu = best->isImuInitialized();
        fm << ",\"map_id\":" << best->GetId() << ",\"keyframes\":" << n_keyframes << ",\"map_points\":" << n_points
           << ",\"bad_map_points\":" << bad << ",\"first_keyframe_id\":" << kfs[0]->mnId
           << ",\"imu_initialized\":" << JsonBool(imu) << ",\"up_W\":";
        if (imu) {
            const Eigen::Vector3f up = T_W_orb.rotationMatrix() * Eigen::Vector3f(0.f, 0.f, 1.f);
            fm << '[' << up.x() << ',' << up.y() << ',' << up.z() << ']';
        } else {
            fm << "null";
        }
        const Eigen::Matrix4f M = T_W_orb.matrix();
        fm << ",\"T_W_orbworld\":[";
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) fm << (r || c ? "," : "") << M(r, c);
        fm << "]}\n";
    }
    fp.flush();
    fk.flush();
    fm.flush();
    if (!fp || !fk || !fm) return "cannot write " + csv + " / " + kf_csv + " / " + meta_json;
    return "";
}

void PrintSummary(const Summary &s, const Options &o) {
    std::ostringstream j;
    j << std::setprecision(9);
    j << "{\"format\":\"umiproc.slam_summary.v2\"";
    j << ",\"exit_code\":" << s.exit_code;
    j << ",\"error\":" << (s.error.empty() ? std::string("null") : JsonString(s.error));
    j << ",\"frames_total\":" << s.frames_total << ",\"frames_fed\":" << s.frames_fed;
    j << ",\"lost_frames\":" << s.lost_frames << ",\"not_tracking_frames\":" << s.not_tracking_frames;
    j << ",\"stopped_early\":" << JsonBool(s.stopped_early);
    j << ",\"stop_reason\":" << (s.stop_reason.empty() ? std::string("null") : JsonString(s.stop_reason));
    j << ",\"state_counts\":{";
    bool first = true;
    for (const auto &kv : s.state_counts) {
        j << (first ? "" : ",") << '"' << kv.first << "\":" << kv.second;
        first = false;
    }
    j << "}";
    j << ",\"maps\":";
    if (!s.have_maps) {
        j << "null";
    } else {
        j << "[";
        for (size_t i = 0; i < s.maps.size(); ++i) {
            const auto &m = s.maps[i];
            j << (i ? "," : "") << "{\"id\":" << m.id << ",\"keyframes\":" << m.keyframes
              << ",\"map_points\":" << m.map_points << ",\"imu_initialized\":" << JsonBool(m.imu_initialized)
              << ",\"inertial_ba1\":" << JsonBool(m.inertial_ba1) << ",\"inertial_ba2\":" << JsonBool(m.inertial_ba2)
              << "}";
        }
        j << "]";
    }
    j << ",\"t0_ns\":" << (s.have_t0 ? std::to_string(s.t0_ns) : std::string("null"));
    j << ",\"map_removed\":" << JsonBool(s.map_removed);
    j << ",\"pacing\":";
    if (s.track_ms.empty()) {
        j << "null";
    } else {
        std::vector<double> t(s.track_ms);
        std::sort(t.begin(), t.end());
        const double n = static_cast<double>(t.size());
        j << std::fixed << std::setprecision(3);
        j << "{\"track_ms\":{\"median\":" << Median(t) << ",\"p95\":" << Percentile(t, 0.95)
          << ",\"max\":" << t.back() << "}";
        j << ",\"frames_over_period\":" << s.frames_over_period;
        j << ",\"feed_rate_hz\":" << (s.feed_wall_s > 0 ? n / s.feed_wall_s : 0.0);
        j << ",\"recording_rate_hz\":" << (s.recording_span_s > 0 ? (n - 1.0) / s.recording_span_s : 0.0);
        j << ",\"feed_wall_s\":" << s.feed_wall_s << "}";
        j << std::defaultfloat << std::setprecision(9);
    }
    j << ",\"options\":{\"vocabulary\":" << JsonString(o.vocabulary) << ",\"setting\":" << JsonString(o.setting)
      << ",\"sequence\":" << JsonString(o.sequence) << ",\"output_trajectory_csv\":" << JsonString(o.out_csv)
      << ",\"load_map\":" << (o.load_map.empty() ? std::string("null") : JsonString(o.load_map))
      << ",\"save_map\":" << (o.save_map.empty() ? std::string("null") : JsonString(o.save_map))
      << ",\"mask_img\":" << (o.mask_img.empty() ? std::string("null") : JsonString(o.mask_img))
      << ",\"max_lost_frames\":" << o.max_lost_frames << ",\"aruco_dict_id\":" << o.aruco_dict_id
      << ",\"init_tag_id\":" << o.init_tag_id << ",\"init_tag_size\":" << o.init_tag_size
      << ",\"realtime\":" << JsonBool(!o.no_realtime) << "}";
    j << "}";
    std::cout << std::endl << "UMIPROC_SUMMARY " << j.str() << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    CLI::App app{"umiproc_slam: UMI ORB-SLAM3 fork on a umiproc EuRoC folder"};
    Options o;
    app.add_option("-v,--vocabulary", o.vocabulary)->required();
    app.add_option("-s,--setting", o.setting)->required();
    app.add_option("-i,--sequence", o.sequence, "EuRoC folder (times.txt, mav0/cam0/data, mav0/imu0/data.csv)")
        ->required();
    app.add_option("-o,--output_trajectory_csv", o.out_csv)->required();
    app.add_option("-l,--load_map", o.load_map, "localize against this atlas (.osa)");
    app.add_option("--save_map", o.save_map, "save the atlas here at shutdown");
    app.add_option("--mask_img", o.mask_img, "gripper mask PNG: nonzero pixels are set to 0 in every frame fed (UMI)");
    app.add_option("--max_lost_frames", o.max_lost_frames,
                   "stop after this many frames without tracking (LocalizeMonocular false; <= 0: never)");
    app.add_option("--aruco_dict_id", o.aruco_dict_id);
    app.add_option("--init_tag_id", o.init_tag_id, "table tag for the fork's tag-based initialisation (-1: off)");
    app.add_option("--init_tag_size", o.init_tag_size, "its side length in m");
    app.add_option("--save_points", o.save_points,
                   "write the saved map's points here (CSV), its keyframes and a meta JSON next to it, in the "
                   "trajectory's world");
    app.add_option("--max_frames", o.max_frames, "feed only the first N frames (< 0: all; 0: none, to export a "
                                                 "loaded map with --save_points)");
    app.add_flag("--no_realtime", o.no_realtime, "feed frames as fast as possible instead of at the recording's rate");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    Summary s;
    auto fail = [&s, &o](const std::string &msg) {
        std::cerr << msg << std::endl;
        s.exit_code = 1;
        s.error = msg;
        PrintSummary(s, o);
        return 1;
    };

    std::vector<int64_t> t_cam;
    std::vector<int64_t> t_imu;
    std::vector<cv::Point3f> acc, gyr;
    try {
        if (!LoadTimes(o.sequence + "/times.txt", t_cam)) return fail("no frame times in " + o.sequence + "/times.txt");
        if (!LoadImu(o.sequence + "/mav0/imu0/data.csv", t_imu, acc, gyr))
            return fail("no IMU samples in " + o.sequence + "/mav0/imu0/data.csv");
    } catch (const std::exception &e) {            // stoll / stod on a malformed line
        return fail("malformed input in " + o.sequence + ": " + e.what());
    }
    s.frames_total = t_cam.size();
    const std::string bad = CheckTimes(t_cam, t_imu);
    if (!bad.empty()) return fail(o.sequence + ": " + bad);
    const int64_t t0 = t_cam.front();
    s.have_t0 = true;
    s.t0_ns = t0;
    auto secs = [t0](int64_t t) { return static_cast<double>(t - t0) * 1e-9; };
    {
        std::ofstream f(o.out_csv + ".t0_ns");
        f << t0 << "\n";
        f.flush();
        if (!f) return fail("cannot write " + o.out_csv + ".t0_ns");
    }

    // first IMU sample: one before the first frame (as mono_inertial_euroc)
    size_t imu_idx = 0;
    while (imu_idx < t_imu.size() && t_imu[imu_idx] <= t_cam.front()) imu_idx++;
    if (imu_idx > 0) imu_idx--;

    cv::FileStorage fs(o.setting, cv::FileStorage::READ);
    if (!fs.isOpened()) return fail("cannot open settings " + o.setting);
    cv::Size img_size(static_cast<int>(fs["Camera.width"]), static_cast<int>(fs["Camera.height"]));
    fs.release();

    cv::Mat mask_img;
    if (!o.mask_img.empty()) {
        mask_img = cv::imread(o.mask_img, cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) return fail("cannot read mask " + o.mask_img);
        if (!SameAspect(mask_img.size(), img_size)) {
            std::ostringstream m;
            m << "mask " << mask_img.size() << " does not have the settings' aspect ratio " << img_size;
            return fail(m.str());
        }
        if (mask_img.size() != img_size) cv::resize(mask_img, mask_img, img_size, 0, 0, cv::INTER_NEAREST);
    }

    std::cout << "frames " << t_cam.size() << ", IMU samples " << t_imu.size() << ", t0_ns " << t0
              << (o.load_map.empty() ? "" : ", load_map " + o.load_map)
              << (o.save_map.empty() ? "" : ", save_map " + o.save_map)
              << (mask_img.empty() ? "" : ", mask " + o.mask_img) << std::endl;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(o.aruco_dict_id);
    ORB_SLAM3::System SLAM(o.vocabulary, o.setting, ORB_SLAM3::System::IMU_MONOCULAR, false, o.load_map, o.save_map,
                           aruco_dict, o.init_tag_id, o.init_tag_size);

    int rc = 0;
    std::vector<ORB_SLAM3::IMU::Point> imu;
    s.track_ms.reserve(t_cam.size());
    const auto feed_start = std::chrono::steady_clock::now();
    auto feed_end = feed_start;
    const size_t n_feed = o.max_frames < 0 ? t_cam.size()
                                           : std::min(t_cam.size(), static_cast<size_t>(o.max_frames));
    if (n_feed < t_cam.size()) s.stop_reason = "max_frames";
    for (size_t i = 0; i < n_feed; ++i) {
        const std::string png = o.sequence + "/mav0/cam0/data/" + std::to_string(t_cam[i]) + ".png";
        cv::Mat im = cv::imread(png, cv::IMREAD_GRAYSCALE);
        if (im.empty()) {
            std::cerr << "cannot read " << png << std::endl;
            s.error = "cannot read " + png;
            s.stop_reason = "unreadable_frame";
            rc = 1;
            break;
        }
        if (im.size() != img_size) {
            if (!SameAspect(im.size(), img_size)) {
                std::ostringstream m;
                m << png << " " << im.size() << " does not have the settings' aspect ratio " << img_size;
                std::cerr << m.str() << std::endl;
                s.error = m.str();
                s.stop_reason = "unreadable_frame";
                rc = 1;
                break;
            }
            cv::resize(im, im, img_size, 0, 0, cv::INTER_AREA);
        }
        if (!mask_img.empty()) im.setTo(cv::Scalar(0), mask_img);
        const auto t_start = std::chrono::steady_clock::now();

        imu.clear();
        if (i > 0) {
            while (imu_idx < t_imu.size() && t_imu[imu_idx] <= t_cam[i]) {
                imu.emplace_back(acc[imu_idx].x, acc[imu_idx].y, acc[imu_idx].z, gyr[imu_idx].x, gyr[imu_idx].y,
                                 gyr[imu_idx].z, secs(t_imu[imu_idx]));
                imu_idx++;
            }
        }
        auto result = SLAM.LocalizeMonocular(im, secs(t_cam[i]), imu);
        feed_end = std::chrono::steady_clock::now();
        const double track_s = std::chrono::duration<double>(feed_end - t_start).count();
        s.track_ms.push_back(track_s * 1e3);
        const size_t a = i + 1 < t_cam.size() ? i : (i > 0 ? i - 1 : i);     // frame period: to the next frame
        if (a + 1 < t_cam.size() && track_s > static_cast<double>(t_cam[a + 1] - t_cam[a]) * 1e-9)
            s.frames_over_period++;
        s.frames_fed++;
        const int state = SLAM.GetTrackingState();
        s.state_counts[state]++;
        if (state != 2) s.lost_frames++;            // Tracking::OK
        if (!result.second) {
            s.not_tracking_frames++;
            if (o.max_lost_frames > 0 && s.not_tracking_frames >= o.max_lost_frames) {
                std::cout << "Lost tracking on " << s.not_tracking_frames << " >= " << o.max_lost_frames
                          << " frames. Stopping." << std::endl;
                s.stop_reason = "max_lost_frames";
                rc = 3;
                break;
            }
        }
        if (!o.no_realtime && i + 1 < t_cam.size()) {     // as mono_inertial_euroc: wait out the frame period
            const double period = static_cast<double>(t_cam[i + 1] - t_cam[i]) * 1e-9;
            const double used =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (used < period) std::this_thread::sleep_for(std::chrono::duration<double>(period - used));
        }
    }
    s.stopped_early = s.frames_fed < t_cam.size();
    if (s.frames_fed > 0) {
        s.feed_wall_s = std::chrono::duration<double>(feed_end - feed_start).count();
        s.recording_span_s = static_cast<double>(t_cam[s.frames_fed - 1] - t_cam[0]) * 1e-9;
        std::vector<double> t(s.track_ms);
        std::sort(t.begin(), t.end());
        std::cout << std::fixed << std::setprecision(1) << "tracking " << Median(t) << " ms median, "
                  << Percentile(t, 0.95) << " ms p95, " << t.back() << " ms max; " << s.frames_over_period
                  << " frames over their period; fed at " << s.frames_fed / s.feed_wall_s << " frames/s"
                  << std::defaultfloat << std::endl;
    }
    std::cout << "lost frames " << s.lost_frames << " of " << s.frames_fed << " fed (" << t_cam.size()
              << " in the sequence)" << std::endl;

    SLAM.Shutdown();
    SLAM.SaveTrajectoryCSV(o.out_csv);
    s.have_maps = true;
    s.maps = SLAM.MapSummary();
    if (!o.save_points.empty()) {
        size_t n_points = 0, n_keyframes = 0;
        const std::string err = SaveMapPoints(SLAM, o.save_points, n_points, n_keyframes);
        if (err.empty()) {
            std::cout << "map export: " << n_points << " map points, " << n_keyframes << " keyframes -> "
                      << o.save_points << std::endl;
        } else {
            std::cerr << err << std::endl;
            if (rc == 0) {
                s.error = err;
                rc = 1;
            }
        }
    }
    if (rc == 0 && !std::ifstream(o.out_csv).good()) {
        std::cerr << "no trajectory written to " << o.out_csv << std::endl;
        s.error = "no trajectory written to " + o.out_csv;
        rc = 1;
    }
    if (rc != 0 && !o.save_map.empty() && std::remove(o.save_map.c_str()) == 0) {
        std::cout << "removed " << o.save_map << " (the run did not finish cleanly)" << std::endl;
        s.map_removed = true;
    }
    s.exit_code = rc;
    PrintSummary(s, o);
    return rc;
}
