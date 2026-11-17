// dataproc_umi_slam: run UMI's ORB-SLAM3 fork (github.com/cheng-chi/ORB_SLAM3) on a dataproc EuRoC folder.
//
// The fork's own front-end (Examples/Monocular-Inertial/gopro_slam.cc) reads a GoPro MP4 + IMU JSON and assumes a
// constant frame rate. This one reads what dataproc's orbslam3 backend already writes (write_euroc):
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
// Pacing: frames are fed at the recording's own rate by default (as upstream mono_inertial_euroc), because ORB-SLAM3
// is built for real-time input: Tracking skips keyframes while LocalMapping is busy, so feeding faster than real time
// changes what the map gets. --no_realtime feeds as fast as possible (faster, fewer keyframes, less repeatable).
//
// Input checks (exit 1 instead of a silently wrong run): camera and IMU times strictly increasing, no camera gap
// > 1 s (ORB-SLAM3 treats that as a new sequence and resets or starts a new map), the IMU spans every frame, and
// frames / mask with the settings' aspect ratio.
//
// Output: SaveTrajectoryCSV (frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w): per-frame
// T_W_cam, W = the first keyframe of the biggest map. Row k is line k of times.txt: join on frame_idx, not on the
// timestamp column (garbage before the first tracked frame, repeated on lost frames, 6 decimals). state 2 = OK
// (visually tracked), 3 = RECENTLY_LOST (is_lost=false, but an IMU-only prediction). Exit 0; 3 when
// --max_lost_frames was reached (the trajectory is still saved, unlike gopro_slam); 1 on bad input. On a non-zero
// exit the --save_map file is removed, so a failed mapping run never leaves a map behind.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI11.hpp>
#include <System.h>

namespace {

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

}  // namespace

int main(int argc, char **argv) {
    CLI::App app{"dataproc_umi_slam: UMI ORB-SLAM3 fork on a dataproc EuRoC folder"};
    std::string vocabulary, setting, sequence, out_csv, load_map, save_map, mask_img_path;
    int max_lost_frames = -1;
    int aruco_dict_id = cv::aruco::DICT_4X4_50;
    int init_tag_id = -1;            // -1: never found, so the fork's tag-based initialisation is off
    float init_tag_size = 0.16f;
    bool no_realtime = false;
    app.add_option("-v,--vocabulary", vocabulary)->required();
    app.add_option("-s,--setting", setting)->required();
    app.add_option("-i,--sequence", sequence, "EuRoC folder (times.txt, mav0/cam0/data, mav0/imu0/data.csv)")
        ->required();
    app.add_option("-o,--output_trajectory_csv", out_csv)->required();
    app.add_option("-l,--load_map", load_map, "localize against this atlas (.osa)");
    app.add_option("--save_map", save_map, "save the atlas here at shutdown");
    app.add_option("--mask_img", mask_img_path, "experiments only: nonzero pixels are set to 0 (UMI convention)");
    app.add_option("--max_lost_frames", max_lost_frames, "stop after this many lost frames (<= 0: never)");
    app.add_option("--aruco_dict_id", aruco_dict_id);
    app.add_option("--init_tag_id", init_tag_id, "table tag for the fork's tag-based initialisation (-1: off)");
    app.add_option("--init_tag_size", init_tag_size, "its side length in m");
    app.add_flag("--no_realtime", no_realtime, "feed frames as fast as possible instead of at the recording's rate");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    std::vector<int64_t> t_cam;
    std::vector<int64_t> t_imu;
    std::vector<cv::Point3f> acc, gyr;
    try {
        if (!LoadTimes(sequence + "/times.txt", t_cam)) {
            std::cerr << "no frame times in " << sequence << "/times.txt" << std::endl;
            return 1;
        }
        if (!LoadImu(sequence + "/mav0/imu0/data.csv", t_imu, acc, gyr)) {
            std::cerr << "no IMU samples in " << sequence << "/mav0/imu0/data.csv" << std::endl;
            return 1;
        }
    } catch (const std::exception &e) {            // stoll / stod on a malformed line
        std::cerr << "malformed input in " << sequence << ": " << e.what() << std::endl;
        return 1;
    }
    const std::string bad = CheckTimes(t_cam, t_imu);
    if (!bad.empty()) {
        std::cerr << sequence << ": " << bad << std::endl;
        return 1;
    }
    const int64_t t0 = t_cam.front();
    auto secs = [t0](int64_t t) { return static_cast<double>(t - t0) * 1e-9; };
    {
        std::ofstream f(out_csv + ".t0_ns");
        f << t0 << "\n";
        f.flush();
        if (!f) {
            std::cerr << "cannot write " << out_csv << ".t0_ns" << std::endl;
            return 1;
        }
    }

    // first IMU sample: one before the first frame (as mono_inertial_euroc)
    size_t imu_idx = 0;
    while (imu_idx < t_imu.size() && t_imu[imu_idx] <= t_cam.front()) imu_idx++;
    if (imu_idx > 0) imu_idx--;

    cv::FileStorage fs(setting, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "cannot open settings " << setting << std::endl;
        return 1;
    }
    cv::Size img_size(static_cast<int>(fs["Camera.width"]), static_cast<int>(fs["Camera.height"]));
    fs.release();

    cv::Mat mask_img;
    if (!mask_img_path.empty()) {
        mask_img = cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) {
            std::cerr << "cannot read mask " << mask_img_path << std::endl;
            return 1;
        }
        if (!SameAspect(mask_img.size(), img_size)) {
            std::cerr << "mask " << mask_img.size() << " does not have the settings' aspect ratio " << img_size
                      << std::endl;
            return 1;
        }
        if (mask_img.size() != img_size) cv::resize(mask_img, mask_img, img_size, 0, 0, cv::INTER_NEAREST);
    }

    std::cout << "frames " << t_cam.size() << ", IMU samples " << t_imu.size() << ", t0_ns " << t0
              << (load_map.empty() ? "" : ", load_map " + load_map)
              << (save_map.empty() ? "" : ", save_map " + save_map)
              << (mask_img.empty() ? "" : ", mask " + mask_img_path) << std::endl;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(aruco_dict_id);
    ORB_SLAM3::System SLAM(vocabulary, setting, ORB_SLAM3::System::IMU_MONOCULAR, false, load_map, save_map,
                           aruco_dict, init_tag_id, init_tag_size);

    int n_lost = 0, rc = 0;
    std::vector<ORB_SLAM3::IMU::Point> imu;
    for (size_t i = 0; i < t_cam.size(); ++i) {
        const std::string png = sequence + "/mav0/cam0/data/" + std::to_string(t_cam[i]) + ".png";
        cv::Mat im = cv::imread(png, cv::IMREAD_GRAYSCALE);
        if (im.empty()) {
            std::cerr << "cannot read " << png << std::endl;
            rc = 1;
            break;
        }
        if (im.size() != img_size) {
            if (!SameAspect(im.size(), img_size)) {
                std::cerr << png << " " << im.size() << " does not have the settings' aspect ratio " << img_size
                          << std::endl;
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
        if (!result.second) {
            n_lost++;
            if (max_lost_frames > 0 && n_lost >= max_lost_frames) {
                std::cout << "Lost tracking on " << n_lost << " >= " << max_lost_frames << " frames. Stopping."
                          << std::endl;
                rc = 3;
                break;
            }
        }
        if (!no_realtime && i + 1 < t_cam.size()) {     // as mono_inertial_euroc: wait out the frame period
            const double period = static_cast<double>(t_cam[i + 1] - t_cam[i]) * 1e-9;
            const double used =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (used < period) std::this_thread::sleep_for(std::chrono::duration<double>(period - used));
        }
    }
    std::cout << "lost frames " << n_lost << " of " << t_cam.size() << std::endl;

    SLAM.Shutdown();
    SLAM.SaveTrajectoryCSV(out_csv);
    if (rc == 0 && !std::ifstream(out_csv).good()) {
        std::cerr << "no trajectory written to " << out_csv << std::endl;
        rc = 1;
    }
    if (rc != 0 && !save_map.empty() && std::remove(save_map.c_str()) == 0)
        std::cout << "removed " << save_map << " (the run did not finish cleanly)" << std::endl;
    return rc;
}
