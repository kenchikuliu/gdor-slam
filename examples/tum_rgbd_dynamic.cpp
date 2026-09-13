/**
 * This file is part of DyGeoFusion-SLAM+
 *
 * Copyright (C) 2024 DyGeoFusion-SLAM+ Authors.
 *
 * DyGeoFusion-SLAM+ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DyGeoFusion-SLAM+ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DyGeoFusion-SLAM+. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file tum_rgbd_dynamic.cpp
 * @brief TUM RGB-D dynamic scene example for DyGeoFusion-SLAM+
 *
 * This example demonstrates DyGeoFusion-SLAM+ on TUM RGB-D dynamic sequences
 * (fr3/walking_*, fr3/sitting_*) using the DynamicMaskRefiner module.
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"
#include "include/Mask/DynamicMaskRefiner.h"
#include "include/Mask/MaskConfig.h"
#include "include/Motion3D/BackgroundSupportRouter.h"
#include "include/Motion3D/DirectRgbdPoseValidator.h"
#include "include/Motion3D/MotionMarginalizedPosePrior.h"
#include "include/Utils/ImageColor.h"
#include "include/Utils/ImageGeometry.h"
#include "viewer/imgui_viewer.h"

void LoadImages(const std::string &strAssociationFilename,
                std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD,
                std::vector<double> &vTimestamps);
void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

namespace
{

constexpr std::uint64_t kNoMapPointId =
    std::numeric_limits<std::uint64_t>::max();

struct FrontendEvidenceFrame
{
    int frame_id = -1;
    double timestamp = 0.0;
    cv::Mat bgr;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<std::uint64_t> map_point_ids;
    std::vector<unsigned char> outlier_flags;
};

struct MatchedOraclePose
{
    int frame_id = -1;
    double timestamp = 0.0;
    bool valid = false;
    Sophus::SE3f tcw;
};

std::vector<std::string> splitCsvRow(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ','))
        fields.push_back(field);
    return fields;
}

bool loadMatchedOraclePoseManifest(
    const std::filesystem::path& path,
    std::vector<MatchedOraclePose>* poses,
    std::string* error)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        if (error)
            *error = "cannot open matched-oracle pose manifest";
        return false;
    }
    std::string line;
    if (!std::getline(input, line) ||
        line != "frame,timestamp,valid,tx,ty,tz,qx,qy,qz,qw") {
        if (error)
            *error = "unexpected matched-oracle pose manifest header";
        return false;
    }
    poses->clear();
    int line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty())
            continue;
        const std::vector<std::string> fields = splitCsvRow(line);
        if (fields.size() != 10) {
            if (error)
                *error = "expected 10 fields at line " +
                    std::to_string(line_number);
            return false;
        }
        try {
            MatchedOraclePose pose;
            pose.frame_id = std::stoi(fields[0]);
            pose.timestamp = std::stod(fields[1]);
            pose.valid = std::stoi(fields[2]) != 0;
            const Eigen::Vector3f translation(
                std::stof(fields[3]), std::stof(fields[4]),
                std::stof(fields[5]));
            Eigen::Quaternionf rotation(
                std::stof(fields[9]), std::stof(fields[6]),
                std::stof(fields[7]), std::stof(fields[8]));
            if (!std::isfinite(pose.timestamp) ||
                !translation.allFinite() ||
                !rotation.coeffs().allFinite() ||
                rotation.norm() < 1e-8f) {
                throw std::runtime_error("non-finite pose");
            }
            rotation.normalize();
            pose.tcw = Sophus::SE3f(rotation, translation);
            if (pose.frame_id != static_cast<int>(poses->size()))
                throw std::runtime_error("non-contiguous frame index");
            poses->push_back(pose);
        } catch (const std::exception& exception) {
            if (error)
                *error = "invalid matched-oracle pose at line " +
                    std::to_string(line_number) + ": " + exception.what();
            return false;
        }
    }
    if (poses->empty()) {
        if (error)
            *error = "matched-oracle pose manifest is empty";
        return false;
    }
    return true;
}

std::string jsonEscape(const std::string& value)
{
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << character; break;
        }
    }
    return escaped.str();
}

bool writeKeypointCsv(
    const std::filesystem::path& path,
    const FrontendEvidenceFrame& frame)
{
    if (frame.map_point_ids.size() != frame.keypoints.size() ||
        frame.outlier_flags.size() != frame.keypoints.size()) {
        return false;
    }
    std::ofstream output(path);
    if (!output.is_open())
        return false;
    output
        << "keypoint_index,x,y,size,angle,response,octave,class_id,"
        << "map_point_id,has_map_point,is_outlier\n";
    output << std::fixed << std::setprecision(9);
    for (std::size_t index = 0; index < frame.keypoints.size(); ++index) {
        const cv::KeyPoint& keypoint = frame.keypoints[index];
        const bool has_map_point =
            frame.map_point_ids[index] != kNoMapPointId;
        output
            << index << ',' << keypoint.pt.x << ',' << keypoint.pt.y << ','
            << keypoint.size << ',' << keypoint.angle << ','
            << keypoint.response << ',' << keypoint.octave << ','
            << keypoint.class_id << ',';
        if (has_map_point)
            output << frame.map_point_ids[index];
        output << ',' << (has_map_point ? 1 : 0) << ','
               << static_cast<int>(frame.outlier_flags[index]) << '\n';
    }
    return output.good();
}

bool writeFrontendEvidence(
    const std::filesystem::path& output_dir,
    const FrontendEvidenceFrame& previous,
    const FrontendEvidenceFrame& current,
    const cv::Mat& current_depth,
    const cv::Mat& raw_semantic_mask,
    const cv::Mat& refined_static_mask,
    const std::filesystem::path& previous_rgb_source,
    const std::filesystem::path& current_rgb_source,
    const std::filesystem::path& current_depth_source,
    const std::filesystem::path& association_source,
    const std::filesystem::path& orb_config_source,
    const std::filesystem::path& mask_config_source)
{
    if (previous.frame_id + 1 != current.frame_id ||
        previous.bgr.empty() || current.bgr.empty() ||
        current_depth.empty() || raw_semantic_mask.empty() ||
        refined_static_mask.empty() ||
        raw_semantic_mask.size() != current.bgr.size() ||
        refined_static_mask.size() != current.bgr.size() ||
        previous.map_point_ids.size() != previous.keypoints.size() ||
        current.map_point_ids.size() != current.keypoints.size() ||
        previous.outlier_flags.size() != previous.keypoints.size() ||
        current.outlier_flags.size() != current.keypoints.size() ||
        previous.descriptors.rows !=
            static_cast<int>(previous.keypoints.size()) ||
        current.descriptors.rows !=
            static_cast<int>(current.keypoints.size())) {
        return false;
    }

    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error)
        return false;
    const std::string previous_stem =
        cv::format("frame_%06d", previous.frame_id);
    const std::string current_stem =
        cv::format("frame_%06d", current.frame_id);
    const std::string match_stem =
        previous_stem + "_to_" + current_stem;

    cv::Mat semantic_u8;
    raw_semantic_mask.convertTo(semantic_u8, CV_8UC1, 255.0);
    cv::Mat static_u8;
    refined_static_mask.convertTo(static_u8, CV_8UC1, 255.0);

    cv::Mat keypoint_vis;
    cv::drawKeypoints(
        current.bgr, current.keypoints, keypoint_vis,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::unordered_map<std::uint64_t, int> previous_by_map_point;
    for (std::size_t index = 0; index < previous.map_point_ids.size(); ++index) {
        const std::uint64_t map_point_id = previous.map_point_ids[index];
        if (map_point_id != kNoMapPointId &&
            previous.outlier_flags[index] == 0) {
            previous_by_map_point.emplace(
                map_point_id, static_cast<int>(index));
        }
    }

    std::vector<cv::DMatch> matches;
    std::vector<std::uint64_t> matched_map_point_ids;
    for (std::size_t index = 0; index < current.map_point_ids.size(); ++index) {
        const std::uint64_t map_point_id = current.map_point_ids[index];
        if (map_point_id == kNoMapPointId ||
            current.outlier_flags[index] != 0) {
            continue;
        }
        const auto previous_match =
            previous_by_map_point.find(map_point_id);
        if (previous_match == previous_by_map_point.end())
            continue;
        matches.emplace_back(
            previous_match->second, static_cast<int>(index), 0.0f);
        matched_map_point_ids.push_back(map_point_id);
    }

    cv::Mat matches_vis;
    cv::drawMatches(
        previous.bgr, previous.keypoints,
        current.bgr, current.keypoints,
        matches, matches_vis, cv::Scalar(0, 255, 0),
        cv::Scalar::all(-1), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const bool images_written =
        cv::imwrite((output_dir / (previous_stem + "_rgb.png")).string(),
                    previous.bgr) &&
        cv::imwrite((output_dir / (current_stem + "_rgb.png")).string(),
                    current.bgr) &&
        cv::imwrite((output_dir / (current_stem + "_depth.png")).string(),
                    current_depth) &&
        cv::imwrite(
            (output_dir / (current_stem + "_raw_semantic_mask.png")).string(),
            semantic_u8) &&
        cv::imwrite(
            (output_dir / (current_stem + "_refined_static_mask.png")).string(),
            static_u8) &&
        cv::imwrite(
            (output_dir / (current_stem + "_orb_keypoints.png")).string(),
            keypoint_vis) &&
        cv::imwrite(
            (output_dir / (match_stem + "_orb_matches.png")).string(),
            matches_vis);
    if (!images_written ||
        !writeKeypointCsv(
            output_dir / (previous_stem + "_orb_keypoints.csv"), previous) ||
        !writeKeypointCsv(
            output_dir / (current_stem + "_orb_keypoints.csv"), current)) {
        return false;
    }

    cv::FileStorage descriptors(
        (output_dir / "orb_descriptors.yml.gz").string(),
        cv::FileStorage::WRITE);
    if (!descriptors.isOpened())
        return false;
    descriptors << "previous_frame" << previous.frame_id;
    descriptors << "previous_descriptors" << previous.descriptors;
    descriptors << "current_frame" << current.frame_id;
    descriptors << "current_descriptors" << current.descriptors;
    descriptors.release();

    std::ofstream match_csv(
        output_dir / (match_stem + "_orb_matches.csv"));
    if (!match_csv.is_open())
        return false;
    match_csv
        << "match_index,map_point_id,previous_keypoint_index,"
        << "current_keypoint_index,previous_x,previous_y,current_x,current_y\n";
    match_csv << std::fixed << std::setprecision(9);
    for (std::size_t index = 0; index < matches.size(); ++index) {
        const cv::DMatch& match = matches[index];
        const cv::Point2f previous_point =
            previous.keypoints[match.queryIdx].pt;
        const cv::Point2f current_point =
            current.keypoints[match.trainIdx].pt;
        match_csv
            << index << ',' << matched_map_point_ids[index] << ','
            << match.queryIdx << ',' << match.trainIdx << ','
            << previous_point.x << ',' << previous_point.y << ','
            << current_point.x << ',' << current_point.y << '\n';
    }
    if (!match_csv.good())
        return false;

    std::ofstream metadata(output_dir / "instrumentation_metadata.json");
    if (!metadata.is_open())
        return false;
    metadata << std::fixed << std::setprecision(9)
        << "{\n"
        << "  \"schema\": \"dynags-frontend-evidence-v1\",\n"
        << "  \"frame_index_basis\": \"zero-based association row\",\n"
        << "  \"previous_frame\": " << previous.frame_id << ",\n"
        << "  \"current_frame\": " << current.frame_id << ",\n"
        << "  \"previous_timestamp\": " << previous.timestamp << ",\n"
        << "  \"current_timestamp\": " << current.timestamp << ",\n"
        << "  \"previous_keypoint_count\": "
        << previous.keypoints.size() << ",\n"
        << "  \"current_keypoint_count\": "
        << current.keypoints.size() << ",\n"
        << "  \"confirmed_map_point_match_count\": "
        << matches.size() << ",\n"
        << "  \"sources\": {\n"
        << "    \"previous_rgb\": \""
        << jsonEscape(previous_rgb_source.string()) << "\",\n"
        << "    \"current_rgb\": \""
        << jsonEscape(current_rgb_source.string()) << "\",\n"
        << "    \"current_depth\": \""
        << jsonEscape(current_depth_source.string()) << "\",\n"
        << "    \"association\": \""
        << jsonEscape(association_source.string()) << "\",\n"
        << "    \"orb_config\": \""
        << jsonEscape(orb_config_source.string()) << "\",\n"
        << "    \"mask_config\": \""
        << jsonEscape(mask_config_source.string()) << "\"\n"
        << "  },\n"
        << "  \"evidence_contract\": {\n"
        << "    \"raw_semantic_mask\": "
        << "\"DynamicMaskRefiner::getSemanticMask() immediately after "
        << "the target-frame compute() call\",\n"
        << "    \"refined_static_mask\": "
        << "\"exact compute() return matrix passed to System::TrackRGBD, "
        << "Tracking::GrabImageRGBD, Frame::Frame, and "
        << "Frame::ExtractORB\",\n"
        << "    \"orb_keypoints\": "
        << "\"Tracking::mCurrentFrame.mvKeysUn copied by System after "
        << "the target-frame TrackRGBD call\",\n"
        << "    \"orb_descriptors\": "
        << "\"Tracking::mCurrentFrame.mDescriptors from the same frame "
        << "and extractor invocation\",\n"
        << "    \"orb_matches\": "
        << "\"one-to-one intersection of non-outlier post-tracking "
        << "MapPoint IDs in the previous and current frames; these are persistent "
        << "ORB-SLAM3 track correspondences, not independently "
        << "recomputed OpenCV matches\",\n"
        << "    \"mask_serialization\": "
        << "\"binary 0/1 matrices multiplied by 255 only for lossless "
        << "PNG storage\",\n"
        << "    \"post_processing\": \"none\"\n"
        << "  },\n"
        << "  \"hash_manifest\": "
        << "\"provenance_manifest.json is generated by the replay "
        << "finalizer after a successful run\"\n"
        << "}\n";
    return metadata.good();
}

} // namespace

struct HeldOutFrame
{
    int frame_id;
    double timestamp;
    Sophus::SE3f Tcw;
    cv::Mat bgr;
    cv::Mat static_mask;
};

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_sequence"                     /*4*/
                  << " path_to_association"                  /*5*/
                  << " path_to_trajectory_output_directory/" /*6*/
                  << " (optional)path_to_mask_config.yaml"   /*7*/
                  << " (optional)no_viewer"                  /*8*/
                  << " (optional)path_to_dinov2_maps"        /*9*/
                  << " [--seed N] [--no-realtime] [--sync-local-mapping]"
                  << " [--sync-loop-closing]"
                  << " [--shadow-translation-horizon N]"
                  << " [--export-replay-packets]"
                  << " [--capture-replay-state]"
                  << " [--export-replay-state]"
                  << " [--export-static-masks]"
                  << " [--disable-gaussian-mapper]"
                  << " [--tracking-only]"
                  << " [--freeze-map-after-frame N]"
                  << " [--frame-period-ms N]"
                  << " [--instrument-frame N]"
                  << " [--instrument-output DIR]"
                  << " [--matched-oracle-mode apply|sham]"
                  << " [--matched-oracle-poses FILE]"
                  << " [--background-support-shadow]"
                  << " [--gaussian-background-shadow]"
                  << " [--heldout-stride N] [--max-frames N]"
                  << std::endl;
        return 1;
    }

    bool use_viewer = true;
    bool realtime_playback = true;
    bool synchronize_local_mapping = false;
    bool synchronize_loop_closing = false;
    bool export_replay_packets = false;
    bool export_replay_state = false;
    bool export_static_masks = false;
    bool disable_gaussian_mapper = false;
    bool tracking_only = false;
    bool background_support_shadow = false;
    bool gaussian_background_shadow = false;
    int freeze_map_after_frame = -1;
    double frame_period_ms = 0.0;
    int seed = 0;
    int heldout_stride = 0;
    int max_frames = 0;
    int instrument_frame = -1;
    int shadow_translation_horizon_override = -1;
    std::string mask_config_path = "";
    std::string dinov2_maps_dir = "";  // DINOv2 pre-computed dynamic probability maps
    std::string matched_oracle_mode = "disabled";
    std::filesystem::path matched_oracle_pose_manifest;
    std::filesystem::path instrument_output_dir;

    // Parse optional arguments
    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "no_viewer") {
            use_viewer = false;
        } else if (arg == "--no-realtime") {
            realtime_playback = false;
        } else if (arg == "--sync-local-mapping") {
            synchronize_local_mapping = true;
        } else if (arg == "--sync-loop-closing") {
            synchronize_loop_closing = true;
        } else if (arg == "--shadow-translation-horizon") {
            if (++i >= argc) {
                std::cerr
                    << "--shadow-translation-horizon requires an integer value"
                    << std::endl;
                return 1;
            }
            try {
                shadow_translation_horizon_override = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid shadow-translation horizon: "
                          << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--export-replay-packets") {
            export_replay_packets = true;
        } else if (arg == "--capture-replay-state") {
            export_replay_state = true;
        } else if (arg == "--export-replay-state") {
            export_replay_state = true;
        } else if (arg == "--export-static-masks") {
            export_static_masks = true;
        } else if (arg == "--disable-gaussian-mapper") {
            disable_gaussian_mapper = true;
            use_viewer = false;
        } else if (arg == "--tracking-only") {
            tracking_only = true;
            use_viewer = false;
        } else if (arg == "--freeze-map-after-frame") {
            if (++i >= argc) {
                std::cerr
                    << "--freeze-map-after-frame requires an integer value"
                    << std::endl;
                return 1;
            }
            try {
                freeze_map_after_frame = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid freeze frame: " << argv[i]
                          << std::endl;
                return 1;
            }
        } else if (arg == "--frame-period-ms") {
            if (++i >= argc) {
                std::cerr << "--frame-period-ms requires a numeric value"
                          << std::endl;
                return 1;
            }
            try {
                frame_period_ms = std::stod(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid frame period: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--instrument-frame") {
            if (++i >= argc) {
                std::cerr << "--instrument-frame requires an integer value"
                          << std::endl;
                return 1;
            }
            try {
                instrument_frame = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid instrumentation frame: "
                          << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--instrument-output") {
            if (++i >= argc) {
                std::cerr << "--instrument-output requires a directory"
                          << std::endl;
                return 1;
            }
            instrument_output_dir = argv[i];
        } else if (arg == "--matched-oracle-mode") {
            if (++i >= argc) {
                std::cerr
                    << "--matched-oracle-mode requires apply or sham"
                    << std::endl;
                return 1;
            }
            matched_oracle_mode = argv[i];
            if (matched_oracle_mode != "apply" &&
                matched_oracle_mode != "sham") {
                std::cerr
                    << "--matched-oracle-mode requires apply or sham"
                    << std::endl;
                return 1;
            }
        } else if (arg == "--matched-oracle-poses") {
            if (++i >= argc) {
                std::cerr
                    << "--matched-oracle-poses requires a CSV file"
                    << std::endl;
                return 1;
            }
            matched_oracle_pose_manifest = argv[i];
        } else if (arg == "--background-support-shadow") {
            background_support_shadow = true;
        } else if (arg == "--gaussian-background-shadow") {
            gaussian_background_shadow = true;
        } else if (arg == "--seed") {
            if (++i >= argc) {
                std::cerr << "--seed requires an integer value" << std::endl;
                return 1;
            }
            try {
                seed = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid seed: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg.rfind("--seed=", 0) == 0) {
            try {
                seed = std::stoi(arg.substr(7));
            } catch (const std::exception&) {
                std::cerr << "Invalid seed: " << arg.substr(7) << std::endl;
                return 1;
            }
        } else if (arg == "--heldout-stride") {
            if (++i >= argc) {
                std::cerr << "--heldout-stride requires an integer value" << std::endl;
                return 1;
            }
            try {
                heldout_stride = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid held-out stride: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-frames") {
            if (++i >= argc) {
                std::cerr << "--max-frames requires an integer value" << std::endl;
                return 1;
            }
            try {
                max_frames = std::stoi(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid frame limit: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg.find(".yaml") != std::string::npos) {
            mask_config_path = arg;
        } else if (std::filesystem::is_directory(arg)) {
            dinov2_maps_dir = arg;  // directory of .bin dynamic maps
            std::cout << "[DyGeoFusion+DINOv2] Using DINOv2 dynamic maps from: " << arg << std::endl;
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    if (seed < 0) {
        std::cerr << "Seed must be non-negative" << std::endl;
        return 1;
    }
    if (heldout_stride < 0) {
        std::cerr << "Held-out stride must be non-negative" << std::endl;
        return 1;
    }
    if (max_frames < 0) {
        std::cerr << "Frame limit must be non-negative" << std::endl;
        return 1;
    }
    if (instrument_frame < -1) {
        std::cerr << "Instrumentation frame must be non-negative"
                  << std::endl;
        return 1;
    }
    if (instrument_frame < 0 && !instrument_output_dir.empty()) {
        std::cerr << "--instrument-output requires --instrument-frame"
                  << std::endl;
        return 1;
    }
    if ((matched_oracle_mode == "disabled") !=
        matched_oracle_pose_manifest.empty()) {
        std::cerr
            << "--matched-oracle-mode and --matched-oracle-poses "
            << "must be provided together" << std::endl;
        return 1;
    }
    if (freeze_map_after_frame < -1) {
        std::cerr << "Freeze frame must be non-negative" << std::endl;
        return 1;
    }
    if (shadow_translation_horizon_override < -1) {
        std::cerr << "Shadow-translation horizon override must be "
                  << "non-negative" << std::endl;
        return 1;
    }
    if (synchronize_loop_closing && !synchronize_local_mapping) {
        std::cerr << "--sync-loop-closing requires --sync-local-mapping"
                  << std::endl;
        return 1;
    }
    if (!std::isfinite(frame_period_ms) || frame_period_ms < 0.0) {
        std::cerr << "Frame period must be finite and non-negative"
                  << std::endl;
        return 1;
    }
    if (tracking_only && heldout_stride > 0) {
        std::cerr << "--tracking-only cannot be combined with "
                  << "--heldout-stride" << std::endl;
        return 1;
    }
    if (gaussian_background_shadow &&
        (tracking_only || disable_gaussian_mapper)) {
        std::cerr
            << "--gaussian-background-shadow requires the Gaussian mapper"
            << std::endl;
        return 1;
    }
    if (freeze_map_after_frame >= 0 &&
        (!tracking_only || !export_replay_packets)) {
        std::cerr
            << "--freeze-map-after-frame requires --tracking-only and "
            << "--export-replay-packets" << std::endl;
        return 1;
    }
    if (export_replay_packets && freeze_map_after_frame < 0) {
        std::cerr
            << "--export-replay-packets requires "
            << "--freeze-map-after-frame" << std::endl;
        return 1;
    }
    std::srand(seed);
    cv::setRNGSeed(seed);
    torch::manual_seed(seed);
    const auto run_start = std::chrono::steady_clock::now();

    std::filesystem::path output_dir(argv[6]);
    std::error_code output_error;
    std::filesystem::create_directories(output_dir, output_error);
    if (output_error) {
        std::cerr << "Failed to create output directory " << output_dir
                  << ": " << output_error.message() << std::endl;
        return 1;
    }
    if (instrument_frame >= 0 && instrument_output_dir.empty())
        instrument_output_dir = output_dir / "instrumentation";
    const std::filesystem::path exported_static_mask_dir =
        output_dir / "static_masks";
    if (export_static_masks) {
        output_error.clear();
        std::filesystem::create_directories(
            exported_static_mask_dir, output_error);
        if (output_error) {
            std::cerr << "Failed to create static-mask export directory "
                      << exported_static_mask_dir << ": "
                      << output_error.message() << std::endl;
            return 1;
        }
    }

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    std::string strAssociationFilename = std::string(argv[5]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (max_frames > 0 && nImages > max_frames) {
        nImages = max_frames;
        vstrImageFilenamesRGB.resize(nImages);
        vstrImageFilenamesD.resize(nImages);
        vTimestamps.resize(nImages);
    }
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }
    if (instrument_frame == 0) {
        std::cerr << "Instrumentation requires a previous frame for "
                  << "true temporal correspondences" << std::endl;
        return 1;
    }
    if (instrument_frame >= nImages) {
        std::cerr << "Instrumentation frame " << instrument_frame
                  << " is outside the processed frame prefix [0, "
                  << (nImages - 1) << "]" << std::endl;
        return 1;
    }
    std::vector<MatchedOraclePose> matched_oracle_poses;
    if (matched_oracle_mode != "disabled") {
        std::string oracle_error;
        if (!loadMatchedOraclePoseManifest(
                matched_oracle_pose_manifest,
                &matched_oracle_poses, &oracle_error)) {
            std::cerr << "Invalid matched-oracle pose manifest: "
                      << oracle_error << std::endl;
            return 1;
        }
        if (matched_oracle_poses.size() !=
            static_cast<std::size_t>(nImages)) {
            std::cerr
                << "Matched-oracle pose manifest has "
                << matched_oracle_poses.size() << " rows for "
                << nImages << " input frames" << std::endl;
            return 1;
        }
        for (int frame = 0; frame < nImages; ++frame) {
            if (std::abs(
                    matched_oracle_poses[frame].timestamp -
                    vTimestamps[frame]) > 1e-6) {
                std::cerr
                    << "Matched-oracle timestamp mismatch at frame "
                    << frame << std::endl;
                return 1;
            }
        }
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Initialize DynamicMaskRefiner configuration
    DyGeoFusion::MaskConfig mask_cfg;
    if (!mask_config_path.empty()) {
        if (mask_cfg.loadFromYAML(mask_config_path)) {
            std::cout << "[DyGeoFusion-SLAM+] Loaded mask config from: " << mask_config_path << std::endl;
        } else {
            std::cerr << "[DyGeoFusion-SLAM+] Failed to load mask config, using defaults" << std::endl;
        }
    } else {
        std::cout << "[DyGeoFusion-SLAM+] Using default mask configuration" << std::endl;
    }
    if (shadow_translation_horizon_override >= 0) {
        mask_cfg.motion_prior_shadow_translation_horizon =
            shadow_translation_horizon_override;
    }

    // Set debug output directory
    mask_cfg.debug_output_dir = (output_dir / "mask_debug").string();
    mask_cfg.print();
    if (export_replay_packets &&
        !mask_cfg.motion_pose_prior_shadow_only) {
        std::cerr << "--export-replay-packets requires a shadow-only "
                  << "motion-prior config" << std::endl;
        return 1;
    }
    if (mask_cfg.motion_prior_shuffle_lag_frames > 0 &&
        !mask_cfg.use_motion_pose_prior) {
        std::cerr << "motion_prior_shuffle_lag_frames requires "
                  << "mask.use_motion_pose_prior" << std::endl;
        return 1;
    }
    if (mask_cfg.motion_prior_shadow_translation_horizon < 0) {
        std::cerr << "motion_prior_shadow_translation_horizon must be "
                  << "non-negative" << std::endl;
        return 1;
    }
    if (mask_cfg.motion_prior_shadow_translation_horizon > 0 &&
        (!mask_cfg.use_motion_pose_prior ||
         mask_cfg.motion_prior_initialization_only)) {
        std::cerr
            << "motion_prior_shadow_translation_horizon requires the "
            << "full Schur optimization path"
            << std::endl;
        return 1;
    }
    if (mask_cfg.motion_prior_max_static_information_leverage > 0.0f &&
        mask_cfg.motion_prior_shadow_translation_horizon == 0) {
        std::cerr
            << "motion_prior_max_static_information_leverage requires "
            << "the shadow-translation posterior path" << std::endl;
        return 1;
    }
    if (mask_cfg.motion_prior_static_information_leverage_mode != 0 &&
        mask_cfg.motion_prior_max_static_information_leverage <= 0.0f) {
        std::cerr
            << "normalize-to-target requires a positive "
            << "motion_prior_max_static_information_leverage"
            << std::endl;
        return 1;
    }
    if (disable_gaussian_mapper &&
        (mask_cfg.use_depth_consistency ||
         mask_cfg.use_temporal_background_refinement)) {
        std::cerr
            << "--disable-gaussian-mapper cannot be combined with "
            << "a rendered-depth mask module"
            << std::endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD);
    pSLAM->ConfigureExternalPosePriorGate(
        mask_cfg.motion_gate_min_inlier_gain,
        mask_cfg.motion_gate_min_inlier_gain_ratio,
        mask_cfg.motion_gate_max_static_inliers,
        mask_cfg.motion_gate_min_translation_innovation,
        mask_cfg.motion_gate_max_translation_innovation,
        mask_cfg.motion_gate_max_rotation_innovation,
        mask_cfg.motion_prior_information_scale,
        mask_cfg.motion_prior_use_direct_validation,
        mask_cfg.motion_prior_direct_score_mode,
        mask_cfg.motion_prior_inject_local_map,
        mask_cfg.motion_prior_initialization_only,
        mask_cfg.motion_prior_require_common_support_improvement,
        mask_cfg.motion_prior_bypass_reliability_gate,
        mask_cfg.motion_prior_gate_policy);
    pSLAM->ConfigureExternalPosePriorShadowTranslation(
        mask_cfg.motion_prior_shadow_translation_horizon,
        mask_cfg.motion_prior_posterior_min_score_improvement,
        mask_cfg.motion_prior_shadow_translation_blend,
        mask_cfg.motion_prior_max_static_information_leverage,
        mask_cfg.motion_prior_static_information_leverage_mode);
    const bool adaptive_feature_risk_gate =
        mask_cfg.use_adaptive_feature_extraction &&
        mask_cfg.adaptive_feature_min_previous_inliers > 0;
    const bool temporal_recovery_risk_gate =
        mask_cfg.temporal_recovery_only &&
        mask_cfg.temporal_recovery_require_tracking_risk &&
        mask_cfg.temporal_recovery_min_previous_inliers > 0;
    pSLAM->ConfigureAdaptiveMaskFeatures(
        mask_cfg.use_adaptive_feature_extraction &&
            !adaptive_feature_risk_gate,
        mask_cfg.adaptive_feature_relaxation);
    const bool capture_replay_state =
        export_replay_packets || export_replay_state;
    pSLAM->EnableExternalPosePriorReplayCapture(
        capture_replay_state);

    // Initialize inlier ratio logging for paper Figure 4
    pSLAM->getTracker()->SetInlierRatioLogPath((output_dir / "InlierRatio.txt").string());

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper;
    std::thread training_thd;
    if (!tracking_only && !disable_gaussian_mapper) {
        pGausMapper = std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, seed, device_type);
        training_thd = std::thread(
            &GaussianMapper::run, pGausMapper.get());
    }

    // Create DynamicMaskRefiner
    std::shared_ptr<DyGeoFusion::DynamicMaskRefiner> pMaskRefiner =
        std::make_shared<DyGeoFusion::DynamicMaskRefiner>(mask_cfg, device_type);

    std::unique_ptr<Motion3D::BackgroundSupportRouter>
        background_support_router;
    std::unique_ptr<Motion3D::BackgroundSupportPoseRefiner>
        background_support_pose_refiner;
    std::unique_ptr<Motion3D::DirectRgbdPoseValidator>
        background_support_pose_validator;
    std::unique_ptr<Motion3D::GaussianBackgroundPoseRefiner>
        gaussian_background_pose_refiner;
    if (background_support_shadow) {
        background_support_router =
            std::make_unique<Motion3D::BackgroundSupportRouter>();
        std::cout
            << "[BackgroundSupportRouter] shadow-only diagnostics enabled; "
               "no pose or mapping mutation will be performed"
            << std::endl;
    }
    if (gaussian_background_shadow) {
        std::cout
            << "[GaussianBackgroundPoseRefiner] shadow-only diagnostics "
               "enabled; the Parent tracker and Gaussian mapper remain "
               "unmodified"
            << std::endl;
    }

    // Initialize coverage logging for paper Figure 5
    pMaskRefiner->SetCoverageLogPath((output_dir / "DynamicCoverage.txt").string());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // Statistics for mask computation
    std::vector<float> vTimesMask;
    vTimesMask.resize(nImages);

    // Get camera intrinsics for depth rendering
    auto pSettings = pSLAM->getSettings();
    auto pCamera = pSettings->camera1();
    float cam_fx = pCamera->getParameter(0);
    float cam_fy = pCamera->getParameter(1);
    float cam_cx = pCamera->getParameter(2);
    float cam_cy = pCamera->getParameter(3);
    cv::Size imSize = pSettings->newImSize();
    int cam_width = imSize.width;
    int cam_height = imSize.height;
    float depth_map_factor = pSettings->depthMapFactor();
    float depth_to_meters = 1.0f;
    if (depth_map_factor < 1e-5f && depth_map_factor > -1e-5f) {
        std::cerr << "[DyGeoFusion-SLAM+] Invalid RGBD.DepthMapFactor; using raw depth scale." << std::endl;
    } else {
        depth_to_meters = 1.0f / depth_map_factor;
    }
    std::cout << "[DyGeoFusion-SLAM+] Dynamic mask depth scale: raw / "
              << depth_map_factor << " meters" << std::endl;

    if (background_support_shadow) {
        Motion3D::BackgroundSupportPoseRefiner::Config pose_config;
        pose_config.router.min_support_points = 40;
        pose_config.router.max_support_points = 256;
        pose_config.router.ring_radius = 3;
        pose_config.router.sample_stride = 4;
        pose_config.router.min_gradient = 0.02f;
        pose_config.router.max_depth_residual = 0.05f;
        pose_config.router.depth_residual_scale = 0.02f;
        pose_config.router.flow_guard_radius = 1;
        pose_config.max_iterations = 4;
        pose_config.huber_delta = 0.03f;
        pose_config.min_inlier_ratio = 0.5f;
        pose_config.min_rmse_improvement = 0.01f;
        pose_config.max_translation_correction = 0.02f;
        pose_config.max_rotation_correction = 0.0f;
        pose_config.translation_only = true;
        background_support_pose_refiner =
            std::make_unique<Motion3D::BackgroundSupportPoseRefiner>(
                pose_config,
                Motion3D::BackgroundSupportPoseRefiner::Intrinsics{
                    cam_fx, cam_fy, cam_cx, cam_cy});
        Motion3D::DirectRgbdPoseValidator::Config validator_config;
        validator_config.grid_step = 8;
        validator_config.min_common_support = 50;
        validator_config.depth_weight = 1.0f;
        validator_config.photometric_weight = 1.0f;
        background_support_pose_validator =
            std::make_unique<Motion3D::DirectRgbdPoseValidator>(
                validator_config);
    }
    if (gaussian_background_shadow) {
        Motion3D::GaussianBackgroundPoseRefiner::Config gaussian_config;
        gaussian_config.sample_stride = 4;
        gaussian_config.split_tile_size = 16;
        gaussian_config.dynamic_dilation_radius = 8;
        gaussian_config.max_iterations = 4;
        gaussian_config.min_proposal_support = 80;
        gaussian_config.min_validation_support = 80;
        gaussian_config.minimum_observable_rank = 2;
        gaussian_config.min_depth = 0.1f;
        gaussian_config.max_depth = 8.0f;
        gaussian_config.max_correspondence_distance = 0.10f;
        gaussian_config.huber_delta = 0.03f;
        gaussian_config.min_proposal_relative_improvement = 0.01f;
        gaussian_config.min_validation_relative_improvement = 0.01f;
        gaussian_config.eigenvalue_relative_threshold = 1e-3f;
        gaussian_config.max_translation_correction = 0.02f;
        gaussian_background_pose_refiner =
            std::make_unique<Motion3D::GaussianBackgroundPoseRefiner>(
                gaussian_config,
                Motion3D::GaussianBackgroundPoseRefiner::Intrinsics{
                    cam_fx, cam_fy, cam_cx, cam_cy});
    }

    std::unique_ptr<Motion3D::MotionMarginalizedPosePrior> motion_pose_prior;
    if (mask_cfg.use_motion_pose_prior) {
        Motion3D::MotionMarginalizedPosePrior::Config motion_config;
        motion_config.max_features = mask_cfg.motion_max_features;
        motion_config.min_features_per_object = mask_cfg.motion_min_features_per_object;
        motion_config.min_component_area = mask_cfg.motion_min_component_area;
        motion_config.min_track_age = mask_cfg.motion_min_track_age;
        motion_config.ransac_threshold = mask_cfg.motion_ransac_threshold;
        motion_config.min_inlier_ratio = mask_cfg.motion_min_inlier_ratio;
        motion_config.min_information = mask_cfg.motion_min_information;
        motion_config.max_translation = mask_cfg.motion_max_translation;
        motion_config.max_rotation = mask_cfg.motion_max_rotation;
        motion_config.use_depth_foreground_filter =
            mask_cfg.motion_use_depth_foreground_filter;
        motion_config.foreground_depth_separation =
            mask_cfg.motion_foreground_depth_separation;
        motion_config.foreground_min_fraction =
            mask_cfg.motion_foreground_min_fraction;
        motion_config.min_object_translation_speed =
            mask_cfg.motion_min_object_translation_speed;
        motion_config.min_object_rotation_speed =
            mask_cfg.motion_min_object_rotation_speed;
        motion_config.measurement_translation_sigma =
            mask_cfg.motion_measurement_translation_sigma;
        motion_config.measurement_rotation_sigma =
            mask_cfg.motion_measurement_rotation_sigma;
        motion_config.camera_translation_sigma =
            mask_cfg.motion_camera_translation_sigma;
        motion_config.camera_rotation_sigma =
            mask_cfg.motion_camera_rotation_sigma;
        motion_config.velocity_process_translation_sigma =
            mask_cfg.motion_velocity_process_translation_sigma;
        motion_config.velocity_process_rotation_sigma =
            mask_cfg.motion_velocity_process_rotation_sigma;
        motion_config.max_information_eigenvalue =
            mask_cfg.motion_max_information_eigenvalue;
        motion_config.candidate_mahalanobis_threshold =
            mask_cfg.motion_candidate_mahalanobis_threshold;
        motion_config.seed = static_cast<unsigned int>(seed);
        Motion3D::MotionMarginalizedPosePrior::CameraIntrinsics intrinsics;
        intrinsics.fx = cam_fx;
        intrinsics.fy = cam_fy;
        intrinsics.cx = cam_cx;
        intrinsics.cy = cam_cy;
        motion_pose_prior = std::make_unique<Motion3D::MotionMarginalizedPosePrior>(
            motion_config, intrinsics);
    }

    // Track camera pose for depth rendering
    Sophus::SE3f lastValidTcw;
    Sophus::SE3f previousValidTcw;
    bool hasValidPose = false;
    bool hasPreviousValidPose = false;
    bool lastFramePoseWasValid = false;

    std::ofstream frame_metrics((output_dir / "frame_metrics.csv").string());
    if (!frame_metrics.is_open()) {
        std::cerr << "Cannot open per-frame metrics log" << std::endl;
        return 1;
    }
    frame_metrics << "frame,timestamp,tracking_state,pose_valid,lost,mask_seconds,motion_prior_seconds,tracking_seconds,compute_seconds,dynamic_ratio,motion_foreground_fraction,prior_valid,prior_raw_valid,prior_source_frame,prior_shuffle_lag_frames,prior_gate_bypassed,prior_candidate,prior_consensus_pass,direct_validation_valid,direct_validation_pass,prior_common_support_pass,prior_would_use,prior_used,oracle_valid,oracle_prefers_dynamic,oracle_applied,oracle_static_translation_error_m,oracle_dynamic_translation_error_m,oracle_static_rotation_error_rad,oracle_dynamic_rotation_error_rad,shadow_translation_active,shadow_translation_applied,shadow_translation_remaining,shadow_translation_innovation_m,shadow_translation_stop_reason,shadow_translation_factor_injected,posterior_valid,posterior_pass,posterior_static_inliers,posterior_dynamic_inliers,posterior_common_support,posterior_static_score,posterior_dynamic_score,velocity_neutralized,prior_information,prior_objects_observed,prior_objects_used,prior_feature_tracks,prior_inlier_tracks,prior_static_matches,prior_dynamic_matches,prior_static_inliers,prior_dynamic_inliers,prior_common_support,prior_static_common_score,prior_dynamic_common_score,direct_common_support,direct_static_depth_score,direct_dynamic_depth_score,direct_static_photometric_score,direct_dynamic_photometric_score,direct_static_combined_score,direct_dynamic_combined_score,trajectory_carried_forward,mask_pose_predicted,temporal_refinement_applied,temporal_recovered_static_pixels,temporal_added_dynamic_pixels,temporal_flow_guard_valid,temporal_flow_guard_rejected_pixels,temporal_recovery_risk_active,temporal_recovery_risk_previous_inliers,temporal_recovery_risk_hold_remaining,tracking_recovery_candidate_pixels,tracking_recovery_blocked_pixels,tracking_recovery_audit_pixels,tracking_recovery_mapping_leak_pixels,static_mask_ratio,mapping_weight_present,mapping_static_ratio,adaptive_feature_active,adaptive_feature_previous_inliers,adaptive_fast_threshold,extracted_features\n";
    frame_metrics << std::fixed << std::setprecision(9);
    std::ofstream background_support_metrics;
    std::ofstream background_support_pose_metrics;
    std::ofstream gaussian_background_pose_metrics;
    if (background_support_router) {
        background_support_metrics.open(
            (output_dir / "background_support_shadow.csv").string());
        if (!background_support_metrics.is_open()) {
            std::cerr
                << "Cannot open background-support shadow metrics log"
                << std::endl;
            return 1;
        }
        background_support_metrics
            << "frame,timestamp,dynamic_pixels,ring_pixels,"
               "candidate_points,selected_points,support_ratio,"
               "median_depth_residual,valid\n";
        background_support_metrics << std::fixed << std::setprecision(9);
        background_support_pose_metrics.open(
            (output_dir / "background_support_pose_shadow.csv").string());
        if (!background_support_pose_metrics.is_open()) {
            std::cerr
                << "Cannot open background-support pose shadow metrics log"
                << std::endl;
            return 1;
        }
        background_support_pose_metrics
            << "frame,timestamp,candidate_pose_available,gate_pass,"
               "proposal_valid,proposal_improved,"
               "routed_support,geometric_support,inliers,"
               "initial_rmse,final_rmse,translation_correction,"
               "rotation_correction,direct_valid,direct_prefers_candidate,"
               "direct_common_support,direct_parent_score,"
               "direct_candidate_score,"
               "parent_twc_tx,parent_twc_ty,parent_twc_tz,"
               "parent_twc_qx,parent_twc_qy,parent_twc_qz,parent_twc_qw,"
               "candidate_twc_tx,candidate_twc_ty,candidate_twc_tz,"
               "candidate_twc_qx,candidate_twc_qy,candidate_twc_qz,"
               "candidate_twc_qw\n";
        background_support_pose_metrics
            << std::fixed << std::setprecision(9);
    }
    if (gaussian_background_pose_refiner) {
        gaussian_background_pose_metrics.open(
            (output_dir / "gaussian_background_pose_shadow.csv").string());
        if (!gaussian_background_pose_metrics.is_open()) {
            std::cerr
                << "Cannot open Gaussian-background pose shadow metrics log"
                << std::endl;
            return 1;
        }
        gaussian_background_pose_metrics
            << "frame,timestamp,map_iteration,render_available,"
               "dynamic_pixels,safe_static_pixels,proposal_points,"
               "validation_points,proposal_validation_overlap,"
               "proposal_support,validation_common_support,"
               "observable_rank,candidate_pose_available,"
               "proposal_improved,validation_valid,"
               "validation_prefers_candidate,gate_pass,"
               "proposal_parent_loss,proposal_candidate_loss,"
               "validation_parent_loss,validation_candidate_loss,"
               "translation_correction,rotation_correction,"
               "parent_twc_tx,parent_twc_ty,parent_twc_tz,"
               "parent_twc_qx,parent_twc_qy,parent_twc_qz,parent_twc_qw,"
               "candidate_twc_tx,candidate_twc_ty,candidate_twc_tz,"
               "candidate_twc_qx,candidate_twc_qy,candidate_twc_qz,"
               "candidate_twc_qw\n";
        gaussian_background_pose_metrics
            << std::fixed << std::setprecision(9);
    }
    std::ofstream counterfactual_poses(
        (output_dir / "motion_prior_counterfactual.csv").string());
    if (!counterfactual_poses.is_open()) {
        std::cerr << "Cannot open motion-prior counterfactual log" << std::endl;
        return 1;
    }
    counterfactual_poses
        << "frame,timestamp,previous_timestamp,prior_consensus_pass,"
        << "direct_validation_valid,direct_validation_pass,"
        << "prior_common_support_pass,prior_would_use,prior_used,"
        << "oracle_valid,oracle_prefers_dynamic,oracle_applied,"
        << "oracle_static_translation_error_m,"
        << "oracle_dynamic_translation_error_m,"
        << "oracle_static_rotation_error_rad,"
        << "oracle_dynamic_rotation_error_rad,"
        << "static_inliers,dynamic_inliers,common_support,"
        << "static_common_score,dynamic_common_score,"
        << "direct_common_support,direct_static_depth_score,"
        << "direct_dynamic_depth_score,direct_static_photometric_score,"
        << "direct_dynamic_photometric_score,direct_static_combined_score,"
        << "direct_dynamic_combined_score,"
        << "contract_version,hash_algorithm,previous_frame_id,map_id,"
        << "map_generation,reference_kf_id,reference_kf_pose_hash,"
        << "map_point_state_hash,static_support_hash,dynamic_support_hash,"
        << "common_support_hash,"
        << "previous_tx,previous_ty,previous_tz,previous_qx,previous_qy,"
        << "previous_qz,previous_qw,"
        << "static_tx,static_ty,static_tz,static_qx,static_qy,static_qz,static_qw,"
        << "dynamic_tx,dynamic_ty,dynamic_tz,dynamic_qx,dynamic_qy,dynamic_qz,"
        << "dynamic_qw\n";
    counterfactual_poses << std::fixed << std::setprecision(9);
    std::ofstream replay_packet_index;
    std::ofstream replay_execution_state;
    std::filesystem::path replay_packet_dir;
    if (export_replay_packets) {
        replay_packet_dir = output_dir / "replay_packets";
        std::error_code replay_error;
        std::filesystem::create_directories(
            replay_packet_dir, replay_error);
        if (replay_error) {
            std::cerr << "Cannot create replay packet directory: "
                      << replay_error.message() << std::endl;
            return 1;
        }
        replay_packet_index.open(
            (replay_packet_dir / "index.csv").string());
        if (!replay_packet_index.is_open()) {
            std::cerr << "Cannot create replay packet index" << std::endl;
            return 1;
        }
        replay_packet_index
            << "frame,timestamp,packet,previous_frame_id,map_id,"
            << "map_generation,reference_kf_id,map_point_state_hash,"
            << "static_support_hash,dynamic_support_hash,"
            << "common_support_hash,freeze_protocol,freeze_requested,"
            << "local_mapping_idle_ack,local_mapping_stopped_ack,"
            << "loop_closing_idle_ack,loop_closing_stopped_ack,"
            << "gba_stopped_ack,freeze_epoch\n";
    }
    if (capture_replay_state) {
        replay_execution_state.open(
            (output_dir / "replay_execution_state.csv").string());
        if (!replay_execution_state.is_open()) {
            std::cerr << "Cannot create replay execution-state log"
                      << std::endl;
            return 1;
        }
        replay_execution_state
            << "frame,timestamp,valid,tracking_state,is_keyframe,"
            << "prior_would_use_id,prior_selected_id,internal_frame_id,"
            << "map_id,map_generation,reference_kf_id,current_gray_hash,"
            << "current_depth_hash,current_static_mask_hash,"
            << "current_descriptors_hash,feature_count,motion_model_ran,"
            << "motion_previous_pose_hash,motion_static_initial_pose_hash,"
            << "motion_static_support_hash,motion_static_support_count,"
            << "motion_static_inliers,motion_static_optimized_pose_hash,"
            << "motion_dynamic_initial_pose_hash,"
            << "motion_dynamic_support_hash,motion_dynamic_support_count,"
            << "motion_dynamic_inliers,motion_dynamic_optimized_pose_hash,"
            << "motion_output_pose_hash,local_map_ran,"
            << "local_map_entry_pose_hash,local_keyframe_sequence_hash,"
            << "local_keyframe_count,local_candidate_state_hash,"
            << "local_candidate_id_hash,local_candidate_position_hash,"
            << "local_candidate_normal_hash,"
            << "local_candidate_distance_hash,"
            << "local_candidate_descriptor_hash,"
            << "local_candidate_observation_hash,"
            << "local_candidate_count,local_map_support_hash,"
            << "local_map_support_count,local_map_optimizer_inliers,"
            << "local_map_optimized_pose_hash,current_pose_hash,"
            << "map_point_count,keyframe_count,map_point_hash,"
            << "keyframe_hash,map_fingerprint,freeze_protocol,"
            << "freeze_requested,local_mapping_idle_ack,"
            << "local_mapping_stopped_ack,loop_closing_idle_ack,"
            << "loop_closing_stopped_ack,gba_stopped_ack,freeze_epoch\n";
    }
    std::ofstream all_frame_trajectory(
        (output_dir / "CameraTrajectory_AllFrames_TUM.txt").string());
    if (!all_frame_trajectory.is_open()) {
        std::cerr << "Cannot open all-frame trajectory log" << std::endl;
        return 1;
    }
    all_frame_trajectory << std::fixed;
    int processed_frames = 0;
    int tracked_frames = 0;
    int failed_frames = 0;
    int accepted_motion_priors = 0;
    int raw_motion_priors = 0;
    int candidate_motion_priors = 0;
    int consensus_pass_motion_priors = 0;
    int direct_pass_motion_priors = 0;
    int common_support_pass_motion_priors = 0;
    int would_use_motion_priors = 0;
    int matched_oracle_evaluations = 0;
    int matched_oracle_dynamic_preferences = 0;
    int matched_oracle_applied = 0;
    int used_motion_priors = 0;
    int shadow_translation_applied_frames = 0;
    int shadow_translation_factor_injections = 0;
    int velocity_neutralized_frames = 0;
    int shadow_translation_stopped_frames = 0;
    int shadow_translation_propagation_rejections = 0;
    int counterfactual_pose_pairs = 0;
    int gaussian_background_render_frames = 0;
    int gaussian_background_candidate_frames = 0;
    int gaussian_background_gate_pass_frames = 0;
    int gaussian_background_overlap_failures = 0;
    int carried_forward_frames = 0;
    int mask_pose_predicted_frames = 0;
    int temporal_refinement_frames = 0;
    std::uint64_t temporal_recovered_static_pixels = 0;
    std::uint64_t temporal_added_dynamic_pixels = 0;
    int temporal_flow_guard_valid_frames = 0;
    std::uint64_t temporal_flow_guard_rejected_pixels = 0;
    int temporal_recovery_risk_active_frames = 0;
    int temporal_recovery_risk_blocked_frames = 0;
    std::uint64_t tracking_recovery_candidate_pixels = 0;
    std::uint64_t tracking_recovery_blocked_pixels = 0;
    std::uint64_t tracking_recovery_audit_pixels = 0;
    std::uint64_t tracking_recovery_mapping_leak_pixels = 0;
    std::uint64_t extracted_features_total = 0;
    double static_mask_ratio_sum = 0.0;
    int mapping_weight_frames = 0;
    double mapping_static_ratio_sum = 0.0;
    int adaptive_feature_active_frames = 0;
    int adaptive_feature_hold_remaining = 0;
    int temporal_recovery_risk_hold_remaining = 0;
    int previous_tracking_inliers =
        std::numeric_limits<int>::max();
    bool adaptive_feature_has_inlier_history = false;
    bool temporal_recovery_risk_has_inlier_history = false;
    double adaptive_fast_threshold_sum = 0.0;
    struct BufferedMotionPrior {
        int frame = -1;
        Motion3D::MotionMarginalizedPosePrior::Result result;
    };
    std::deque<BufferedMotionPrior> motion_prior_shuffle_buffer;
    std::vector<HeldOutFrame> heldout_frames;
    int last_heldout_frame = -heldout_stride;
    bool local_mapping_sync_failed = false;
    bool loop_closing_sync_failed = false;
    bool replay_freeze_failed = false;
    bool instrumentation_failed = false;
    bool instrumentation_written = false;
    FrontendEvidenceFrame previous_instrumented_frame;
    cv::Mat background_support_previous_bgr;
    cv::Mat background_support_previous_depth;
    cv::Mat background_support_previous_static_mask;
    Sophus::SE3f background_support_previous_tcw;
    bool background_support_previous_valid = false;

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "DyGeoFusion-SLAM+ - Dynamic Scene Processing" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

    // Main loop
    cv::Mat imBGR, imRGB, imD;
    for (int ni = 0; ni < nImages; ni++)
    {
        const auto frame_wall_start = std::chrono::steady_clock::now();
        if (pSLAM->isShutDown())
            break;

        // Read image and depthmap from file.
        imBGR = cv::imread(
            std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni],
            cv::IMREAD_UNCHANGED);
        imD = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imBGR.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr << std::endl << "Failed to load depth image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        if (!DyGeoFusion::resizeRgbdToCameraSize(
                imBGR, imD, cv::Size(cam_width, cam_height))) {
            std::cerr
                << "[DyGeoFusion-SLAM+] Invalid RGB-D camera size"
                << std::endl;
            return 1;
        }
        // Semantic and motion modules consume OpenCV BGR. ORB-SLAM3 and the
        // inherited Photo-SLAM mapper consume RGB for Camera.RGB=1.
        imRGB = DyGeoFusion::openCvBgrToRgb(imBGR);

        // Convert depth to float meters
        cv::Mat imD_float;
        if (imD.type() == CV_16UC1) {
            imD.convertTo(imD_float, CV_32FC1, depth_to_meters);
        } else {
            imD.convertTo(imD_float, CV_32FC1);
        }

        // Compute dynamic mask
        std::chrono::steady_clock::time_point t_mask_start = std::chrono::steady_clock::now();

        Sophus::SE3f maskPoseTcw = lastValidTcw;
        bool mask_pose_predicted = false;
        if (mask_cfg.use_temporal_background_refinement &&
            hasValidPose && hasPreviousValidPose) {
            const Sophus::SE3f increment =
                lastValidTcw * previousValidTcw.inverse();
            const float increment_translation =
                increment.translation().norm();
            const float increment_rotation =
                increment.so3().log().norm();
            const Sophus::SE3f predicted =
                increment * lastValidTcw;
            if (increment_translation <=
                    mask_cfg.temporal_prediction_max_translation &&
                increment_rotation <=
                    mask_cfg.temporal_prediction_max_rotation &&
                predicted.matrix().allFinite()) {
                maskPoseTcw = predicted;
                mask_pose_predicted = true;
            }
        }

        // Render the current background at a bounded constant-velocity pose.
        cv::Mat renderedDepth;
        if (pGausMapper &&
            (mask_cfg.use_depth_consistency ||
             mask_cfg.use_temporal_background_refinement) &&
            hasValidPose) {
            renderedDepth = pGausMapper->renderDepthFromPose(
                maskPoseTcw, cam_width, cam_height,
                cam_fx, cam_fy, cam_cx, cam_cy,
                mask_cfg.temporal_min_render_opacity);
        }

        // Compute mask with rendered depth for geometric consistency
        cv::Mat staticMask = pMaskRefiner->compute(
            imBGR, imD_float, maskPoseTcw,
            renderedDepth.empty() ? nullptr : &renderedDepth,
            tframe, hasValidPose);

        const int recovery_risk_previous_inliers =
            previous_tracking_inliers ==
                    std::numeric_limits<int>::max()
                ? -1
                : previous_tracking_inliers;
        if (temporal_recovery_risk_gate &&
            temporal_recovery_risk_has_inlier_history &&
            previous_tracking_inliers <
                mask_cfg.temporal_recovery_min_previous_inliers) {
            temporal_recovery_risk_hold_remaining = std::max(
                temporal_recovery_risk_hold_remaining,
                mask_cfg.temporal_recovery_hold_frames);
        }
        const int recovery_risk_hold_for_frame =
            temporal_recovery_risk_hold_remaining;
        const bool temporal_recovery_risk_active =
            temporal_recovery_risk_gate &&
            recovery_risk_hold_for_frame > 0;
        const bool temporal_recovery_allowed =
            !temporal_recovery_risk_gate ||
            temporal_recovery_risk_active;
        const cv::Mat raw_dynamic =
            pMaskRefiner->getRawDynamicMask();
        const cv::Mat temporal_static_mask = staticMask.clone();
        int frame_tracking_recovery_candidate_pixels = 0;
        if (!raw_dynamic.empty() &&
            raw_dynamic.type() == CV_8UC1 &&
            raw_dynamic.size() == temporal_static_mask.size()) {
            cv::Mat candidate_tracking_support;
            cv::bitwise_and(
                raw_dynamic, temporal_static_mask,
                candidate_tracking_support);
            frame_tracking_recovery_candidate_pixels =
                cv::countNonZero(candidate_tracking_support);
        }
        int frame_tracking_recovery_blocked_pixels = 0;
        if (!temporal_recovery_allowed) {
            const cv::Mat raw_static =
                pMaskRefiner->getRawStaticMask();
            if (!raw_static.empty() &&
                raw_static.type() == CV_8UC1 &&
                raw_static.size() == staticMask.size()) {
                staticMask = raw_static;
                frame_tracking_recovery_blocked_pixels =
                    frame_tracking_recovery_candidate_pixels;
            }
        }
        if (temporal_recovery_risk_gate &&
            temporal_recovery_risk_hold_remaining > 0) {
            --temporal_recovery_risk_hold_remaining;
        }
        if (export_static_masks) {
            cv::Mat static_mask_u8;
            staticMask.convertTo(static_mask_u8, CV_8UC1, 255.0);
            std::ostringstream mask_name;
            mask_name << std::setfill('0') << std::setw(6) << ni << ".png";
            if (!cv::imwrite(
                    (exported_static_mask_dir / mask_name.str()).string(),
                    static_mask_u8)) {
                std::cerr << "Failed to export static mask for frame "
                          << ni << std::endl;
                return 1;
            }
        }

        // DyGeoFusion-SLAM+: Compute soft weight for 3DGS optimization
        // ONLY enable soft weight when use_depth_consistency is true (geometric prior active)
        // Otherwise, use pure YOLO hard mask for baseline behavior
        // Compute two independent soft weights and fuse them (E2: DINO-Flow + 3-prior fusion)
        cv::Mat dino_weight;   // from DINO-Flow consensus bin
        cv::Mat pdyn_weight;   // from DyGeoFusion 3-prior P_dyn
        const float alpha = mask_cfg.soft_mapping_alpha;
        const float min_weight = mask_cfg.soft_mapping_min_weight;

        // --- DINO-Flow weight ---
        if (mask_cfg.use_soft_mapping && !dinov2_maps_dir.empty()) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << tframe;
            std::string bin_path = dinov2_maps_dir + "/" + oss.str() + ".bin";
            if (std::filesystem::exists(bin_path)) {
                std::ifstream fin(bin_path, std::ios::binary);
                int h = imBGR.rows, w = imBGR.cols;
                cv::Mat dyn_map(h, w, CV_32FC1);
                const std::streamsize expected_bytes =
                    static_cast<std::streamsize>(h) * w * sizeof(float);
                fin.read(reinterpret_cast<char*>(dyn_map.data), expected_bytes);
                const bool has_expected_size =
                    fin.gcount() == expected_bytes && fin.peek() == std::ifstream::traits_type::eof();
                if (has_expected_size) {
                    dino_weight = cv::Mat(h, w, CV_32FC1);
                    for (int y = 0; y < h; ++y) {
                        float* pd = dyn_map.ptr<float>(y);
                        float* pw = dino_weight.ptr<float>(y);
                        for (int x = 0; x < w; ++x) {
                            const float probability = std::isfinite(pd[x])
                                ? std::clamp(pd[x], 0.0f, 1.0f)
                                : 0.0f;
                            pd[x] = probability;
                            pw[x] = std::max(min_weight, 1.0f - alpha * probability);
                        }
                    }
                    const bool has_semantic_mask_pipeline = mask_cfg.use_external_mask ||
                        (mask_cfg.use_yolo && !mask_cfg.yolo_model_path.empty());
                    if (!has_semantic_mask_pipeline) {
                        cv::Mat dyn_binary;
                        cv::threshold(dyn_map, dyn_binary, 0.5f, 255.0f, cv::THRESH_BINARY_INV);
                        dyn_binary.convertTo(staticMask, CV_8UC1);
                    }
                } else {
                    std::cerr << "[DyGeoFusion+DINOv2] Invalid map size: " << bin_path
                              << " (expected " << expected_bytes << " bytes)" << std::endl;
                }
            } else if (ni < 5 || ni % 100 == 0) {
                std::cerr << "[DyGeoFusion+DINOv2] Map not found: " << bin_path << std::endl;
            }
        }

        // --- 3-prior P_dyn weight (independent of DINO-Flow) ---
        pdyn_weight = pMaskRefiner->getStaticMappingWeight();
        if (mask_cfg.use_hard_mapping_mask) {
            staticMask.convertTo(pdyn_weight, CV_32FC1);
        }
        if (mask_cfg.temporal_conservative_mapping) {
            const cv::Mat raw_static = pMaskRefiner->getRawStaticMask();
            if (!raw_static.empty()) {
                raw_static.convertTo(pdyn_weight, CV_32FC1);
            }
        }

        // --- Fuse: pixel-wise multiply, re-clamp to min_weight ---
        cv::Mat staticWeight;
        if (!dino_weight.empty() && !pdyn_weight.empty()) {
            cv::multiply(dino_weight, pdyn_weight, staticWeight);
            cv::max(staticWeight, min_weight, staticWeight);
        } else if (!dino_weight.empty()) {
            staticWeight = dino_weight;
        } else if (!pdyn_weight.empty()) {
            staticWeight = pdyn_weight;
        }
        // When neither is active, staticWeight stays empty (original behavior)

        // Audit the exact action boundary. A recovered Tracking pixel is one
        // that was dynamic in the raw semantic-plus-flow mask but static in
        // the final feature mask. Conservative mapping must keep every such
        // pixel below the static-support threshold. This is diagnostic only;
        // it does not modify either mask.
        int frame_tracking_recovery_audit_pixels = 0;
        int frame_tracking_recovery_mapping_leak_pixels = 0;
        if (mask_cfg.temporal_recovery_only) {
            const cv::Mat raw_dynamic =
                pMaskRefiner->getRawDynamicMask();
            if (!raw_dynamic.empty() &&
                raw_dynamic.type() == CV_8UC1 &&
                raw_dynamic.size() == staticMask.size()) {
                cv::Mat recovered_tracking_support;
                cv::bitwise_and(
                    raw_dynamic, staticMask,
                    recovered_tracking_support);
                frame_tracking_recovery_audit_pixels =
                    cv::countNonZero(recovered_tracking_support);
                if (frame_tracking_recovery_audit_pixels > 0) {
                    if (staticWeight.empty()) {
                        // No mapping weight means the mapper would treat all
                        // image support as admissible. Fail the audit closed.
                        frame_tracking_recovery_mapping_leak_pixels =
                            frame_tracking_recovery_audit_pixels;
                    } else {
                        cv::Mat mapping_static_support;
                        cv::compare(
                            staticWeight, 0.5f,
                            mapping_static_support, cv::CMP_GE);
                        cv::Mat recovered_mapping_overlap;
                        cv::bitwise_and(
                            recovered_tracking_support,
                            mapping_static_support,
                            recovered_mapping_overlap);
                        frame_tracking_recovery_mapping_leak_pixels =
                            cv::countNonZero(
                                recovered_mapping_overlap);
                    }
                }
            }
        }

        if (background_support_router) {
            cv::Mat dynamic_source =
                pMaskRefiner->getRawDynamicMask();
            if (dynamic_source.empty()) {
                dynamic_source = pMaskRefiner->getSemanticMask();
            }
            const auto support = background_support_router->select(
                imBGR, imD_float, dynamic_source, staticMask,
                renderedDepth,
                pMaskRefiner->getMotionMask());
            background_support_metrics
                << ni << ',' << tframe << ','
                << support.dynamic_pixels << ','
                << support.ring_pixels << ','
                << support.candidate_points << ','
                << support.selected_points << ','
                << support.support_ratio << ','
                << support.median_depth_residual << ','
                << (support.valid ? 1 : 0) << '\n';
        }

        std::chrono::steady_clock::time_point t_mask_end = std::chrono::steady_clock::now();
        double tmask = std::chrono::duration_cast<std::chrono::duration<double>>(t_mask_end - t_mask_start).count();
        vTimesMask[ni] = tmask;

        Motion3D::MotionMarginalizedPosePrior::Result raw_motion_prior_result;
        const auto motion_prior_start = std::chrono::steady_clock::now();
        if (motion_pose_prior) {
            raw_motion_prior_result = motion_pose_prior->prepare(
                imBGR, imD_float, pMaskRefiner->getSemanticMask(), tframe);
        }
        Motion3D::MotionMarginalizedPosePrior::Result motion_prior_result =
            raw_motion_prior_result;
        int motion_prior_source_frame = ni;
        const int shuffle_lag =
            mask_cfg.motion_prior_shuffle_lag_frames;
        if (shuffle_lag > 0) {
            motion_prior_shuffle_buffer.push_back(
                BufferedMotionPrior{ni, raw_motion_prior_result});
            motion_prior_result =
                Motion3D::MotionMarginalizedPosePrior::Result();
            motion_prior_source_frame = -1;
            if (motion_prior_shuffle_buffer.size() >
                static_cast<std::size_t>(shuffle_lag)) {
                motion_prior_result =
                    motion_prior_shuffle_buffer.front().result;
                motion_prior_source_frame =
                    motion_prior_shuffle_buffer.front().frame;
                motion_prior_shuffle_buffer.pop_front();
            }
        }
        const double motion_prior_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - motion_prior_start).count();

        if (adaptive_feature_risk_gate &&
            adaptive_feature_has_inlier_history &&
            previous_tracking_inliers <
                mask_cfg.adaptive_feature_min_previous_inliers) {
            adaptive_feature_hold_remaining = std::max(
                adaptive_feature_hold_remaining,
                mask_cfg.adaptive_feature_hold_frames);
        }
        const bool adaptive_feature_active =
            mask_cfg.use_adaptive_feature_extraction &&
            (!adaptive_feature_risk_gate ||
             adaptive_feature_hold_remaining > 0);
        const int adaptive_feature_previous_inliers =
            previous_tracking_inliers ==
                    std::numeric_limits<int>::max()
                ? -1
                : previous_tracking_inliers;
        if (adaptive_feature_risk_gate) {
            pSLAM->ConfigureAdaptiveMaskFeatures(
                adaptive_feature_active,
                mask_cfg.adaptive_feature_relaxation);
        }
        if (adaptive_feature_active) {
            ++adaptive_feature_active_frames;
        }
        if (adaptive_feature_risk_gate &&
            adaptive_feature_hold_remaining > 0) {
            --adaptive_feature_hold_remaining;
        }

        // Track with mask - pass staticMask to filter out dynamic features
        // Also pass staticWeight for soft-weight 3DGS optimization (higher weight = more static)
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image, mask, and soft weight to the SLAM system
        // - staticMask: hard binary mask for feature extraction (255=static, 0=dynamic)
        // - staticWeight: soft weight for 3DGS loss (higher=more static, CV_32FC1 in [0,1])
        const Sophus::SE3f* relative_pose_prior =
            motion_prior_result.valid ? &motion_prior_result.relative_pose : nullptr;
        const Motion3D::MotionMarginalizedPosePrior::Matrix6f*
            relative_pose_information =
                motion_prior_result.valid
                    ? &motion_prior_result.information
                    : nullptr;
        const MatchedOraclePose* matched_oracle_pose =
            matched_oracle_mode != "disabled"
                ? &matched_oracle_poses[ni]
                : nullptr;
        const Sophus::SE3f* matched_oracle_tcw =
            matched_oracle_pose && matched_oracle_pose->valid
                ? &matched_oracle_pose->tcw
                : nullptr;
        const bool matched_oracle_apply =
            matched_oracle_mode == "apply" &&
            matched_oracle_tcw != nullptr;
        const bool shadow_only_this_frame =
            mask_cfg.motion_pose_prior_shadow_only ||
            (matched_oracle_mode != "disabled" &&
             matched_oracle_tcw == nullptr);
        Sophus::SE3f currentPose = pSLAM->TrackRGBD(
            imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(),
            vstrImageFilenamesRGB[ni], staticMask, staticWeight,
            relative_pose_prior, relative_pose_information,
            shadow_only_this_frame, matched_oracle_tcw,
            matched_oracle_apply);
        if (instrument_frame >= 0 &&
            (ni == instrument_frame - 1 || ni == instrument_frame)) {
            FrontendEvidenceFrame frontend_frame;
            frontend_frame.frame_id = ni;
            frontend_frame.timestamp = tframe;
            frontend_frame.bgr = imBGR.clone();
            frontend_frame.keypoints =
                pSLAM->GetTrackedKeyPointsUn();
            frontend_frame.descriptors =
                pSLAM->GetTrackedDescriptors();
            frontend_frame.map_point_ids =
                pSLAM->GetTrackedMapPointIds();
            frontend_frame.outlier_flags =
                pSLAM->GetTrackedOutlierFlags();
            if (ni == instrument_frame - 1) {
                previous_instrumented_frame =
                    std::move(frontend_frame);
            } else {
                const std::filesystem::path dataset_root(argv[4]);
                instrumentation_written = writeFrontendEvidence(
                    instrument_output_dir,
                    previous_instrumented_frame, frontend_frame, imD,
                    pMaskRefiner->getSemanticMask(), staticMask,
                    dataset_root /
                        vstrImageFilenamesRGB[instrument_frame - 1],
                    dataset_root / vstrImageFilenamesRGB[instrument_frame],
                    dataset_root / vstrImageFilenamesD[instrument_frame],
                    std::filesystem::path(argv[5]),
                    std::filesystem::path(argv[2]),
                    std::filesystem::path(mask_config_path));
                if (!instrumentation_written) {
                    std::cerr
                        << "Failed to export claim-grade instrumentation "
                        << "for frame " << instrument_frame << std::endl;
                    instrumentation_failed = true;
                    break;
                }
            }
        }
        if (synchronize_local_mapping &&
            !pSLAM->WaitForLocalMappingIdle(30.0)) {
            std::cerr << "Timed out waiting for LocalMapping after frame "
                      << ni << std::endl;
            local_mapping_sync_failed = true;
            break;
        }
        if (synchronize_loop_closing &&
            !pSLAM->WaitForLoopClosingIdle(30.0)) {
            std::cerr << "Timed out waiting for LoopClosing after frame "
                      << ni << std::endl;
            loop_closing_sync_failed = true;
            break;
        }
        if (ni == freeze_map_after_frame &&
            !pSLAM->FreezeMapForDeterministicReplay(30.0)) {
            std::cerr
                << "Timed out freezing map mutation after frame "
                << ni << std::endl;
            replay_freeze_failed = true;
            break;
        }
        if (capture_replay_state) {
            // The state captured inside Track() can precede asynchronous
            // backend completion. Refresh after the requested barriers.
            pSLAM->RefreshExternalPosePriorReplayExecutionState();
        }

        const int tracking_state = pSLAM->GetTrackingState();
        const bool current_tracking_valid =
            tracking_state == ORB_SLAM3::Tracking::OK ||
            tracking_state == ORB_SLAM3::Tracking::OK_KLT;
        const int current_tracking_inliers = current_tracking_valid
            ? std::max(0, pSLAM->GetMatchesInliers())
            : 0;

        Motion3D::GaussianBackgroundPoseRefiner::Result
            gaussian_background_pose_result;
        bool gaussian_background_render_available = false;
        int gaussian_background_map_iteration =
            pGausMapper ? pGausMapper->getIteration() : -1;
        Sophus::SE3f gaussian_background_candidate_tcw = currentPose;
        if (gaussian_background_pose_refiner &&
            pGausMapper &&
            current_tracking_valid &&
            currentPose.matrix().allFinite()) {
            // The mask renderer above uses a bounded prediction available
            // before tracking.  This shadow mechanism deliberately renders
            // again at the exact, immutable Parent pose returned by TrackRGBD.
            const cv::Mat gaussian_depth_at_parent =
                pGausMapper->renderDepthFromPose(
                    currentPose, cam_width, cam_height,
                    cam_fx, cam_fy, cam_cx, cam_cy,
                    mask_cfg.temporal_min_render_opacity);
            gaussian_background_map_iteration =
                pGausMapper->getIteration();
            gaussian_background_render_available =
                !gaussian_depth_at_parent.empty();
            if (gaussian_background_render_available) {
                cv::Mat final_dynamic_mask;
                cv::compare(
                    staticMask, 0, final_dynamic_mask, cv::CMP_EQ);
                gaussian_background_pose_result =
                    gaussian_background_pose_refiner->refine(
                        imD_float, staticMask, final_dynamic_mask,
                        gaussian_depth_at_parent, currentPose);
                if (gaussian_background_pose_result.candidate_available) {
                    gaussian_background_candidate_tcw =
                        gaussian_background_pose_result.refined_tcw;
                }
            }
        }
        if (gaussian_background_render_available) {
            ++gaussian_background_render_frames;
        }
        if (gaussian_background_pose_result.candidate_available) {
            ++gaussian_background_candidate_frames;
        }
        if (gaussian_background_pose_result.gate_pass) {
            ++gaussian_background_gate_pass_frames;
        }
        if (gaussian_background_pose_result.proposal_validation_overlap != 0) {
            ++gaussian_background_overlap_failures;
        }
        if (gaussian_background_pose_metrics) {
            const Sophus::SE3f parent_twc =
                current_tracking_valid && currentPose.matrix().allFinite()
                    ? currentPose.inverse()
                    : Sophus::SE3f();
            const Sophus::SE3f candidate_twc =
                gaussian_background_pose_result.candidate_available
                    ? gaussian_background_candidate_tcw.inverse()
                    : parent_twc;
            const Eigen::Quaternionf parent_quaternion =
                parent_twc.unit_quaternion();
            const Eigen::Quaternionf candidate_quaternion =
                candidate_twc.unit_quaternion();
            gaussian_background_pose_metrics
                << ni << ',' << tframe << ','
                << gaussian_background_map_iteration << ','
                << (gaussian_background_render_available ? 1 : 0) << ','
                << gaussian_background_pose_result.dynamic_pixels << ','
                << gaussian_background_pose_result.safe_static_pixels << ','
                << gaussian_background_pose_result.proposal_points << ','
                << gaussian_background_pose_result.validation_points << ','
                << gaussian_background_pose_result
                       .proposal_validation_overlap
                << ','
                << gaussian_background_pose_result.proposal_support << ','
                << gaussian_background_pose_result
                       .validation_common_support
                << ','
                << gaussian_background_pose_result.observable_rank << ','
                << (gaussian_background_pose_result.candidate_available
                        ? 1 : 0)
                << ','
                << (gaussian_background_pose_result.proposal_improved
                        ? 1 : 0)
                << ','
                << (gaussian_background_pose_result.validation_valid
                        ? 1 : 0)
                << ','
                << (gaussian_background_pose_result
                            .validation_prefers_candidate
                        ? 1 : 0)
                << ','
                << (gaussian_background_pose_result.gate_pass ? 1 : 0)
                << ','
                << gaussian_background_pose_result.proposal_parent_loss
                << ','
                << gaussian_background_pose_result.proposal_candidate_loss
                << ','
                << gaussian_background_pose_result.validation_parent_loss
                << ','
                << gaussian_background_pose_result
                       .validation_candidate_loss
                << ','
                << gaussian_background_pose_result.translation_correction
                << ','
                << gaussian_background_pose_result.rotation_correction
                << ','
                << parent_twc.translation().x() << ','
                << parent_twc.translation().y() << ','
                << parent_twc.translation().z() << ','
                << parent_quaternion.x() << ','
                << parent_quaternion.y() << ','
                << parent_quaternion.z() << ','
                << parent_quaternion.w() << ','
                << candidate_twc.translation().x() << ','
                << candidate_twc.translation().y() << ','
                << candidate_twc.translation().z() << ','
                << candidate_quaternion.x() << ','
                << candidate_quaternion.y() << ','
                << candidate_quaternion.z() << ','
                << candidate_quaternion.w()
                << '\n';
        }

        Motion3D::BackgroundSupportPoseRefiner::Result
            background_support_pose_result;
        Motion3D::DirectRgbdPoseValidator::Result
            background_support_direct_result;
        bool background_support_direct_prefers_candidate = false;
        bool background_support_candidate_pose_available = false;
        Sophus::SE3f background_support_candidate_tcw = currentPose;
        if (background_support_pose_refiner &&
            background_support_pose_validator &&
            background_support_previous_valid &&
            current_tracking_valid &&
            currentPose.matrix().allFinite()) {
            const Sophus::SE3f parent_relative =
                currentPose * background_support_previous_tcw.inverse();
            cv::Mat dynamic_source = pMaskRefiner->getRawDynamicMask();
            if (dynamic_source.empty()) {
                dynamic_source = pMaskRefiner->getSemanticMask();
            }
            background_support_pose_result =
                background_support_pose_refiner->refine(
                    background_support_previous_bgr, imBGR,
                    background_support_previous_depth, imD_float,
                    background_support_previous_static_mask, staticMask,
                    dynamic_source, renderedDepth,
                    pMaskRefiner->getMotionMask(), parent_relative);
            if (background_support_pose_result.geometric_support > 0) {
                background_support_candidate_tcw =
                    background_support_pose_result.refined_relative_pose *
                    background_support_previous_tcw;
                background_support_candidate_pose_available =
                    background_support_candidate_tcw.matrix().allFinite();
                Motion3D::DirectRgbdPoseValidator::Intrinsics intrinsics{
                    cam_fx, cam_fy, cam_cx, cam_cy};
                background_support_direct_result =
                    background_support_pose_validator->score(
                        background_support_previous_bgr, imBGR,
                        background_support_previous_depth, imD_float,
                        background_support_previous_static_mask, staticMask,
                        background_support_previous_tcw, currentPose,
                        background_support_candidate_tcw, intrinsics);
                background_support_direct_prefers_candidate =
                    Motion3D::DirectRgbdPoseValidator::prefersDynamic(
                        background_support_direct_result,
                        Motion3D::DirectRgbdPoseValidator::ScoreMode::Combined);
            }
        }
        if (background_support_pose_metrics) {
            const bool background_support_gate_pass =
                background_support_pose_result.valid &&
                background_support_direct_result.valid &&
                background_support_direct_prefers_candidate;
            const Sophus::SE3f parent_twc = currentPose.inverse();
            const Sophus::SE3f candidate_twc =
                background_support_candidate_pose_available
                    ? background_support_candidate_tcw.inverse()
                    : parent_twc;
            const Eigen::Quaternionf parent_quaternion =
                parent_twc.unit_quaternion();
            const Eigen::Quaternionf candidate_quaternion =
                candidate_twc.unit_quaternion();
            background_support_pose_metrics
                << ni << ',' << tframe << ','
                << (background_support_candidate_pose_available ? 1 : 0)
                << ','
                << (background_support_gate_pass ? 1 : 0) << ','
                << (background_support_pose_result.valid ? 1 : 0) << ','
                << (background_support_pose_result.improved ? 1 : 0) << ','
                << background_support_pose_result.routed_support << ','
                << background_support_pose_result.geometric_support << ','
                << background_support_pose_result.inliers << ','
                << background_support_pose_result.initial_rmse << ','
                << background_support_pose_result.final_rmse << ','
                << background_support_pose_result.translation_correction << ','
                << background_support_pose_result.rotation_correction << ','
                << (background_support_direct_result.valid ? 1 : 0) << ','
                << (background_support_direct_prefers_candidate ? 1 : 0)
                << ','
                << background_support_direct_result.common_support << ','
                << background_support_direct_result.static_combined_score
                << ','
                << background_support_direct_result.dynamic_combined_score
                << ','
                << parent_twc.translation().x() << ','
                << parent_twc.translation().y() << ','
                << parent_twc.translation().z() << ','
                << parent_quaternion.x() << ','
                << parent_quaternion.y() << ','
                << parent_quaternion.z() << ','
                << parent_quaternion.w() << ','
                << candidate_twc.translation().x() << ','
                << candidate_twc.translation().y() << ','
                << candidate_twc.translation().z() << ','
                << candidate_quaternion.x() << ','
                << candidate_quaternion.y() << ','
                << candidate_quaternion.z() << ','
                << candidate_quaternion.w()
                << '\n';
        }
        if (current_tracking_inliers > 0) {
            adaptive_feature_has_inlier_history = true;
            temporal_recovery_risk_has_inlier_history = true;
        }
        previous_tracking_inliers = current_tracking_inliers;
        const bool prior_candidate = pSLAM->WasExternalPosePriorCandidate();
        const bool prior_consensus_pass =
            pSLAM->ExternalPosePriorConsensusPass();
        const bool prior_direct_pass =
            pSLAM->ExternalPosePriorDirectPass();
        const bool prior_common_support_pass =
            pSLAM->ExternalPosePriorCommonSupportPass();
        const bool prior_would_use = pSLAM->WouldUseExternalPosePrior();
        const bool prior_used = pSLAM->WasExternalPosePriorUsed();
        const auto matched_oracle_result =
            pSLAM->ExternalPosePriorOracleResult();
        const bool matched_oracle_applied_this_frame =
            pSLAM->ExternalPosePriorOracleApplied();
        const bool shadow_translation_active =
            pSLAM->ExternalPosePriorShadowTranslationActive();
        const bool shadow_translation_applied =
            pSLAM->ExternalPosePriorShadowTranslationApplied();
        const int shadow_translation_remaining =
            pSLAM->ExternalPosePriorShadowTranslationRemaining();
        const float shadow_translation_innovation =
            pSLAM->ExternalPosePriorShadowTranslationInnovation();
        const std::string shadow_translation_stop_reason =
            pSLAM->ExternalPosePriorShadowTranslationStopReason();
        const bool shadow_translation_factor_injected =
            pSLAM->ExternalPosePriorShadowTranslationFactorInjected();
        const bool posterior_valid =
            pSLAM->ExternalPosePriorPosteriorValid();
        const bool posterior_pass =
            pSLAM->ExternalPosePriorPosteriorPass();
        const int posterior_static_inliers =
            pSLAM->ExternalPosePriorPosteriorStaticInliers();
        const int posterior_dynamic_inliers =
            pSLAM->ExternalPosePriorPosteriorDynamicInliers();
        const int posterior_common_support =
            pSLAM->ExternalPosePriorPosteriorCommonSupport();
        const float posterior_static_score =
            pSLAM->ExternalPosePriorPosteriorStaticScore();
        const float posterior_dynamic_score =
            pSLAM->ExternalPosePriorPosteriorDynamicScore();
        const bool velocity_neutralized =
            pSLAM->ExternalPosePriorVelocityNeutralized();
        const int prior_static_matches = pSLAM->ExternalPosePriorStaticMatches();
        const int prior_dynamic_matches = pSLAM->ExternalPosePriorDynamicMatches();
        const int prior_static_inliers = pSLAM->ExternalPosePriorStaticInliers();
        const int prior_dynamic_inliers = pSLAM->ExternalPosePriorDynamicInliers();
        const int prior_common_support =
            pSLAM->ExternalPosePriorCommonSupportCount();
        const float prior_static_common_score =
            pSLAM->ExternalPosePriorStaticCommonScore();
        const float prior_dynamic_common_score =
            pSLAM->ExternalPosePriorDynamicCommonScore();
        const auto direct_validation =
            pSLAM->ExternalPosePriorDirectValidation();
        const auto replay_state =
            pSLAM->ExternalPosePriorReplayStateIdentity();
        const auto replay_packet =
            pSLAM->ExternalPosePriorReplayPacket();
        const auto replay_execution =
            pSLAM->ExternalPosePriorReplayExecutionState();
        if (export_replay_packets && replay_packet.valid &&
            Motion3D::replayPacketPastFreezeSettlingFrame(
                ni, freeze_map_after_frame)) {
            std::ostringstream packet_name;
            packet_name << "frame_" << std::setfill('0')
                        << std::setw(6) << ni << ".yml.gz";
            const std::filesystem::path packet_path =
                replay_packet_dir / packet_name.str();
            std::string replay_error;
            if (!Motion3D::writeFrozenReplayPacket(
                    replay_packet, packet_path.string(),
                    &replay_error)) {
                std::cerr << replay_error << std::endl;
                return 1;
            }
            replay_packet_index
                << ni << ',' << std::setprecision(9) << tframe << ','
                << packet_name.str() << ','
                << replay_state.previous_frame_id << ','
                << replay_state.map_id << ','
                << replay_state.map_generation << ','
                << replay_state.reference_kf_id << ','
                << Motion3D::hashToHex(
                    replay_state.map_point_state_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.static_support_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.dynamic_support_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.common_support_hash) << ','
                << replay_packet.freeze.protocol << ','
                << (replay_packet.freeze.requested ? 1 : 0) << ','
                << (replay_packet.freeze.local_mapping_idle_ack ? 1 : 0)
                << ','
                << (replay_packet.freeze.local_mapping_stopped_ack ? 1 : 0)
                << ','
                << (replay_packet.freeze.loop_closing_idle_ack ? 1 : 0)
                << ','
                << (replay_packet.freeze.loop_closing_stopped_ack ? 1 : 0)
                << ','
                << (replay_packet.freeze.gba_stopped_ack ? 1 : 0) << ','
                << replay_packet.freeze.epoch << '\n';
        }
        if (capture_replay_state) {
            const long long wouldUseId =
                prior_would_use && replay_execution.valid
                ? static_cast<long long>(replay_execution.frame_id)
                : -1;
            const long long selectedId =
                prior_used && replay_execution.valid
                ? static_cast<long long>(replay_execution.frame_id)
                : -1;
            replay_execution_state
                << ni << ',' << std::setprecision(9) << tframe << ','
                << (replay_execution.valid ? 1 : 0) << ','
                << tracking_state << ','
                << (replay_execution.is_keyframe ? 1 : 0) << ','
                << wouldUseId << ',' << selectedId << ','
                << replay_execution.frame_id << ','
                << replay_execution.map_id << ','
                << replay_execution.map_generation << ','
                << replay_execution.reference_kf_id << ','
                << Motion3D::hashToHex(
                    replay_execution.current_gray_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.current_depth_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.current_static_mask_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.current_descriptors_hash) << ','
                << replay_execution.feature_count << ','
                << (replay_execution.tracking.motion_model_ran ? 1 : 0)
                << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.motion_previous_pose_hash)
                << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .motion_static_initial_pose_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.motion_static_support_hash)
                << ','
                << replay_execution.tracking.motion_static_support_count
                << ','
                << replay_execution.tracking.motion_static_inliers << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .motion_static_optimized_pose_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .motion_dynamic_initial_pose_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.motion_dynamic_support_hash)
                << ','
                << replay_execution.tracking.motion_dynamic_support_count
                << ','
                << replay_execution.tracking.motion_dynamic_inliers << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .motion_dynamic_optimized_pose_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.motion_output_pose_hash)
                << ','
                << (replay_execution.tracking.local_map_ran ? 1 : 0)
                << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.local_map_entry_pose_hash)
                << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_keyframe_sequence_hash) << ','
                << replay_execution.tracking.local_keyframe_count << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_state_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_id_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_position_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_normal_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_distance_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_descriptor_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking
                        .local_candidate_observation_hash) << ','
                << replay_execution.tracking.local_candidate_count << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.local_map_support_hash)
                << ','
                << replay_execution.tracking.local_map_support_count
                << ','
                << replay_execution.tracking.local_map_optimizer_inliers
                << ','
                << Motion3D::hashToHex(
                    replay_execution.tracking.local_map_optimized_pose_hash)
                << ','
                << Motion3D::hashToHex(
                    replay_execution.current_pose_hash) << ','
                << replay_execution.map_point_count << ','
                << replay_execution.keyframe_count << ','
                << Motion3D::hashToHex(
                    replay_execution.map_point_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.keyframe_hash) << ','
                << Motion3D::hashToHex(
                    replay_execution.map_fingerprint) << ','
                << replay_execution.freeze.protocol << ','
                << (replay_execution.freeze.requested ? 1 : 0) << ','
                << (replay_execution.freeze.local_mapping_idle_ack ? 1 : 0)
                << ','
                << (replay_execution.freeze.local_mapping_stopped_ack ? 1 : 0)
                << ','
                << (replay_execution.freeze.loop_closing_idle_ack ? 1 : 0)
                << ','
                << (replay_execution.freeze.loop_closing_stopped_ack ? 1 : 0)
                << ','
                << (replay_execution.freeze.gba_stopped_ack ? 1 : 0) << ','
                << replay_execution.freeze.epoch << '\n';
        }
        if (pSLAM->HasExternalPosePriorHypotheses()) {
            const auto writeTwc = [&](const Sophus::SE3f& Tcw) {
                const Sophus::SE3f Twc = Tcw.inverse();
                const Eigen::Vector3f translation = Twc.translation();
                const Eigen::Quaternionf rotation = Twc.unit_quaternion();
                counterfactual_poses
                    << ',' << translation.x() << ',' << translation.y()
                    << ',' << translation.z() << ',' << rotation.x()
                    << ',' << rotation.y() << ',' << rotation.z()
                    << ',' << rotation.w();
            };
            counterfactual_poses
                << ni << ',' << tframe << ','
                << pSLAM->ExternalPosePriorPreviousTimestamp() << ','
                << (prior_consensus_pass ? 1 : 0) << ','
                << (direct_validation.valid ? 1 : 0) << ','
                << (prior_direct_pass ? 1 : 0) << ','
                << (prior_common_support_pass ? 1 : 0) << ','
                << (prior_would_use ? 1 : 0) << ','
                << (prior_used ? 1 : 0) << ','
                << (matched_oracle_result.valid ? 1 : 0) << ','
                << (matched_oracle_result.prefers_dynamic ? 1 : 0) << ','
                << (matched_oracle_applied_this_frame ? 1 : 0) << ','
                << matched_oracle_result.static_translation_error_m << ','
                << matched_oracle_result.dynamic_translation_error_m << ','
                << matched_oracle_result.static_rotation_error_rad << ','
                << matched_oracle_result.dynamic_rotation_error_rad << ','
                << prior_static_inliers << ',' << prior_dynamic_inliers << ','
                << prior_common_support << ',' << prior_static_common_score << ','
                << prior_dynamic_common_score << ','
                << direct_validation.common_support << ','
                << direct_validation.static_depth_score << ','
                << direct_validation.dynamic_depth_score << ','
                << direct_validation.static_photometric_score << ','
                << direct_validation.dynamic_photometric_score << ','
                << direct_validation.static_combined_score << ','
                << direct_validation.dynamic_combined_score << ','
                << replay_state.contract_version << ','
                << replay_state.hash_algorithm << ','
                << replay_state.previous_frame_id << ','
                << replay_state.map_id << ','
                << replay_state.map_generation << ','
                << replay_state.reference_kf_id << ','
                << Motion3D::hashToHex(
                    replay_state.reference_kf_pose_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.map_point_state_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.static_support_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.dynamic_support_hash) << ','
                << Motion3D::hashToHex(
                    replay_state.common_support_hash);
            writeTwc(pSLAM->ExternalPosePriorPreviousPose());
            writeTwc(pSLAM->ExternalPosePriorStaticPose());
            writeTwc(pSLAM->ExternalPosePriorDynamicPose());
            counterfactual_poses << '\n';
            ++counterfactual_pose_pairs;
        }
        const bool pose_valid = current_tracking_valid;
        if (pose_valid) {
            if (hasValidPose && lastFramePoseWasValid) {
                previousValidTcw = lastValidTcw;
                hasPreviousValidPose = true;
            } else {
                hasPreviousValidPose = false;
            }
            lastValidTcw = currentPose;
            hasValidPose = true;
            lastFramePoseWasValid = true;
        } else {
            hasPreviousValidPose = false;
            lastFramePoseWasValid = false;
        }
        const bool trajectory_carried_forward = !pose_valid;
        if (trajectory_carried_forward) ++carried_forward_frames;
        const Sophus::SE3f trajectory_Tcw = hasValidPose ? lastValidTcw : Sophus::SE3f();
        const Sophus::SE3f trajectory_Twc = trajectory_Tcw.inverse();
        const Eigen::Vector3f trajectory_translation = trajectory_Twc.translation();
        const Eigen::Quaternionf trajectory_rotation = trajectory_Twc.unit_quaternion();
        all_frame_trajectory
            << std::setprecision(6) << tframe << ' '
            << std::setprecision(9)
            << trajectory_translation.x() << ' ' << trajectory_translation.y() << ' '
            << trajectory_translation.z() << ' ' << trajectory_rotation.x() << ' '
            << trajectory_rotation.y() << ' ' << trajectory_rotation.z() << ' '
            << trajectory_rotation.w() << '\n';
        if (motion_pose_prior) {
            motion_pose_prior->commitCameraPose(currentPose, pose_valid);
        }
        if (background_support_pose_refiner) {
            if (current_tracking_valid && currentPose.matrix().allFinite()) {
                background_support_previous_bgr = imBGR.clone();
                background_support_previous_depth = imD_float.clone();
                background_support_previous_static_mask = staticMask.clone();
                background_support_previous_tcw = currentPose;
                background_support_previous_valid = true;
            } else {
                background_support_previous_bgr.release();
                background_support_previous_depth.release();
                background_support_previous_static_mask.release();
                background_support_previous_valid = false;
            }
        }
        if (heldout_stride > 0 && pose_valid && !pSLAM->CurrentFrameIsKeyFrame() &&
            ni > mask_cfg.warmup_frames && ni - last_heldout_frame >= heldout_stride) {
            heldout_frames.push_back(HeldOutFrame{
                ni, tframe, currentPose, imBGR.clone(), staticMask.clone()});
            last_heldout_frame = ni;
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        const bool lost = tracking_state == ORB_SLAM3::Tracking::RECENTLY_LOST ||
                          tracking_state == ORB_SLAM3::Tracking::LOST;
        const int dynamic_pixels = cv::countNonZero(1 - staticMask);
        const double dynamic_ratio = static_cast<double>(dynamic_pixels) /
                                     static_cast<double>(staticMask.total());
        const bool temporal_refinement_applied =
            pMaskRefiner->temporalRefinementApplied();
        const int recovered_static_pixels =
            pMaskRefiner->temporalRecoveredStaticPixels();
        const int added_dynamic_pixels =
            pMaskRefiner->temporalAddedDynamicPixels();
        const bool temporal_flow_guard_valid =
            pMaskRefiner->temporalFlowGuardValid();
        const int flow_guard_rejected_pixels =
            pMaskRefiner->temporalFlowGuardRejectedPixels();
        const double static_mask_ratio =
            pSLAM->GetLastStaticMaskRatio();
        const bool mapping_weight_present = !staticWeight.empty();
        double mapping_static_ratio = 1.0;
        if (mapping_weight_present) {
            cv::Mat mapping_static_pixels;
            cv::compare(
                staticWeight, 0.5f,
                mapping_static_pixels, cv::CMP_GE);
            mapping_static_ratio =
                static_cast<double>(
                    cv::countNonZero(mapping_static_pixels)) /
                static_cast<double>(staticWeight.total());
        }
        const int adaptive_fast_threshold =
            pSLAM->GetLastAdaptiveFastThreshold();
        const std::size_t extracted_features =
            pSLAM->GetTrackedKeyPointsUn().size();
        ++processed_frames;
        if (pose_valid) ++tracked_frames;
        if (lost) ++failed_frames;
        if (mask_pose_predicted) ++mask_pose_predicted_frames;
        if (temporal_refinement_applied) ++temporal_refinement_frames;
        temporal_recovered_static_pixels +=
            static_cast<std::uint64_t>(
                std::max(0, recovered_static_pixels));
        temporal_added_dynamic_pixels +=
            static_cast<std::uint64_t>(
                std::max(0, added_dynamic_pixels));
        if (temporal_flow_guard_valid) {
            ++temporal_flow_guard_valid_frames;
        }
        temporal_flow_guard_rejected_pixels +=
            static_cast<std::uint64_t>(
                std::max(0, flow_guard_rejected_pixels));
        if (temporal_recovery_risk_active) {
            ++temporal_recovery_risk_active_frames;
        }
        if (frame_tracking_recovery_blocked_pixels > 0) {
            ++temporal_recovery_risk_blocked_frames;
        }
        tracking_recovery_candidate_pixels +=
            static_cast<std::uint64_t>(std::max(
                0, frame_tracking_recovery_candidate_pixels));
        tracking_recovery_blocked_pixels +=
            static_cast<std::uint64_t>(std::max(
                0, frame_tracking_recovery_blocked_pixels));
        tracking_recovery_audit_pixels +=
            static_cast<std::uint64_t>(
                std::max(0, frame_tracking_recovery_audit_pixels));
        tracking_recovery_mapping_leak_pixels +=
            static_cast<std::uint64_t>(
                std::max(
                    0,
                    frame_tracking_recovery_mapping_leak_pixels));
        extracted_features_total +=
            static_cast<std::uint64_t>(extracted_features);
        static_mask_ratio_sum += static_mask_ratio;
        if (mapping_weight_present) {
            ++mapping_weight_frames;
            mapping_static_ratio_sum += mapping_static_ratio;
        }
        adaptive_fast_threshold_sum += adaptive_fast_threshold;
        if (raw_motion_prior_result.valid) ++raw_motion_priors;
        if (motion_prior_result.valid) ++accepted_motion_priors;
        if (prior_candidate) ++candidate_motion_priors;
        if (prior_consensus_pass) ++consensus_pass_motion_priors;
        if (prior_direct_pass) ++direct_pass_motion_priors;
        if (prior_common_support_pass) ++common_support_pass_motion_priors;
        if (prior_would_use) ++would_use_motion_priors;
        if (prior_used) ++used_motion_priors;
        if (matched_oracle_result.valid)
            ++matched_oracle_evaluations;
        if (matched_oracle_result.valid &&
            matched_oracle_result.prefers_dynamic)
            ++matched_oracle_dynamic_preferences;
        if (matched_oracle_applied_this_frame)
            ++matched_oracle_applied;
        if (shadow_translation_applied)
            ++shadow_translation_applied_frames;
        if (shadow_translation_factor_injected)
            ++shadow_translation_factor_injections;
        if (velocity_neutralized)
            ++velocity_neutralized_frames;
        if (shadow_translation_stop_reason != "none")
            ++shadow_translation_stopped_frames;
        if (shadow_translation_stop_reason ==
                "propagation_innovation_below_floor" ||
            shadow_translation_stop_reason ==
                "propagation_innovation_exceeded" ||
            shadow_translation_stop_reason ==
                "propagation_innovation_nonfinite")
            ++shadow_translation_propagation_rejections;
        frame_metrics << ni << ',' << tframe << ',' << tracking_state << ','
                      << (pose_valid ? 1 : 0) << ',' << (lost ? 1 : 0) << ','
                      << tmask << ',' << motion_prior_seconds << ',' << ttrack << ','
                      << (tmask + motion_prior_seconds + ttrack) << ',' << dynamic_ratio << ','
                      << motion_prior_result.foreground_fraction << ','
                      << (motion_prior_result.valid ? 1 : 0) << ','
                      << (raw_motion_prior_result.valid ? 1 : 0) << ','
                      << motion_prior_source_frame << ','
                      << shuffle_lag << ','
                      << (mask_cfg.motion_prior_bypass_reliability_gate
                              ? 1 : 0) << ','
                      << (prior_candidate ? 1 : 0) << ','
                      << (prior_consensus_pass ? 1 : 0) << ','
                      << (direct_validation.valid ? 1 : 0) << ','
                      << (prior_direct_pass ? 1 : 0) << ','
                      << (prior_common_support_pass ? 1 : 0) << ','
                      << (prior_would_use ? 1 : 0) << ','
                      << (prior_used ? 1 : 0) << ','
                      << (matched_oracle_result.valid ? 1 : 0) << ','
                      << (matched_oracle_result.prefers_dynamic ? 1 : 0)
                      << ','
                      << (matched_oracle_applied_this_frame ? 1 : 0)
                      << ','
                      << matched_oracle_result.static_translation_error_m
                      << ','
                      << matched_oracle_result.dynamic_translation_error_m
                      << ','
                      << matched_oracle_result.static_rotation_error_rad
                      << ','
                      << matched_oracle_result.dynamic_rotation_error_rad
                      << ','
                      << (shadow_translation_active ? 1 : 0) << ','
                      << (shadow_translation_applied ? 1 : 0) << ','
                      << shadow_translation_remaining << ','
                      << shadow_translation_innovation << ','
                      << shadow_translation_stop_reason << ','
                      << (shadow_translation_factor_injected ? 1 : 0) << ','
                      << (posterior_valid ? 1 : 0) << ','
                      << (posterior_pass ? 1 : 0) << ','
                      << posterior_static_inliers << ','
                      << posterior_dynamic_inliers << ','
                      << posterior_common_support << ','
                      << posterior_static_score << ','
                      << posterior_dynamic_score << ','
                      << (velocity_neutralized ? 1 : 0) << ','
                      << motion_prior_result.information_score << ','
                      << motion_prior_result.objects_observed << ','
                      << motion_prior_result.objects_used << ','
                      << motion_prior_result.feature_tracks << ','
                      << motion_prior_result.inlier_tracks << ','
                      << prior_static_matches << ',' << prior_dynamic_matches << ','
                      << prior_static_inliers << ',' << prior_dynamic_inliers << ','
                      << prior_common_support << ',' << prior_static_common_score << ','
                      << prior_dynamic_common_score << ','
                      << direct_validation.common_support << ','
                      << direct_validation.static_depth_score << ','
                      << direct_validation.dynamic_depth_score << ','
                      << direct_validation.static_photometric_score << ','
                      << direct_validation.dynamic_photometric_score << ','
                      << direct_validation.static_combined_score << ','
                      << direct_validation.dynamic_combined_score << ','
                      << (trajectory_carried_forward ? 1 : 0) << ','
                      << (mask_pose_predicted ? 1 : 0) << ','
                      << (temporal_refinement_applied ? 1 : 0) << ','
                      << recovered_static_pixels << ','
                      << added_dynamic_pixels << ','
                      << (temporal_flow_guard_valid ? 1 : 0) << ','
                      << flow_guard_rejected_pixels << ','
                      << (temporal_recovery_risk_active ? 1 : 0) << ','
                      << recovery_risk_previous_inliers << ','
                      << recovery_risk_hold_for_frame << ','
                      << frame_tracking_recovery_candidate_pixels << ','
                      << frame_tracking_recovery_blocked_pixels << ','
                      << frame_tracking_recovery_audit_pixels << ','
                      << frame_tracking_recovery_mapping_leak_pixels << ','
                      << static_mask_ratio << ','
                      << (mapping_weight_present ? 1 : 0) << ','
                      << mapping_static_ratio << ','
                      << (adaptive_feature_active ? 1 : 0) << ','
                      << adaptive_feature_previous_inliers << ','
                      << adaptive_fast_threshold << ','
                      << extracted_features << '\n';

        // Log progress periodically
        if (ni % 100 == 0) {
            float dynamic_ratio = static_cast<float>(dynamic_pixels) / (staticMask.rows * staticMask.cols) * 100.0f;

            std::cout << "[Frame " << ni << "/" << nImages << "] "
                      << "Track: " << std::fixed << std::setprecision(3) << ttrack * 1000 << " ms, "
                      << "Mask: " << std::fixed << std::setprecision(3) << tmask * 1000 << " ms, "
                      << "Dynamic: " << std::fixed << std::setprecision(1) << dynamic_ratio << "%, "
                      << "State: " << tracking_state
                      << std::endl;
        }

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        const double frame_wall_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - frame_wall_start).count();
        const double configured_period_seconds = frame_period_ms / 1000.0;
        if (configured_period_seconds > 0.0 &&
            frame_wall_seconds < configured_period_seconds) {
            usleep((configured_period_seconds - frame_wall_seconds) * 1e6);
        } else if (realtime_playback && frame_wall_seconds < T) {
            usleep((T - frame_wall_seconds) * 1e6);
        }
    }
    if (instrument_frame >= 0 && !instrumentation_written)
        instrumentation_failed = true;
    frame_metrics.close();
    background_support_metrics.close();
    background_support_pose_metrics.close();
    gaussian_background_pose_metrics.close();
    counterfactual_poses.close();
    all_frame_trajectory.close();

    // Stop all threads
    // --max-frames is a bounded smoke mode, so it must not run the mapper's
    // potentially long tail optimization after the requested input prefix.
    if (pGausMapper && max_frames > 0) pGausMapper->signalStop();
    pSLAM->Shutdown();
    if (training_thd.joinable())
        training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    int rendered_heldout_frames = 0;
    if (pGausMapper && !heldout_frames.empty()) {
        const std::filesystem::path heldout_dir = output_dir / "heldout_novel_view";
        std::filesystem::create_directories(heldout_dir / "ground_truth");
        std::filesystem::create_directories(heldout_dir / "render");
        std::filesystem::create_directories(heldout_dir / "static_mask");
        std::ofstream metadata((heldout_dir / "metadata.csv").string());
        std::ofstream manifest((heldout_dir / "manifest.csv").string());
        metadata << "frame,timestamp,pose_source,is_mapping_keyframe\n";
        manifest << "frame,timestamp,ground_truth,static_mask\n";
        for (const HeldOutFrame& frame : heldout_frames) {
            const std::string stem = cv::format("%06d", frame.frame_id);
            manifest << frame.frame_id << ',' << std::fixed
                     << std::setprecision(9) << frame.timestamp
                     << ",ground_truth/" << stem << ".png"
                     << ",static_mask/" << stem << ".png\n";
            const cv::Mat rendered_float = pGausMapper->renderFromPose(
                frame.Tcw, frame.bgr.cols, frame.bgr.rows, true);
            if (rendered_float.empty()) continue;
            cv::Mat rendered_u8;
            rendered_float.convertTo(rendered_u8, CV_8UC3, 255.0);
            const cv::Mat rendered_bgr =
                DyGeoFusion::rgbToOpenCvBgr(rendered_u8);
            cv::Mat static_mask_u8;
            frame.static_mask.convertTo(static_mask_u8, CV_8UC1, 255.0);
            const bool wrote_ground_truth = cv::imwrite(
                (heldout_dir / "ground_truth" / (stem + ".png")).string(), frame.bgr);
            const bool wrote_render = cv::imwrite(
                (heldout_dir / "render" / (stem + ".png")).string(),
                rendered_bgr);
            const bool wrote_mask = cv::imwrite(
                (heldout_dir / "static_mask" / (stem + ".png")).string(), static_mask_u8);
            if (!wrote_ground_truth || !wrote_render || !wrote_mask) continue;
            metadata << frame.frame_id << ',' << std::fixed << std::setprecision(9)
                     << frame.timestamp << ",online_estimate,0\n";
            ++rendered_heldout_frames;
        }
    }

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Mask time statistics
    saveTrackingTime(vTimesMask, (output_dir / "MaskTime.txt").string());

    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    const double end_to_end_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - run_start).count();
    const auto freeze_state =
        pSLAM->GetDeterministicReplayFreezeState();
    const GaussianMappingAuditCounters mapping_audit =
        pGausMapper
            ? pGausMapper->getMappingAuditCounters()
            : GaussianMappingAuditCounters();
    std::ofstream run_summary((output_dir / "run_summary.json").string());
    run_summary << std::fixed << std::setprecision(9)
                << "{\n"
                << "  \"seed\": " << seed << ",\n"
                << "  \"input_frames\": " << nImages << ",\n"
                << "  \"processed_frames\": " << processed_frames << ",\n"
                << "  \"tracked_frames\": " << tracked_frames << ",\n"
                << "  \"failed_frames\": " << failed_frames << ",\n"
                << "  \"valid_motion_priors\": " << accepted_motion_priors << ",\n"
                << "  \"accepted_motion_priors\": " << accepted_motion_priors << ",\n"
                << "  \"raw_motion_priors\": " << raw_motion_priors << ",\n"
                << "  \"candidate_motion_priors\": " << candidate_motion_priors << ",\n"
                << "  \"consensus_pass_motion_priors\": "
                << consensus_pass_motion_priors << ",\n"
                << "  \"direct_pass_motion_priors\": "
                << direct_pass_motion_priors << ",\n"
                << "  \"common_support_pass_motion_priors\": "
                << common_support_pass_motion_priors << ",\n"
                << "  \"would_use_motion_priors\": " << would_use_motion_priors << ",\n"
                << "  \"used_motion_priors\": " << used_motion_priors << ",\n"
                << "  \"matched_oracle_mode\": \""
                << matched_oracle_mode << "\",\n"
                << "  \"matched_oracle_pose_manifest\": \""
                << jsonEscape(matched_oracle_pose_manifest.string())
                << "\",\n"
                << "  \"matched_oracle_evaluations\": "
                << matched_oracle_evaluations << ",\n"
                << "  \"matched_oracle_dynamic_preferences\": "
                << matched_oracle_dynamic_preferences << ",\n"
                << "  \"matched_oracle_applied\": "
                << matched_oracle_applied << ",\n"
                << "  \"replay_state_capture_enabled\": "
                << (capture_replay_state ? "true" : "false") << ",\n"
                << "  \"shadow_translation_applied_frames\": "
                << shadow_translation_applied_frames << ",\n"
                << "  \"shadow_translation_factor_injections\": "
                << shadow_translation_factor_injections << ",\n"
                << "  \"velocity_neutralized_frames\": "
                << velocity_neutralized_frames << ",\n"
                << "  \"shadow_translation_stopped_frames\": "
                << shadow_translation_stopped_frames << ",\n"
                << "  \"shadow_translation_propagation_rejections\": "
                << shadow_translation_propagation_rejections << ",\n"
                << "  \"motion_gate_min_translation_innovation_m\": "
                << mask_cfg.motion_gate_min_translation_innovation << ",\n"
                << "  \"motion_gate_max_translation_innovation_m\": "
                << mask_cfg.motion_gate_max_translation_innovation << ",\n"
                << "  \"motion_prior_initialization_only\": "
                << (mask_cfg.motion_prior_initialization_only
                        ? "true" : "false") << ",\n"
                << "  \"motion_prior_require_common_support_improvement\": "
                << (mask_cfg.motion_prior_require_common_support_improvement
                        ? "true" : "false") << ",\n"
                << "  \"motion_prior_bypass_reliability_gate\": "
                << (mask_cfg.motion_prior_bypass_reliability_gate
                        ? "true" : "false") << ",\n"
                << "  \"motion_prior_gate_policy\": \""
                << (mask_cfg.motion_prior_gate_policy == 1
                        ? "direct-combined" : "legacy-conjunction")
                << "\",\n"
                << "  \"motion_prior_shuffle_lag_frames\": "
                << mask_cfg.motion_prior_shuffle_lag_frames << ",\n"
                << "  \"motion_prior_shadow_translation_horizon\": "
                << mask_cfg.motion_prior_shadow_translation_horizon << ",\n"
                << "  \"motion_prior_posterior_min_score_improvement\": "
                << mask_cfg.motion_prior_posterior_min_score_improvement
                << ",\n"
                << "  \"motion_prior_shadow_translation_blend\": "
                << mask_cfg.motion_prior_shadow_translation_blend << ",\n"
                << "  \"motion_prior_max_static_information_leverage\": "
                << mask_cfg.motion_prior_max_static_information_leverage
                << ",\n"
                << "  \"motion_prior_static_information_leverage_mode\": \""
                << (mask_cfg.motion_prior_static_information_leverage_mode == 1
                        ? "normalize-to-target" : "cap-only")
                << "\",\n"
                << "  \"counterfactual_pose_pairs\": " << counterfactual_pose_pairs << ",\n"
                << "  \"gaussian_background_shadow_enabled\": "
                << (gaussian_background_shadow ? "true" : "false")
                << ",\n"
                << "  \"gaussian_background_render_frames\": "
                << gaussian_background_render_frames << ",\n"
                << "  \"gaussian_background_candidate_frames\": "
                << gaussian_background_candidate_frames << ",\n"
                << "  \"gaussian_background_gate_pass_frames\": "
                << gaussian_background_gate_pass_frames << ",\n"
                << "  \"gaussian_background_overlap_failures\": "
                << gaussian_background_overlap_failures << ",\n"
                << "  \"carried_forward_frames\": " << carried_forward_frames << ",\n"
                << "  \"mask_pose_predicted_frames\": "
                << mask_pose_predicted_frames << ",\n"
                << "  \"temporal_refinement_frames\": "
                << temporal_refinement_frames << ",\n"
                << "  \"temporal_recovered_static_pixels\": "
                << temporal_recovered_static_pixels << ",\n"
                << "  \"temporal_added_dynamic_pixels\": "
                << temporal_added_dynamic_pixels << ",\n"
                << "  \"temporal_flow_guard_valid_frames\": "
                << temporal_flow_guard_valid_frames << ",\n"
                << "  \"temporal_flow_guard_rejected_pixels\": "
                << temporal_flow_guard_rejected_pixels << ",\n"
                << "  \"temporal_recovery_require_tracking_risk\": "
                << (mask_cfg.temporal_recovery_require_tracking_risk
                        ? "true" : "false") << ",\n"
                << "  \"temporal_recovery_min_previous_inliers\": "
                << mask_cfg.temporal_recovery_min_previous_inliers
                << ",\n"
                << "  \"temporal_recovery_hold_frames\": "
                << mask_cfg.temporal_recovery_hold_frames << ",\n"
                << "  \"temporal_recovery_risk_active_frames\": "
                << temporal_recovery_risk_active_frames << ",\n"
                << "  \"temporal_recovery_risk_blocked_frames\": "
                << temporal_recovery_risk_blocked_frames << ",\n"
                << "  \"tracking_recovery_candidate_pixels\": "
                << tracking_recovery_candidate_pixels << ",\n"
                << "  \"tracking_recovery_blocked_pixels\": "
                << tracking_recovery_blocked_pixels << ",\n"
                << "  \"tracking_recovery_audit_pixels\": "
                << tracking_recovery_audit_pixels << ",\n"
                << "  \"tracking_recovery_mapping_leak_pixels\": "
                << tracking_recovery_mapping_leak_pixels << ",\n"
                << "  \"extracted_features_total\": "
                << extracted_features_total << ",\n"
                << "  \"static_mask_ratio_mean\": "
                << (processed_frames > 0
                        ? static_mask_ratio_sum / processed_frames : 0.0)
                << ",\n"
                << "  \"mapping_weight_frames\": "
                << mapping_weight_frames << ",\n"
                << "  \"mapping_static_ratio_mean\": "
                << (mapping_weight_frames > 0
                        ? mapping_static_ratio_sum /
                            mapping_weight_frames
                        : 1.0)
                << ",\n"
                << "  \"mapping_keyframes_with_static_weight\": "
                << mapping_audit.keyframes_with_static_weight
                << ",\n"
                << "  \"mapping_weight_pixels\": "
                << mapping_audit.mapping_weight_pixels << ",\n"
                << "  \"mapping_static_pixels\": "
                << mapping_audit.mapping_static_pixels << ",\n"
                << "  \"mapping_orb_map_points_considered\": "
                << mapping_audit.orb_map_points_considered << ",\n"
                << "  \"mapping_orb_map_points_rejected_static_mask\": "
                << mapping_audit.orb_map_points_rejected_static_mask
                << ",\n"
                << "  \"mapping_rgbd_densification_candidates\": "
                << mapping_audit.rgbd_densification_candidates
                << ",\n"
                << "  \"mapping_rgbd_densification_rejected_static_mask\": "
                << mapping_audit
                       .rgbd_densification_rejected_static_mask
                << ",\n"
                << "  \"adaptive_feature_active_frames\": "
                << adaptive_feature_active_frames << ",\n"
                << "  \"adaptive_feature_min_previous_inliers\": "
                << mask_cfg.adaptive_feature_min_previous_inliers
                << ",\n"
                << "  \"adaptive_feature_hold_frames\": "
                << mask_cfg.adaptive_feature_hold_frames << ",\n"
                << "  \"adaptive_fast_threshold_mean\": "
                << (processed_frames > 0
                        ? adaptive_fast_threshold_sum / processed_frames : 0.0)
                << ",\n"
                << "  \"extracted_features_mean\": "
                << (processed_frames > 0
                        ? static_cast<double>(extracted_features_total) /
                            processed_frames
                        : 0.0)
                << ",\n"
                << "  \"heldout_candidates\": " << heldout_frames.size() << ",\n"
                << "  \"heldout_frames\": " << rendered_heldout_frames << ",\n"
                << "  \"export_static_masks\": "
                << (export_static_masks ? "true" : "false") << ",\n"
                << "  \"instrument_frame\": "
                << instrument_frame << ",\n"
                << "  \"instrumentation_written\": "
                << (instrumentation_written ? "true" : "false") << ",\n"
                << "  \"failure_rate\": "
                << (processed_frames > 0 ? static_cast<double>(failed_frames) / processed_frames : 1.0) << ",\n"
                << "  \"carried_forward_rate\": "
                << (processed_frames > 0 ? static_cast<double>(carried_forward_frames) / processed_frames : 1.0) << ",\n"
                << "  \"end_to_end_seconds\": " << end_to_end_seconds << ",\n"
                << "  \"realtime_playback\": " << (realtime_playback ? "true" : "false") << ",\n"
                << "  \"synchronize_local_mapping\": "
                << (synchronize_local_mapping ? "true" : "false") << ",\n"
                << "  \"synchronize_loop_closing\": "
                << (synchronize_loop_closing ? "true" : "false") << ",\n"
                << "  \"tracking_only\": "
                << (tracking_only ? "true" : "false") << ",\n"
                << "  \"gaussian_mapper_enabled\": "
                << (pGausMapper ? "true" : "false") << ",\n"
                << "  \"freeze_map_after_frame\": "
                << freeze_map_after_frame << ",\n"
                << "  \"replay_freeze_settling_frames\": "
                << Motion3D::kReplayFreezeSettlingFrames << ",\n"
                << "  \"replay_freeze_complete\": "
                << (freeze_state.complete() ? "true" : "false") << ",\n"
                << "  \"replay_freeze_epoch\": "
                << freeze_state.epoch << ",\n"
                << "  \"frame_period_ms\": " << frame_period_ms << ",\n"
                << "  \"local_mapping_sync_failed\": "
                << (local_mapping_sync_failed ? "true" : "false") << ",\n"
                << "  \"loop_closing_sync_failed\": "
                << (loop_closing_sync_failed ? "true" : "false") << ",\n"
                << "  \"replay_freeze_failed\": "
                << (replay_freeze_failed ? "true" : "false") << ",\n"
                << "  \"instrumentation_failed\": "
                << (instrumentation_failed ? "true" : "false") << "\n"
                << "}\n";

    if (local_mapping_sync_failed || loop_closing_sync_failed ||
        replay_freeze_failed || instrumentation_failed)
        return 2;

    std::cout << std::endl << "DyGeoFusion-SLAM+ finished successfully!" << std::endl;

    return 0;
}

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps)
{
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof())
    {
        std::string s;
        std::getline(fAssociation, s);
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (size_t ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }
    out.close();

    std::cout << "Average time: " << std::fixed << std::setprecision(4)
              << totaltime / nImages * 1000 << " ms" << std::endl;
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}
