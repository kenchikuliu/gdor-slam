/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

namespace {
struct BatchPose {
    int frame_id = -1;
    double timestamp = 0.0;
    std::vector<float> pose_twcs; // tx ty tz qx qy qz qw (Twc)
};

struct RenderOptions {
    bool render_once = false;
    bool render_batch = false;
    std::vector<float> pose_twcs; // tx ty tz qx qy qz qw (Twc)
    std::string output_path;
    std::string pose_file;
    std::string output_dir;
    int width = 0;
    int height = 0;
};

bool parseArgs(int argc, char** argv, RenderOptions& opts, std::string& error)
{
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pose" && i + 7 < argc) {
            opts.pose_twcs.clear();
            for (int k = 0; k < 7; ++k) {
                opts.pose_twcs.push_back(std::stof(argv[i + 1 + k]));
            }
            opts.render_once = true;
            i += 7;
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_path = argv[++i];
            opts.render_once = true;
        } else if (arg == "--pose-file" && i + 1 < argc) {
            opts.pose_file = argv[++i];
            opts.render_batch = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            opts.output_dir = argv[++i];
            opts.render_batch = true;
        } else if (arg == "--width" && i + 1 < argc) {
            opts.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            opts.height = std::stoi(argv[++i]);
        } else {
            error = "Unknown or incomplete argument: " + arg;
            return false;
        }
    }
    return true;
}

std::vector<BatchPose> loadBatchPoses(const std::string& pose_file)
{
    std::ifstream input(pose_file);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open pose file: " + pose_file);
    }

    std::vector<BatchPose> poses;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        BatchPose pose;
        pose.pose_twcs.resize(7);
        if (!(stream >> pose.frame_id >> pose.timestamp
              >> pose.pose_twcs[0] >> pose.pose_twcs[1] >> pose.pose_twcs[2]
              >> pose.pose_twcs[3] >> pose.pose_twcs[4] >> pose.pose_twcs[5]
              >> pose.pose_twcs[6])) {
            if (line_number == 1 && line.find("frame") != std::string::npos) {
                continue;
            }
            throw std::runtime_error(
                "Invalid pose row at line " + std::to_string(line_number));
        }
        if (pose.frame_id < 0) {
            throw std::runtime_error(
                "Frame id must be non-negative at line " +
                std::to_string(line_number));
        }
        poses.push_back(std::move(pose));
    }
    if (poses.empty()) {
        throw std::runtime_error("Pose file contains no valid rows: " + pose_file);
    }
    return poses;
}

Sophus::SE3f poseTwcToTcw(const std::vector<float>& pose_twcs)
{
    // pose_twcs: tx ty tz qx qy qz qw (Twc)
    const float tx = pose_twcs[0];
    const float ty = pose_twcs[1];
    const float tz = pose_twcs[2];
    const float qx = pose_twcs[3];
    const float qy = pose_twcs[4];
    const float qz = pose_twcs[5];
    const float qw = pose_twcs[6];

    Eigen::Quaternionf q(qw, qx, qy, qz);
    if (q.norm() <= 1e-6f) {
        throw std::invalid_argument("Pose quaternion must be non-zero");
    }
    q.normalize();
    Sophus::SE3f Twc(q, Eigen::Vector3f(tx, ty, tz));
    return Twc.inverse();
}

bool saveRenderedImage(const cv::Mat& img_rgb_f32, const std::string& output_path)
{
    if (img_rgb_f32.empty()) {
        return false;
    }

    cv::Mat img_bgr;
    cv::cvtColor(img_rgb_f32, img_bgr, cv::COLOR_RGB2BGR);
    cv::Mat img_u8;
    img_bgr.convertTo(img_u8, CV_8UC3, 255.0f);
    return cv::imwrite(output_path, img_u8);
}

bool hasGaussianPlyProperties(
    const std::filesystem::path& ply_path,
    std::string& error)
{
    std::ifstream input(ply_path, std::ios::binary);
    if (!input.is_open()) {
        error = "Cannot open PLY file: " + ply_path.string();
        return false;
    }
    std::string header;
    std::string line;
    while (std::getline(input, line)) {
        header += line;
        header.push_back('\n');
        if (line == "end_header") {
            break;
        }
        if (header.size() > 1024 * 1024) {
            error = "PLY header exceeds 1 MiB";
            return false;
        }
    }
    if (header.find("end_header") == std::string::npos) {
        error = "PLY file has no complete header";
        return false;
    }
    const std::vector<std::string> required = {
        "property float f_dc_0",
        "property float opacity",
        "property float scale_0",
        "property float rot_0",
    };
    for (const std::string& property : required) {
        if (header.find(property) == std::string::npos) {
            error = (
                "PLY is not a trained Gaussian map; missing header entry: " +
                property);
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_camera_parameters"            /*2*/
                  << " path_to_result_ply_file"              /*3*/
                  << " [--pose tx ty tz qx qy qz qw --output output.png --width W --height H]"
                  << " [--pose-file poses.csv --output-dir renders --width W --height H]"
                  << std::endl;
        return 1;
    }

    RenderOptions opts;
    std::string parse_error;
    try {
        if (!parseArgs(argc, argv, opts, parse_error)) {
            std::cerr << "Error: " << parse_error << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: invalid render argument: " << e.what() << std::endl;
        return 1;
    }

    if (opts.render_once && opts.render_batch) {
        std::cerr << "Error: single-pose and batch rendering options cannot be combined."
                  << std::endl;
        return 1;
    }
    if (opts.render_once && (opts.pose_twcs.size() != 7 || opts.output_path.empty())) {
        std::cerr << "Error: --pose and --output must be provided for one-shot rendering." << std::endl;
        return 1;
    }
    if (opts.render_batch && (opts.pose_file.empty() || opts.output_dir.empty())) {
        std::cerr << "Error: --pose-file and --output-dir must be provided for batch rendering."
                  << std::endl;
        return 1;
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

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::filesystem::path camera_path(argv[2]);
    std::filesystem::path result_ply_path(argv[3]);
    std::string ply_error;
    if (!hasGaussianPlyProperties(result_ply_path, ply_error)) {
        std::cerr << "Error: " << ply_error << std::endl;
        return 1;
    }
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            nullptr, gaussian_cfg_path, std::filesystem::path(), 0, device_type);
    try {
        pGausMapper->loadPly(result_ply_path, camera_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to load Gaussian map: " << e.what()
                  << std::endl;
        return 1;
    }

    if (opts.render_once || opts.render_batch) {
        int width = opts.width;
        int height = opts.height;
        if (width <= 0 || height <= 0) {
            if (pGausMapper->scene_->cameras_.empty()) {
                std::cerr << "Error: loaded scene does not contain camera dimensions." << std::endl;
                return 1;
            }
            auto cam_it = pGausMapper->scene_->cameras_.begin();
            width = cam_it->second.width_;
            height = cam_it->second.height_;
        }

        if (opts.render_batch) {
            std::vector<BatchPose> poses;
            try {
                poses = loadBatchPoses(opts.pose_file);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }

            const std::filesystem::path output_dir(opts.output_dir);
            if (std::filesystem::exists(output_dir)) {
                std::cerr << "Error: refusing to reuse batch output directory: "
                          << output_dir << std::endl;
                return 1;
            }
            std::error_code output_error;
            std::filesystem::create_directories(output_dir, output_error);
            if (output_error) {
                std::cerr << "Error: cannot create output directory: "
                          << output_error.message() << std::endl;
                return 1;
            }
            std::ofstream metadata(output_dir / "metadata.csv");
            if (!metadata.is_open()) {
                std::cerr << "Error: cannot write batch render metadata." << std::endl;
                return 1;
            }
            metadata << "frame,timestamp,render\n";
            for (const BatchPose& pose : poses) {
                Sophus::SE3f Tcw;
                try {
                    Tcw = poseTwcToTcw(pose.pose_twcs);
                } catch (const std::exception& e) {
                    std::cerr << "Error: invalid pose for frame " << pose.frame_id
                              << ": " << e.what() << std::endl;
                    return 1;
                }
                std::ostringstream stem;
                stem << std::setfill('0') << std::setw(6) << pose.frame_id;
                const std::filesystem::path output_path =
                    output_dir / (stem.str() + ".png");
                cv::Mat rendered = pGausMapper->renderFromPose(
                    Tcw, width, height, true);
                if (!saveRenderedImage(rendered, output_path.string())) {
                    std::cerr << "Error: failed to write rendered image "
                              << output_path << std::endl;
                    return 1;
                }
                metadata << pose.frame_id << ',' << std::fixed
                         << std::setprecision(9) << pose.timestamp << ','
                         << output_path.filename().string() << '\n';
            }
            std::cout << "[view_result] Saved " << poses.size()
                      << " batch renders to " << output_dir << std::endl;
            return 0;
        }

        std::filesystem::path output_path(opts.output_path);
        if (!output_path.parent_path().empty()) {
            std::error_code output_error;
            std::filesystem::create_directories(output_path.parent_path(), output_error);
            if (output_error) {
                std::cerr << "Error: cannot create output directory: "
                          << output_error.message() << std::endl;
                return 1;
            }
        }

        Sophus::SE3f Tcw;
        try {
            Tcw = poseTwcToTcw(opts.pose_twcs);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        cv::Mat rendered = pGausMapper->renderFromPose(Tcw, width, height, true);
        if (!saveRenderedImage(rendered, opts.output_path)) {
            std::cerr << "Error: failed to write rendered image to " << opts.output_path << std::endl;
            return 1;
        }
        std::cout << "[view_result] Saved novel view to " << opts.output_path << std::endl;
        return 0;
    }

    // Interactive viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    pViewer = std::make_shared<ImGuiViewer>(nullptr, pGausMapper, false);
    pViewer->run();

    return 0;
}
