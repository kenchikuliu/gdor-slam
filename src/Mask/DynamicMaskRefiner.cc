/**
 * This file is part of DyGeoFusion-SLAM+
 *
 * Copyright (C) 2024 DyGeoFusion-SLAM+ Authors.
 *
 * DyGeoFusion-SLAM+ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "include/Mask/DynamicMaskRefiner.h"
#include "include/Mask/YoloSegmentor.h"
#include "include/Mask/OpticalFlowLK.h"
#include "include/tensor_utils.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <stdexcept>

namespace DyGeoFusion
{

DynamicMaskRefiner::DynamicMaskRefiner(const MaskConfig& cfg,
                                       torch::DeviceType device_type)
    : cfg_(cfg),
      device_type_(device_type),
      frame_id_(0),
      global_geo_(0.5f),
      geo_scale_(1.0f),
      no_semantic_dyn_frames_(0),
      disable_geo_(false),
      has_prev_pose_(false),
      viewpoint_scale_(1.0f)
{
    // Initialize morphological kernel
    if (cfg_.morph_kernel > 1) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(cfg_.morph_kernel, cfg_.morph_kernel));
    }

    // Initialize YOLO segmentor if enabled (TensorRT backend)
    if (cfg_.use_yolo && !cfg_.use_external_mask) {
        if (cfg_.yolo_model_path.empty()) {
            throw std::invalid_argument(
                "YOLO is enabled but mask.yolo_model_path is empty");
        }
        yolo_segmentor_ = std::make_shared<YoloSegmentor>(
            cfg_.yolo_model_path,
            cfg_.yolo_conf_threshold,
            cfg_.yolo_nms_threshold);
        if (!yolo_segmentor_->isReady()) {
            throw std::runtime_error(
                "YOLO TensorRT segmentor failed to initialize from " +
                cfg_.yolo_model_path);
        }
        std::cout
            << "[DynamicMaskRefiner] YOLO TensorRT segmentor initialized successfully"
            << std::endl;
    }

    // Initialize optical flow calculator
    optical_flow_ = std::make_shared<OpticalFlowLK>(
        cfg_.lk_win_size,
        cfg_.lk_max_level,
        cfg_.lk_grid_step,
        cfg_.tau_flow);

    std::cout << "[DynamicMaskRefiner] Initialized with config:" << std::endl;
    cfg_.print();
}

DynamicMaskRefiner::~DynamicMaskRefiner()
{
    // Cleanup handled by smart pointers
}

bool DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
    const cv::Mat& guard_mask, int x, int y, int safe_radius)
{
    return recoveryFlowGuardAllowsPixel(
        guard_mask, guard_mask, x, y, safe_radius, false);
}

bool DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
    const cv::Mat& guard_mask,
    const cv::Mat& high_confidence_guard_mask,
    int x,
    int y,
    int safe_radius,
    bool adaptive_radius)
{
    if (guard_mask.empty() || guard_mask.type() != CV_8UC1 ||
        x < 0 || y < 0 || x >= guard_mask.cols || y >= guard_mask.rows) {
        return false;
    }

    const int radius = std::max(0, safe_radius);
    if (adaptive_radius) {
        if (guard_mask.at<uchar>(y, x) != 0) {
            return false;
        }
        if (radius == 0) {
            return true;
        }
        if (high_confidence_guard_mask.empty() ||
            high_confidence_guard_mask.type() != CV_8UC1 ||
            high_confidence_guard_mask.size() != guard_mask.size()) {
            return false;
        }
    }
    const cv::Mat& neighborhood_guard =
        adaptive_radius ? high_confidence_guard_mask : guard_mask;
    const int y_begin = std::max(0, y - radius);
    const int y_end = std::min(guard_mask.rows - 1, y + radius);
    const int x_begin = std::max(0, x - radius);
    const int x_end = std::min(guard_mask.cols - 1, x + radius);
    for (int ny = y_begin; ny <= y_end; ++ny) {
        const uchar* guard_row = neighborhood_guard.ptr<uchar>(ny);
        for (int nx = x_begin; nx <= x_end; ++nx) {
            if (guard_row[nx] != 0) {
                return false;
            }
        }
    }
    return true;
}

cv::Mat DynamicMaskRefiner::compute(const cv::Mat& rgb,
                                     const cv::Mat& depth,
                                     const Sophus::SE3f& Tcw,
                                     const cv::Mat* rendered_depth,
                                     double timestamp,
                                     bool pose_valid)
{
    auto start_time = std::chrono::steady_clock::now();
    current_timestamp_ = timestamp;

    // Convert to grayscale for optical flow
    cv::Mat gray;
    if (rgb.channels() == 3) {
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = rgb.clone();
    }

    // Initialize output masks to image size
    M_sem_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
    M_flow_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
    M_recovery_flow_guard_ =
        cv::Mat::zeros(rgb.size(), CV_8UC1);
    M_recovery_flow_guard_high_confidence_ =
        cv::Mat::zeros(rgb.size(), CV_8UC1);
    P_geo_ = cv::Mat(rgb.size(), CV_32FC1, cv::Scalar(0.5f));
    P_dyn_ = cv::Mat::zeros(rgb.size(), CV_32FC1);
    valid_geo_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
    M_temporal_dynamic_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
    M_raw_dynamic_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
    temporal_refinement_applied_ = false;
    temporal_recovered_static_pixels_ = 0;
    temporal_added_dynamic_pixels_ = 0;
    temporal_flow_guard_valid_ = false;
    temporal_flow_guard_rejected_pixels_ = 0;

    // Step 1: Semantic prior (YOLO or external precomputed mask)
    if (cfg_.use_yolo || cfg_.use_external_mask) {
        computeSemanticPrior(rgb);
    }

    // Step 2: Motion prior (LK optical flow)
    if (cfg_.use_flow && !prev_gray_.empty()) {
        computeMotionPrior(rgb, gray);
    }

    // The conservative mapping route must be available even when temporal
    // refinement is disabled. Keep this raw semantic-plus-flow mask separate
    // from the optional temporal tracking correction.
    M_raw_dynamic_ = M_sem_.clone();
    if (cfg_.use_flow && !M_flow_.empty()) {
        cv::bitwise_or(M_raw_dynamic_, M_flow_, M_raw_dynamic_);
    }
    M_temporal_dynamic_ = M_raw_dynamic_.clone();

    if (cfg_.use_temporal_background_refinement) {
        refineWithTemporalBackground(depth, rendered_depth);
    }

    // Viewpoint gating: compute camera motion to determine if G-MR is reliable
    // When camera rotation/translation is large, rendered depth is unreliable
    viewpoint_scale_ = 1.0f;  // Default: full G-MR
    if (has_prev_pose_ && pose_valid && cfg_.use_depth_consistency) {
        // Compute relative pose from previous to current frame
        Sophus::SE3f T_rel = Tcw * prev_Tcw_.inverse();

        // Extract rotation angle (in degrees)
        // Using Frobenius norm of log map for SO(3)
        Eigen::AngleAxisf aa(T_rel.rotationMatrix());
        float rot_deg = std::abs(aa.angle()) * 180.0f / M_PI;

        // Extract translation magnitude (in meters)
        float trans_m = T_rel.translation().norm();

        // Smooth gating using smoothstep-like function
        // Convert rotation threshold from radians to degrees for comparison
        float rot_thresh_deg = cfg_.viewpoint_gating_rot_thresh * 180.0f / M_PI;
        // rot_scale: 0 when rot_deg >= threshold, 1 when rot_deg = 0
        float rot_scale = std::max(0.0f, 1.0f - rot_deg / rot_thresh_deg);
        // trans_scale: 0 when trans_m >= threshold, 1 when trans_m = 0
        float trans_scale = std::max(0.0f, 1.0f - trans_m / cfg_.viewpoint_gating_trans_thresh);

        // Combined viewpoint scale (use minimum of both)
        viewpoint_scale_ = std::min(rot_scale, trans_scale);

        // Debug: log viewpoint gating periodically
        if (frame_id_ < 5 || frame_id_ % 100 == 0 || viewpoint_scale_ < 0.5f) {
            std::cout << "[DynamicMaskRefiner] VIEWPOINT GATING: rot=" << rot_deg
                      << "deg trans=" << trans_m << "m"
                      << " → rot_scale=" << rot_scale << " trans_scale=" << trans_scale
                      << " viewpoint_scale=" << viewpoint_scale_ << std::endl;
        }
    }

    // Step 3: Geometric prior (3DGS depth consistency)
    if (cfg_.use_depth_consistency &&
        rendered_depth != nullptr &&
        !rendered_depth->empty() &&
        frame_id_ > cfg_.warmup_frames) {
        computeGeometricPrior(depth, *rendered_depth);
    }

    // Step 4: Fuse priors
    fusePriors();

    // Step 5: Binarize and morphological refinement
    cv::Mat static_mask = binarizeAndRefine();

    // Save debug images if enabled
    if (cfg_.save_debug_images) {
        saveDebugImages(static_mask);
    }

    // Update state for next frame
    prev_gray_ = gray.clone();
    prev_depth_ = depth.clone();
    if (pose_valid) {
        prev_Tcw_ = Tcw;
        has_prev_pose_ = true;
    }
    frame_id_++;

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (frame_id_ % 100 == 0) {
        std::cout << "[DynamicMaskRefiner] Frame " << frame_id_
                  << " processed in " << duration.count() << " ms" << std::endl;
    }

    // Log dynamic coverage statistics for paper Figure 5
    if (coverage_log_.is_open() && !static_mask.empty()) {
        const int total_pixels = static_mask.rows * static_mask.cols;
        const int dynamic_pixels = total_pixels - cv::countNonZero(static_mask);
        float coverage_percent = static_cast<float>(dynamic_pixels) / total_pixels * 100.0f;

        coverage_log_ << std::fixed << std::setprecision(6)
                      << current_timestamp_ << " "
                      << std::setprecision(2)
                      << coverage_percent << " "
                      << dynamic_pixels << " "
                      << total_pixels << "\n";
    }

    return static_mask;
}

void DynamicMaskRefiner::computeSemanticPrior(const cv::Mat& rgb)
{
    // External precomputed mask mode (replaces YOLO)
    if (cfg_.use_external_mask && !cfg_.external_mask_dir.empty()) {
        auto try_load = [&](int width) -> cv::Mat {
            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(width) << frame_id_;
            return cv::imread((std::filesystem::path(cfg_.external_mask_dir) / (oss.str() + ".png")).string(),
                              cv::IMREAD_GRAYSCALE);
        };
        cv::Mat raw = try_load(8);
        if (raw.empty()) raw = try_load(6);
        if (raw.empty()) raw = try_load(0);  // plain number
        if (!raw.empty()) {
            if (raw.size() != rgb.size()) {
                cv::resize(raw, raw, rgb.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            cv::Mat bin;
            cv::threshold(raw, bin, 127, 1, cv::THRESH_BINARY);
            bin.convertTo(last_semantic_mask_, CV_8UC1);
        } else {
            if (frame_id_ < 5 || frame_id_ % 100 == 0) {
                std::cerr << "[DynamicMaskRefiner] External mask missing for frame "
                          << frame_id_ << " in " << cfg_.external_mask_dir << std::endl;
            }
            last_semantic_mask_ = cv::Mat::zeros(rgb.size(), CV_8UC1);
        }
        if (!last_semantic_mask_.empty())
            M_sem_ = last_semantic_mask_.clone();
        if (frame_id_ < cfg_.global_geo_warmup_frames && cfg_.use_depth_consistency) {
            if (cv::countNonZero(M_sem_) == 0) no_semantic_dyn_frames_++;
            if (frame_id_ == cfg_.global_geo_warmup_frames - 1) {
                if (no_semantic_dyn_frames_ >= cfg_.global_geo_no_semantic_thresh) {
                    disable_geo_ = true;
                    std::cout << "[DynamicMaskRefiner] ADAPTIVE G-MR: Static scene detected, disabling geo prior." << std::endl;
                }
            }
        }
        return;
    }

    // Only run YOLO at specified intervals
    if (frame_id_ % cfg_.yolo_every_n_frames == 0) {
        if (yolo_segmentor_ && yolo_segmentor_->isReady()) {
            last_semantic_mask_ = yolo_segmentor_->getDynamicMask(rgb, false);
            // Debug: check semantic mask
            if (frame_id_ < 5 || frame_id_ % 100 == 0) {
                int dynamic_count = cv::countNonZero(last_semantic_mask_);
                std::cout << "[DynamicMaskRefiner] Frame " << frame_id_
                          << ": M_sem has " << dynamic_count << " dynamic pixels" << std::endl;
            }
        } else {
            throw std::runtime_error(
                "YOLO semantic prior is enabled but the segmentor is not ready");
        }
    }

    // Reuse last semantic mask
    if (!last_semantic_mask_.empty()) {
        M_sem_ = last_semantic_mask_.clone();
    }

    // Adaptive G-MR Step 3: Track no-semantic-dynamic frames during warmup
    // If YOLO consistently detects no dynamic objects, this is likely a static scene
    // In static scenes, G-MR should be disabled to avoid false positives
    if (frame_id_ < cfg_.global_geo_warmup_frames && cfg_.use_depth_consistency) {
        int sem_count = cv::countNonZero(M_sem_);
        if (sem_count == 0) {
            no_semantic_dyn_frames_++;
        }

        // Check at the end of warmup period
        if (frame_id_ == cfg_.global_geo_warmup_frames - 1) {
            if (no_semantic_dyn_frames_ >= cfg_.global_geo_no_semantic_thresh) {
                disable_geo_ = true;
                std::cout << "[DynamicMaskRefiner] ADAPTIVE G-MR: Detected static scene ("
                          << no_semantic_dyn_frames_ << "/" << cfg_.global_geo_warmup_frames
                          << " frames with no YOLO detection). Disabling geometric prior." << std::endl;
            } else {
                std::cout << "[DynamicMaskRefiner] ADAPTIVE G-MR: Detected dynamic scene ("
                          << (cfg_.global_geo_warmup_frames - no_semantic_dyn_frames_) << "/" << cfg_.global_geo_warmup_frames
                          << " frames with YOLO detection). Geometric prior enabled." << std::endl;
            }
        }
    }
}

void DynamicMaskRefiner::computeMotionPrior(const cv::Mat& rgb, const cv::Mat& gray)
{
    if (optical_flow_) {
        cv::Mat semantic_static_roi = 1 - M_sem_;
        if (cfg_.temporal_recovery_flow_guard) {
            M_recovery_flow_guard_ =
                optical_flow_->getFullFrameResidualMotionMask(
                    prev_gray_, gray, semantic_static_roi,
                    cfg_.temporal_flow_guard_high_confidence_scale);
            M_recovery_flow_guard_high_confidence_ =
                optical_flow_->lastHighConfidenceResidualMotionMask().clone();
            temporal_flow_guard_valid_ =
                optical_flow_->lastResidualModelValid();
            M_flow_ =
                optical_flow_->lastStaticResidualMotionMask().clone();
        } else {
            M_flow_ = optical_flow_->getResidualMotionMask(
                prev_gray_, gray, semantic_static_roi);
        }
    }
}

void DynamicMaskRefiner::refineWithTemporalBackground(
    const cv::Mat& depth, const cv::Mat* rendered_depth)
{
    M_temporal_dynamic_ = M_raw_dynamic_.clone();

    if (depth.empty() || depth.type() != CV_32FC1) {
        return;
    }

    if (temporal_background_depth_.size() != depth.size() ||
        temporal_background_depth_.type() != CV_32FC1) {
        temporal_background_depth_ =
            cv::Mat::zeros(depth.size(), CV_32FC1);
    }

    const bool rendered_valid =
        rendered_depth != nullptr &&
        !rendered_depth->empty() &&
        rendered_depth->type() == CV_32FC1 &&
        rendered_depth->size() == depth.size();
    const float render_weight =
        std::clamp(cfg_.temporal_render_weight, 0.0f, 1.0f);
    const float history_weight = 1.0f - render_weight;
    cv::Mat prior_background = cv::Mat::zeros(depth.size(), CV_32FC1);
    cv::Mat valid_render_mask = cv::Mat::zeros(depth.size(), CV_8UC1);
    int valid_prior_pixels = 0;
    int valid_render_pixels = 0;

    for (int y = 0; y < depth.rows; ++y) {
        const float* history_row =
            temporal_background_depth_.ptr<float>(y);
        const float* render_row = rendered_valid
            ? rendered_depth->ptr<float>(y) : nullptr;
        float* prior_row = prior_background.ptr<float>(y);
        uchar* render_valid_row = valid_render_mask.ptr<uchar>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float history = history_row[x];
            const float rendered = render_row ? render_row[x] : 0.0f;
            const bool has_history =
                history > 0.0f && std::isfinite(history);
            const bool has_render =
                rendered > 0.0f && std::isfinite(rendered);
            if (has_render) {
                render_valid_row[x] = 1;
                ++valid_render_pixels;
            }
            float weighted_depth = 0.0f;
            float total_weight = 0.0f;
            if (has_history && history_weight > 0.0f) {
                weighted_depth += history_weight * history;
                total_weight += history_weight;
            }
            if (has_render && render_weight > 0.0f) {
                weighted_depth += render_weight * rendered;
                total_weight += render_weight;
            }
            if (total_weight > 0.0f) {
                prior_row[x] = weighted_depth / total_weight;
                ++valid_prior_pixels;
            }
        }
    }

    const int total_pixels = depth.rows * depth.cols;
    const float valid_fraction = total_pixels > 0
        ? static_cast<float>(valid_prior_pixels) /
            static_cast<float>(total_pixels)
        : 0.0f;
    const float render_valid_fraction = total_pixels > 0
        ? static_cast<float>(valid_render_pixels) /
            static_cast<float>(total_pixels)
        : 0.0f;
    const bool can_refine =
        rendered_valid &&
        frame_id_ >= cfg_.temporal_background_warmup_frames &&
        render_valid_fraction >= cfg_.temporal_min_valid_fraction;

    if (can_refine) {
        const int radius = cfg_.temporal_neighbor_radius;
        const int required = cfg_.temporal_min_consistent_neighbors;
        const float threshold = cfg_.temporal_depth_threshold;
        cv::Mat refined = M_temporal_dynamic_.clone();

        for (int y = 0; y < depth.rows; ++y) {
            const float* prior_row = prior_background.ptr<float>(y);
            const uchar* raw_row = M_temporal_dynamic_.ptr<uchar>(y);
            const uchar* render_valid_row =
                valid_render_mask.ptr<uchar>(y);
            uchar* refined_row = refined.ptr<uchar>(y);
            for (int x = 0; x < depth.cols; ++x) {
                const float background = prior_row[x];
                if (render_valid_row[x] == 0 ||
                    !(background > 0.0f) ||
                    !std::isfinite(background)) {
                    continue;
                }

                int valid_neighbors = 0;
                int consistent_neighbors = 0;
                const int y_begin = std::max(0, y - radius);
                const int y_end = std::min(depth.rows - 1, y + radius);
                const int x_begin = std::max(0, x - radius);
                const int x_end = std::min(depth.cols - 1, x + radius);
                for (int ny = y_begin; ny <= y_end; ++ny) {
                    const float* depth_row = depth.ptr<float>(ny);
                    for (int nx = x_begin; nx <= x_end; ++nx) {
                        const float observed = depth_row[nx];
                        if (!(observed > 0.0f) ||
                            !std::isfinite(observed)) {
                            continue;
                        }
                        ++valid_neighbors;
                        if (std::abs(observed - background) <
                            threshold) {
                            ++consistent_neighbors;
                        }
                    }
                }
                if (valid_neighbors < required) {
                    continue;
                }

                const uchar refined_value =
                    consistent_neighbors >= required ? 0 : 1;
                if (cfg_.temporal_recovery_only &&
                    raw_row[x] == 0 && refined_value != 0) {
                    continue;
                }
                if (raw_row[x] != 0 && refined_value == 0 &&
                    cfg_.temporal_recovery_flow_guard &&
                    (!temporal_flow_guard_valid_ ||
                     !recoveryFlowGuardAllowsPixel(
                         M_recovery_flow_guard_,
                         M_recovery_flow_guard_high_confidence_,
                         x, y,
                         cfg_.temporal_flow_guard_safe_radius,
                         cfg_.temporal_flow_guard_adaptive_radius))) {
                    ++temporal_flow_guard_rejected_pixels_;
                    continue;
                }
                refined_row[x] = refined_value;
                if (raw_row[x] != 0 && refined_value == 0) {
                    ++temporal_recovered_static_pixels_;
                } else if (raw_row[x] == 0 &&
                           refined_value != 0) {
                    ++temporal_added_dynamic_pixels_;
                }
            }
        }
        M_temporal_dynamic_ = refined;
        temporal_refinement_applied_ = true;
    }

    const float observation_weight =
        std::clamp(cfg_.temporal_observation_weight, 0.0f, 1.0f);
    cv::Mat updated_background = prior_background.clone();
    for (int y = 0; y < depth.rows; ++y) {
        const float* observed_row = depth.ptr<float>(y);
        const uchar* dynamic_row = M_temporal_dynamic_.ptr<uchar>(y);
        float* updated_row = updated_background.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float observed = observed_row[x];
            if (dynamic_row[x] != 0 ||
                !(observed > 0.0f) ||
                !std::isfinite(observed)) {
                continue;
            }
            const float prior = updated_row[x];
            if (prior > 0.0f && std::isfinite(prior)) {
                updated_row[x] =
                    (1.0f - observation_weight) * prior +
                    observation_weight * observed;
            } else {
                updated_row[x] = observed;
            }
        }
    }
    temporal_background_depth_ = updated_background;

    if (frame_id_ < 5 || frame_id_ % 100 == 0 ||
        temporal_recovered_static_pixels_ > 0 ||
        temporal_added_dynamic_pixels_ > 0) {
        std::cout
            << "[DynamicMaskRefiner] TEMPORAL BACKGROUND: applied="
            << (temporal_refinement_applied_ ? "true" : "false")
            << " prior_valid=" << std::fixed << std::setprecision(3)
            << valid_fraction
            << " render_valid=" << render_valid_fraction
            << " recovered_static="
            << temporal_recovered_static_pixels_
            << " added_dynamic="
            << temporal_added_dynamic_pixels_
            << " flow_guard_valid="
            << (temporal_flow_guard_valid_ ? "true" : "false")
            << " flow_guard_rejected="
            << temporal_flow_guard_rejected_pixels_
            << std::endl;
    }
}

void DynamicMaskRefiner::computeGeometricPrior(const cv::Mat& depth,
                                                const cv::Mat& rendered_depth)
{
    // Initialize valid_geo_ mask (tracks which pixels have valid geometric info)
    valid_geo_ = cv::Mat::zeros(depth.size(), CV_8UC1);

    // Validate inputs
    if (depth.empty() || rendered_depth.empty()) {
        // Default P_geo = 1.0 (favor static) when no geometric info available
        P_geo_.setTo(1.0f);
        return;
    }

    if (depth.size() != rendered_depth.size()) {
        std::cerr << "[DynamicMaskRefiner] Depth size mismatch!" << std::endl;
        P_geo_.setTo(1.0f);
        return;
    }

    // Conservative two-stage geometric prior:
    // - Only penalize "clearly inconsistent" depths (diff > tau_lo)
    // - Small differences (< tau_lo) are considered normal sensor noise
    // This prevents false positives from sparse point-projection depth
    const float tau_lo = cfg_.tau_depth;         // ~5cm: normal noise threshold
    const float tau_hi = cfg_.tau_depth * 3.0f;  // ~15cm: clearly dynamic threshold

    // Track valid pixel count and P_geo sum for computing global_geo only on valid pixels
    int valid_count = 0;
    float p_geo_sum = 0.0f;

    // Diagnostic: depth error analysis (every 100 frames)
    std::vector<int> error_bins(6, 0);  // [0-3cm), [3-6cm), [6-9cm), [9-15cm), [15-30cm), [30cm+]
    std::vector<int> pgeo_bins(5, 0);   // [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    std::vector<float> error_values;
    error_values.reserve(depth.rows * depth.cols);

    // Compute P_geo with two-stage approach:
    // P_geo close to 1 means static consistent
    // P_geo close to 0 means dynamic inconsistent
    // DEFAULT: P_geo = 1.0 (favor static when no geometric info)
    for (int y = 0; y < depth.rows; ++y) {
        const float* d_curr_row = depth.ptr<float>(y);
        const float* d_render_row = rendered_depth.ptr<float>(y);
        float* p_geo_row = P_geo_.ptr<float>(y);
        uchar* valid_row = valid_geo_.ptr<uchar>(y);

        for (int x = 0; x < depth.cols; ++x) {
            float dc = d_curr_row[x];
            float dr = d_render_row[x];

            // Handle invalid depth values
            // Changed: default P_geo = 1.0 (favor static) instead of 0.5 (neutral)
            if (dc <= 0.0f || dr <= 0.0f || std::isnan(dc) || std::isnan(dr)) {
                p_geo_row[x] = 1.0f;  // Favor static for missing data
                valid_row[x] = 0;     // Mark as invalid (no geometric info)
            } else {
                // Both dc and dr are valid - this pixel has geometric info
                valid_row[x] = 1;
                valid_count++;

                float diff = std::abs(dc - dr);

                // Two-stage conservative approach:
                // - diff <= tau_lo: completely ignore (P_geo = 1.0, no penalty)
                // - diff >= tau_hi: full penalty (P_geo = 0.0, dynamic)
                // - in between: linear interpolation
                float geo_dyn;  // 0 = static, 1 = dynamic
                if (diff <= tau_lo) {
                    geo_dyn = 0.0f;  // No penalty for small differences
                } else if (diff >= tau_hi) {
                    geo_dyn = 1.0f;  // Full penalty for large differences
                } else {
                    geo_dyn = (diff - tau_lo) / (tau_hi - tau_lo);  // Linear interp
                }

                // P_geo = 1 - geo_dyn (high P_geo = static)
                float p_geo = 1.0f - geo_dyn;
                p_geo_row[x] = p_geo;
                p_geo_sum += p_geo;

                // Collect diagnostic data (only for valid pixels)
                error_values.push_back(diff);

                // Update error histogram
                if (diff < 0.03f) error_bins[0]++;
                else if (diff < 0.06f) error_bins[1]++;
                else if (diff < 0.09f) error_bins[2]++;
                else if (diff < 0.15f) error_bins[3]++;
                else if (diff < 0.30f) error_bins[4]++;
                else error_bins[5]++;

                // Update P_geo histogram
                int pgeo_idx = std::min(4, static_cast<int>(p_geo * 5.0f));
                pgeo_bins[pgeo_idx]++;
            }
        }
    }

    // Adaptive G-MR Step 1: Compute global geometric consistency
    // CRITICAL FIX: Only compute mean over VALID pixels (where both dc>0 and dr>0)
    // This prevents sparse rendered depth from diluting global_geo to 0.5
    if (valid_count > 0) {
        global_geo_ = p_geo_sum / static_cast<float>(valid_count);
    } else {
        global_geo_ = 1.0f;  // No valid pixels = assume static (conservative)
    }

    // Diagnostic output (every 100 frames)
    if ((frame_id_ < 5 || frame_id_ % 100 == 0) && !error_values.empty()) {
        // Compute statistics
        float mean_error = std::accumulate(error_values.begin(), error_values.end(), 0.0f) / error_values.size();
        std::sort(error_values.begin(), error_values.end());
        float median_error = error_values[error_values.size() / 2];
        float min_error = error_values.front();
        float max_error = error_values.back();

        float var_sum = 0.0f;
        for (float err : error_values) {
            float dev = err - mean_error;
            var_sum += dev * dev;
        }
        float std_error = std::sqrt(var_sum / error_values.size());

        std::cout << "[DynamicMaskRefiner] DEPTH ERROR ANALYSIS (frame " << frame_id_ << "):" << std::endl;
        std::cout << "  Error histogram: [0-3cm)=" << error_bins[0]
                  << " [3-6cm)=" << error_bins[1] << " [6-9cm)=" << error_bins[2]
                  << " [9-15cm)=" << error_bins[3] << " [15-30cm)=" << error_bins[4]
                  << " [30cm+)=" << error_bins[5] << std::endl;
        std::cout << "  Error stats: mean=" << (mean_error*100) << "cm median=" << (median_error*100)
                  << "cm std=" << (std_error*100) << "cm min=" << (min_error*100)
                  << "cm max=" << (max_error*100) << "cm" << std::endl;
        std::cout << "  P_geo histogram: [0.0-0.2)=" << pgeo_bins[0] << " [0.2-0.4)=" << pgeo_bins[1]
                  << " [0.4-0.6)=" << pgeo_bins[2] << " [0.6-0.8)=" << pgeo_bins[3]
                  << " [0.8-1.0]=" << pgeo_bins[4] << std::endl;
    }

    // Adaptive G-MR Step 2: Compute adaptive scaling factor for w_geo
    // geo_scale = clamp((global_geo - 0.4) / 0.4, 0, 1)
    // When global_geo < 0.4: geo_scale = 0 (completely disable geometric prior)
    // When global_geo > 0.8: geo_scale = 1 (full geometric prior)
    // In between: linear interpolation
    geo_scale_ = std::max(0.0f, std::min(1.0f, (global_geo_ - 0.4f) / 0.4f));

    // Debug: log adaptive G-MR state periodically
    if (frame_id_ < 5 || frame_id_ % 100 == 0) {
        int total_pixels = depth.rows * depth.cols;
        float valid_ratio = static_cast<float>(valid_count) / static_cast<float>(total_pixels) * 100.0f;
        std::cout << "[DynamicMaskRefiner] ADAPTIVE G-MR: global_geo=" << global_geo_
                  << " geo_scale=" << geo_scale_
                  << " w_geo_effective=" << (cfg_.w_geo * geo_scale_)
                  << " valid_geo_pixels=" << valid_count << " (" << valid_ratio << "%)"
                  << " disable_geo=" << (disable_geo_ ? "true" : "false") << std::endl;
    }
}

void DynamicMaskRefiner::fusePriors()
{
    // Debug: check M_sem before fusion
    int sem_count = cv::countNonZero(M_sem_);
    int flow_count = cv::countNonZero(M_flow_);

    // Debug: check a few pixels from M_sem_
    static bool first_debug = true;
    if (first_debug && sem_count > 0) {
        // Find first non-zero pixel in M_sem_
        for (int y = 0; y < M_sem_.rows && first_debug; ++y) {
            const uchar* row = M_sem_.ptr<uchar>(y);
            for (int x = 0; x < M_sem_.cols && first_debug; ++x) {
                if (row[x] > 0) {
                    std::cout << "[DEBUG] First non-zero M_sem pixel at (" << x << "," << y
                              << ") value=" << (int)row[x] << std::endl;
                    first_debug = false;
                }
            }
        }
    }

    const float semantic_reliability = cfg_.use_yolo || cfg_.use_external_mask
        ? std::clamp(cfg_.w_sem, 0.0f, 1.0f) : 0.0f;
    const float flow_reliability = cfg_.use_flow
        ? std::clamp(cfg_.w_flow, 0.0f, 1.0f) : 0.0f;
    const float geo_reliability = disable_geo_ || !cfg_.use_depth_consistency
        ? 0.0f : (geo_scale_ * viewpoint_scale_);

    for (int y = 0; y < P_dyn_.rows; ++y) {
        const uchar* m_sem_row = M_sem_.ptr<uchar>(y);
        const uchar* m_flow_row = M_flow_.ptr<uchar>(y);
        const float* p_geo_row = P_geo_.ptr<float>(y);
        const uchar* valid_geo_row = valid_geo_.ptr<uchar>(y);
        float* p_dyn_row = P_dyn_.ptr<float>(y);

        for (int x = 0; x < P_dyn_.cols; ++x) {
            const float p_sem = m_sem_row[x] > 0 ? semantic_reliability : 0.0f;
            const float p_flow = m_flow_row[x] > 0 ? flow_reliability : 0.0f;
            float p_geo_dyn = 0.0f;
            if (valid_geo_row[x] != 0 && geo_reliability > 0.0f) {
                const float geo_dyn = std::clamp(1.0f - p_geo_row[x], 0.0f, 1.0f);
                p_geo_dyn = geo_reliability * (1.0f - std::exp(-std::max(0.0f, cfg_.w_geo) * geo_dyn));
            }

            // Noisy-OR keeps P_dyn=0 when there is no evidence and preserves
            // a hard semantic decision when w_sem=1.
            p_dyn_row[x] = std::clamp(
                1.0f - (1.0f - p_sem) * (1.0f - p_flow) * (1.0f - p_geo_dyn),
                0.0f, 1.0f);
        }
    }

    // Debug: check P_dyn statistics
    if (frame_id_ < 5 || frame_id_ % 100 == 0) {
        double minVal, maxVal;
        cv::minMaxLoc(P_dyn_, &minVal, &maxVal);
        int count_above_thresh = 0;
        for (int y = 0; y < P_dyn_.rows; ++y) {
            const float* row = P_dyn_.ptr<float>(y);
            for (int x = 0; x < P_dyn_.cols; ++x) {
                if (row[x] >= cfg_.dyn_threshold) count_above_thresh++;
            }
        }
        std::cout << "[DynamicMaskRefiner] Frame " << frame_id_
                  << ": P_dyn min=" << minVal << " max=" << maxVal
                  << " count>=" << cfg_.dyn_threshold << ": " << count_above_thresh
                  << " (sem=" << sem_count << ", flow=" << flow_count << ")" << std::endl;
    }
}

cv::Mat DynamicMaskRefiner::binarizeAndRefine()
{
    // The temporal path is explicitly allowed to correct semantic false
    // positives and false negatives. Other configurations retain the raw
    // semantic decision exactly.
    cv::Mat M_sem_bin = temporal_refinement_applied_
        ? M_temporal_dynamic_.clone()
        : M_sem_.clone();

    // Step 2: Binarize P_dyn from soft fusion to get soft dynamic mask
    cv::Mat M_dyn_soft(P_dyn_.size(), CV_8UC1);
    for (int y = 0; y < P_dyn_.rows; ++y) {
        const float* p_dyn_row = P_dyn_.ptr<float>(y);
        uchar* m_dyn_row = M_dyn_soft.ptr<uchar>(y);
        for (int x = 0; x < P_dyn_.cols; ++x) {
            m_dyn_row[x] = (p_dyn_row[x] >= cfg_.dyn_threshold) ? 1 : 0;
        }
    }

    // Step 3: Initialize geo hard mask
    M_geo_dyn_strong_ = cv::Mat::zeros(P_geo_.size(), CV_8UC1);

    // Step 4: Geo Hard Mask Logic
    // Key principle: Geo can only ADD dynamic regions (补充 YOLO 漏掉的动态)
    // Conditions for geo hard mask:
    //   1. cfg_.enable_geo_hard is true
    //   2. global_geo_ > global_geo_min (map is reliable enough)
    //   3. viewpoint_scale_ > vp_scale_min (camera motion is small enough)
    //   4. Not in warmup period (frame_id_ > warmup_frames)
    //   5. disable_geo_ is false (not a pure static scene)
    bool allow_geo_hard = cfg_.enable_geo_hard &&
                          !disable_geo_ &&
                          (global_geo_ > cfg_.global_geo_min) &&
                          (viewpoint_scale_ > cfg_.vp_scale_min) &&
                          (frame_id_ > cfg_.warmup_frames);

    int geo_hard_count = 0;
    if (allow_geo_hard) {
        // Compute geo_dyn_strong: regions where YOLO missed but geo is confident
        // CRITICAL: Only consider pixels where valid_geo_ == 1 (have geometric info)
        for (int y = 0; y < P_geo_.rows; ++y) {
            const uchar* m_sem_row = M_sem_bin.ptr<uchar>(y);
            const float* p_geo_row = P_geo_.ptr<float>(y);
            const uchar* valid_row = valid_geo_.empty() ? nullptr : valid_geo_.ptr<uchar>(y);
            uchar* geo_hard_row = M_geo_dyn_strong_.ptr<uchar>(y);

            for (int x = 0; x < P_geo_.cols; ++x) {
                // RED LINE 1: If YOLO already marked as dynamic, skip (don't touch)
                if (m_sem_row[x] != 0) {
                    continue;
                }

                // NEW: Only consider pixels with valid geometric info
                // If valid_geo_ is empty (no geometric prior computed), skip all
                if (valid_row == nullptr || valid_row[x] == 0) {
                    continue;
                }

                // geo_dyn = 1 - P_geo: low P_geo = high dynamic probability
                float geo_dyn = 1.0f - p_geo_row[x];

                // If geo evidence is strong enough, mark as dynamic
                if (geo_dyn > cfg_.geo_dyn_thresh) {
                    geo_hard_row[x] = 1;
                    geo_hard_count++;
                }
            }
        }
    }

    // Step 5: Combine explicitly enabled hard-mask sources.
    // This ensures:
    //   - YOLO dynamic regions are always included (RED LINE 1: YOLO says dynamic = always dynamic)
    //   - Geo can only ADD to dynamic regions, never remove (RED LINE 2)
    cv::Mat M_dyn_final;
    M_dyn_final = M_sem_bin.clone();
    if (!temporal_refinement_applied_ &&
        cfg_.enable_flow_hard && cfg_.use_flow) {
        cv::bitwise_or(M_dyn_final, M_flow_, M_dyn_final);
    }
    cv::bitwise_or(M_dyn_final, M_geo_dyn_strong_, M_dyn_final);

    // Debug: check mask statistics
    if (frame_id_ < 5 || frame_id_ % 100 == 0 || geo_hard_count > 0) {
        int sem_count = cv::countNonZero(M_sem_bin);
        int soft_count = cv::countNonZero(M_dyn_soft);
        int final_count = cv::countNonZero(M_dyn_final);
        std::cout << "[DynamicMaskRefiner] Frame " << frame_id_
                  << ": M_sem=" << sem_count
                  << " M_dyn_soft=" << soft_count
                  << " geo_hard=" << geo_hard_count
                  << " M_dyn_final=" << final_count
                  << " allow_geo_hard=" << (allow_geo_hard ? "true" : "false")
                  << " (global_geo=" << global_geo_
                  << " vp_scale=" << viewpoint_scale_ << ")" << std::endl;
    }

    // Apply morphological closing to smooth edges
    if (!morph_kernel_.empty()) {
        cv::morphologyEx(M_dyn_final, M_dyn_final, cv::MORPH_CLOSE, morph_kernel_);
    }

    // Invert to get static mask (1 = static, 0 = dynamic)
    cv::Mat static_mask = 1 - M_dyn_final;

    return static_mask;
}

cv::Mat DynamicMaskRefiner::getStaticMappingWeight() const
{
    if (!cfg_.use_soft_mapping || P_dyn_.empty()) {
        return cv::Mat();
    }

    cv::Mat weight(P_dyn_.size(), CV_32FC1);
    for (int y = 0; y < P_dyn_.rows; ++y) {
        const float* probability = P_dyn_.ptr<float>(y);
        float* confidence = weight.ptr<float>(y);
        for (int x = 0; x < P_dyn_.cols; ++x) {
            confidence[x] = std::max(
                cfg_.soft_mapping_min_weight,
                1.0f - cfg_.soft_mapping_alpha * std::clamp(probability[x], 0.0f, 1.0f));
        }
    }
    return weight;
}

cv::Mat DynamicMaskRefiner::getRawStaticMask() const
{
    if (M_raw_dynamic_.empty()) {
        return cv::Mat();
    }

    cv::Mat raw_dynamic = M_raw_dynamic_.clone();
    if (!morph_kernel_.empty()) {
        cv::morphologyEx(
            raw_dynamic, raw_dynamic, cv::MORPH_CLOSE, morph_kernel_);
    }
    return 1 - raw_dynamic;
}

void DynamicMaskRefiner::saveDebugImages(const cv::Mat& static_mask)
{
    if (cfg_.debug_output_dir.empty()) return;

    std::filesystem::path output_dir(cfg_.debug_output_dir);
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    std::string frame_str = std::to_string(frame_id_);

    // Save semantic mask
    cv::Mat M_sem_vis = M_sem_ * 255;
    cv::imwrite((output_dir / ("sem_" + frame_str + ".png")).string(), M_sem_vis);

    // Save motion mask
    cv::Mat M_flow_vis = M_flow_ * 255;
    cv::imwrite((output_dir / ("flow_" + frame_str + ".png")).string(), M_flow_vis);

    // Save geometric consistency (normalized to 0-255)
    cv::Mat P_geo_vis;
    P_geo_.convertTo(P_geo_vis, CV_8UC1, 255.0);
    cv::imwrite((output_dir / ("geo_" + frame_str + ".png")).string(), P_geo_vis);

    // Save dynamic probability
    cv::Mat P_dyn_vis;
    P_dyn_.convertTo(P_dyn_vis, CV_8UC1, 255.0);
    cv::imwrite((output_dir / ("pdyn_" + frame_str + ".png")).string(), P_dyn_vis);

    // Save geo hard mask (geo_dyn_strong)
    if (!M_geo_dyn_strong_.empty()) {
        cv::Mat geo_hard_vis = M_geo_dyn_strong_ * 255;
        cv::imwrite((output_dir / ("geo_hard_" + frame_str + ".png")).string(), geo_hard_vis);
    }

    // Save final static mask
    cv::Mat static_vis = static_mask * 255;
    cv::imwrite((output_dir / ("static_" + frame_str + ".png")).string(), static_vis);
}

void DynamicMaskRefiner::reset()
{
    frame_id_ = 0;
    prev_gray_.release();
    prev_depth_.release();
    prev_points_.clear();
    last_semantic_mask_.release();
    M_sem_.release();
    M_flow_.release();
    M_recovery_flow_guard_.release();
    M_recovery_flow_guard_high_confidence_.release();
    P_geo_.release();
    P_dyn_.release();
    M_geo_dyn_strong_.release();
    valid_geo_.release();
    M_temporal_dynamic_.release();
    M_raw_dynamic_.release();
    temporal_background_depth_.release();
    temporal_refinement_applied_ = false;
    temporal_recovered_static_pixels_ = 0;
    temporal_added_dynamic_pixels_ = 0;
    temporal_flow_guard_valid_ = false;
    temporal_flow_guard_rejected_pixels_ = 0;

    // Reset adaptive G-MR state
    global_geo_ = 0.5f;
    geo_scale_ = 1.0f;
    no_semantic_dyn_frames_ = 0;
    disable_geo_ = false;

    // Reset viewpoint gating state
    has_prev_pose_ = false;
    viewpoint_scale_ = 1.0f;
}

void DynamicMaskRefiner::SetCoverageLogPath(const std::string& log_path)
{
    if (coverage_log_.is_open())
        coverage_log_.close();

    coverage_log_.open(log_path);
    if (coverage_log_.is_open()) {
        coverage_log_ << std::fixed;
        coverage_log_ << "# timestamp coverage_percent dynamic_pixels total_pixels\n";
    } else {
        std::cerr << "[DynamicMaskRefiner] Cannot open coverage log: "
                  << log_path << std::endl;
    }
}

void DynamicMaskRefiner::setConfig(const MaskConfig& cfg)
{
    cfg_ = cfg;

    // Update morphological kernel
    if (cfg_.morph_kernel > 1) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(cfg_.morph_kernel, cfg_.morph_kernel));
    } else {
        morph_kernel_.release();
    }

    // Update optical flow threshold
    if (optical_flow_) {
        optical_flow_->setFlowThreshold(cfg_.tau_flow);
    }
}

void DynamicMaskRefiner::setYoloSegmentor(std::shared_ptr<YoloSegmentor> segmentor)
{
    yolo_segmentor_ = segmentor;
}

torch::Tensor DynamicMaskRefiner::maskToTensor(const cv::Mat& mask) const
{
    // Convert CV_8UC1 mask to torch tensor
    cv::Mat mask_float;
    mask.convertTo(mask_float, CV_32FC1);

    // Use tensor_utils from Photo-SLAM
    return tensor_utils::cvMat2TorchTensor_Float32(mask_float, device_type_);
}

} // namespace DyGeoFusion
