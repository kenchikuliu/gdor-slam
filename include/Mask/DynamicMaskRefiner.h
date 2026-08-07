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

#pragma once

#include <memory>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "MaskConfig.h"

namespace DyGeoFusion
{

// Forward declarations
class YoloSegmentor;
class OpticalFlowLK;
class GaussianRenderer;

/**
 * @brief DynamicMaskRefiner - Core module for 3DGS-aware probabilistic dynamic masking
 *
 * This module implements a three-prior fusion approach to generate stable and accurate
 * dynamic masks for Photo-SLAM in dynamic environments:
 *
 * 1. Semantic Prior (M_sem): YOLOv8 segmentation for detecting potentially dynamic objects
 * 2. Motion Prior (M_flow): Lucas-Kanade optical flow for detecting actual motion
 * 3. Geometric Prior (P_geo): 3DGS rendered depth consistency check
 *
 * The three priors are fused to produce a final dynamic probability map P_dyn,
 * which is binarized to generate the staticMask used for tracking and mapping.
 */
class DynamicMaskRefiner
{
public:
    /**
     * @brief Construct a new DynamicMaskRefiner
     * @param cfg Configuration parameters
     * @param device_type Torch device type (CUDA or CPU)
     */
    DynamicMaskRefiner(const MaskConfig& cfg,
                       torch::DeviceType device_type = torch::kCUDA);

    /**
     * @brief Destructor
     */
    ~DynamicMaskRefiner();

    /**
     * @brief Compute static mask for current frame
     *
     * @param rgb Current RGB image (CV_8UC3, BGR or RGB)
     * @param depth Current depth image (CV_32FC1, meters, aligned to RGB)
     * @param Tcw Latest valid world-to-camera pose. During online tracking this
     *            is normally the previous frame pose.
     * @param rendered_depth Optional: 3DGS rendered depth (if nullptr, uses internal renderer)
     * @param timestamp Current frame timestamp (for logging)
     * @return cv::Mat Static mask (CV_8UC1, 1=static, 0=dynamic)
     */
    cv::Mat compute(const cv::Mat& rgb,
                    const cv::Mat& depth,
                    const Sophus::SE3f& Tcw,
                    const cv::Mat* rendered_depth = nullptr,
                    double timestamp = 0.0,
                    bool pose_valid = true);

    /**
     * @brief Get the dynamic probability map from last computation
     * @return cv::Mat P_dyn map (CV_32FC1, values in [0,1])
     */
    cv::Mat getDynamicProbability() const { return P_dyn_.clone(); }

    /**
     * @brief Get semantic mask from last computation
     * @return cv::Mat M_sem (CV_8UC1, 0/1)
     */
    cv::Mat getSemanticMask() const { return M_sem_.clone(); }

    /**
     * @brief Get motion mask from last computation
     * @return cv::Mat M_flow (CV_8UC1, 0/1)
     */
    cv::Mat getMotionMask() const { return M_flow_.clone(); }

    /**
     * @brief Get geometric consistency map from last computation
     * @return cv::Mat P_geo (CV_32FC1, values in [0,1])
     */
    cv::Mat getGeometricConsistency() const { return P_geo_.clone(); }

    cv::Mat getValidGeometricMask() const { return valid_geo_.clone(); }
    cv::Mat getGeometricHardMask() const { return M_geo_dyn_strong_.clone(); }
    cv::Mat getTemporalDynamicMask() const {
        return M_temporal_dynamic_.clone();
    }
    cv::Mat getRawDynamicMask() const {
        return M_raw_dynamic_.clone();
    }
    cv::Mat getTemporalBackgroundDepth() const {
        return temporal_background_depth_.clone();
    }
    bool temporalRefinementApplied() const {
        return temporal_refinement_applied_;
    }
    int temporalRecoveredStaticPixels() const {
        return temporal_recovered_static_pixels_;
    }
    int temporalAddedDynamicPixels() const {
        return temporal_added_dynamic_pixels_;
    }
    bool temporalFlowGuardValid() const {
        return temporal_flow_guard_valid_;
    }
    int temporalFlowGuardRejectedPixels() const {
        return temporal_flow_guard_rejected_pixels_;
    }
    static bool recoveryFlowGuardAllowsPixel(
        const cv::Mat& guard_mask, int x, int y, int safe_radius);
    static bool recoveryFlowGuardAllowsPixel(
        const cv::Mat& guard_mask,
        const cv::Mat& high_confidence_guard_mask,
        int x,
        int y,
        int safe_radius,
        bool adaptive_radius);

    /** Return the mapping confidence implied by P_dyn, or an empty matrix when disabled. */
    cv::Mat getStaticMappingWeight() const;

    /**
     * @brief Get current frame ID
     */
    int getFrameId() const { return frame_id_; }

    /**
     * @brief Reset the refiner state
     */
    void reset();

    /**
     * @brief Update configuration
     * @param cfg New configuration
     */
    void setConfig(const MaskConfig& cfg);

    /**
     * @brief Get current configuration
     */
    const MaskConfig& getConfig() const { return cfg_; }

    /**
     * @brief Set coverage statistics log file path
     * @param log_path Path to log file
     */
    void SetCoverageLogPath(const std::string& log_path);

    /**
     * @brief Set external YOLO segmentor
     * @param segmentor Shared pointer to YoloSegmentor
     */
    void setYoloSegmentor(std::shared_ptr<YoloSegmentor> segmentor);

    /**
     * @brief Convert static mask to torch tensor for 3DGS loss masking
     * @param mask OpenCV mask (CV_8UC1)
     * @return torch::Tensor Mask tensor on configured device
     */
    torch::Tensor maskToTensor(const cv::Mat& mask) const;

private:
    /**
     * @brief Step 1: Compute semantic prior using YOLO
     * @param rgb Input RGB image
     */
    void computeSemanticPrior(const cv::Mat& rgb);

    /**
     * @brief Step 2: Compute motion prior using LK optical flow
     * @param rgb Current RGB image
     * @param gray Current grayscale image
     */
    void computeMotionPrior(const cv::Mat& rgb, const cv::Mat& gray);

    /**
     * @brief Step 3: Compute geometric consistency prior
     * @param depth Current observed depth
     * @param rendered_depth 3DGS rendered depth
     */
    void computeGeometricPrior(const cv::Mat& depth, const cv::Mat& rendered_depth);

    /**
     * @brief Refine the raw semantic/flow mask against a temporal background model
     * @param depth Current observed depth in meters
     * @param rendered_depth Background depth rendered at the predicted current pose
     */
    void refineWithTemporalBackground(const cv::Mat& depth,
                                      const cv::Mat* rendered_depth);

    /**
     * @brief Step 4: Fuse three priors to compute P_dyn
     */
    void fusePriors();

    /**
     * @brief Step 5: Binarize and apply morphological operations
     * @return cv::Mat Final static mask
     */
    cv::Mat binarizeAndRefine();

    /**
     * @brief Save debug images if enabled
     * @param static_mask Final static mask
     */
    void saveDebugImages(const cv::Mat& static_mask);

private:
    MaskConfig cfg_;
    torch::DeviceType device_type_;
    int frame_id_;

    // Previous frame data for optical flow
    cv::Mat prev_gray_;
    cv::Mat prev_depth_;
    std::vector<cv::Point2f> prev_points_;

    // Last semantic mask (for reuse between YOLO intervals)
    cv::Mat last_semantic_mask_;

    // Intermediate results
    cv::Mat M_sem_;    ///< Semantic mask (CV_8UC1, 0/1)
    cv::Mat M_flow_;   ///< Motion mask (CV_8UC1, 0/1)
    cv::Mat M_recovery_flow_guard_; ///< Full-frame residual mask used only to veto temporal recovery
    cv::Mat M_recovery_flow_guard_high_confidence_; ///< High-confidence residual mask used for adaptive neighborhood veto
    cv::Mat P_geo_;    ///< Geometric consistency (CV_32FC1, [0,1])
    cv::Mat P_dyn_;    ///< Dynamic probability (CV_32FC1, [0,1])
    cv::Mat M_geo_dyn_strong_;  ///< Geo hard mask: strong dynamic evidence from geometry (CV_8UC1, 0/1)
    cv::Mat valid_geo_;  ///< Valid geometric pixels mask (CV_8UC1, 1=valid, 0=invalid)
    cv::Mat M_temporal_dynamic_; ///< Prior-image-refined dynamic mask (CV_8UC1, 0/1)
    cv::Mat M_raw_dynamic_; ///< Raw semantic/flow dynamic mask before temporal correction (CV_8UC1, 0/1)
    cv::Mat temporal_background_depth_; ///< Running static background depth model (CV_32FC1)
    bool temporal_refinement_applied_ = false;
    int temporal_recovered_static_pixels_ = 0;
    int temporal_added_dynamic_pixels_ = 0;
    bool temporal_flow_guard_valid_ = false;
    int temporal_flow_guard_rejected_pixels_ = 0;

    // Morphological kernel
    cv::Mat morph_kernel_;

    // YOLO segmentor
    std::shared_ptr<YoloSegmentor> yolo_segmentor_;

    // Optical flow calculator
    std::shared_ptr<OpticalFlowLK> optical_flow_;

    // Adaptive G-MR state variables
    float global_geo_;           ///< Global geometric consistency (mean of P_geo)
    float geo_scale_;            ///< Adaptive scaling factor for w_geo [0, 1]
    int no_semantic_dyn_frames_; ///< Count of frames with no YOLO dynamic detection
    bool disable_geo_;           ///< Flag to completely disable geometric prior

    // Viewpoint gating for large camera motion
    Sophus::SE3f prev_Tcw_;      ///< Previous valid world-to-camera pose
    bool has_prev_pose_;         ///< Flag indicating if previous pose is available
    float viewpoint_scale_;      ///< Scale factor based on viewpoint change [0, 1]

    // Coverage statistics logging
    std::ofstream coverage_log_; ///< Log file for dynamic coverage statistics
    double current_timestamp_;   ///< Current frame timestamp for logging
};

} // namespace DyGeoFusion
