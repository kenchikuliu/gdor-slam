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

#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace DyGeoFusion
{

/**
 * @brief Configuration for the DynamicMaskRefiner module
 *
 * This structure holds all configurable parameters for the three-prior
 * dynamic mask fusion system: semantic (YOLO), motion (LK flow), and
 * geometric (3DGS depth consistency).
 */
struct MaskConfig
{
    // ============= Feature Toggles =============
    bool use_yolo              = true;   ///< Enable YOLO semantic segmentation
    bool use_flow              = false;  ///< Enable ego-motion-compensated LK residual flow
    bool use_depth_consistency = false;  ///< Enable 3DGS depth consistency check
    bool use_soft_mapping      = false;  ///< Pass continuous confidence weights to Gaussian mapping
    bool use_hard_mapping_mask = false;  ///< Pass the final binary static mask to Gaussian mapping
    bool enable_flow_hard      = false;  ///< Allow flow to remove tracking features
    bool use_temporal_background_refinement = false; ///< Refine the raw mask with a rendered temporal depth model
    bool use_adaptive_feature_extraction = false; ///< Replenish static ORB features after masking
    bool temporal_recovery_only = false; ///< Temporal evidence may restore tracking support but not add exclusions
    bool temporal_conservative_mapping = false; ///< Keep raw semantic/flow exclusions in the Gaussian mapper
    bool temporal_recovery_flow_guard = false; ///< Require a valid full-frame residual-flow model and no independent motion before temporal recovery
    bool temporal_recovery_require_tracking_risk = false; ///< Apply temporal recovery only while the previous tracking support is weak
    int temporal_recovery_min_previous_inliers = 0; ///< Previous-frame inlier threshold for the independent recovery-risk gate; 0 disables it
    int temporal_recovery_hold_frames = 1; ///< Minimum consecutive recovery-risk frames after a low-inlier trigger
    int temporal_flow_guard_safe_radius = 0; ///< Reject recovery if residual motion appears within this pixel radius
    bool temporal_flow_guard_adaptive_radius = false; ///< Apply the safety radius only to high-confidence residual motion
    float temporal_flow_guard_high_confidence_scale = 2.0f; ///< Residual threshold multiplier for adaptive neighborhood veto

    // ============= Timing Parameters =============
    int warmup_frames          = 50;     ///< Frames before enabling depth consistency
    int yolo_every_n_frames    = 1;      ///< YOLO inference interval (1 = every frame)
    int temporal_background_warmup_frames = 5; ///< Frames before applying temporal refinement

    // ============= Threshold Parameters =============
    float tau_flow             = 1.5f;   ///< Optical flow magnitude threshold (pixels)
    float tau_depth            = 0.05f;  ///< Depth difference threshold (meters) - 5cm for geometric consistency

    // ============= Prior-image background refinement =============
    float temporal_render_weight = 0.6f; ///< Weight of current predicted-pose rendered depth
    float temporal_observation_weight = 0.2f; ///< Weight of accepted current static depth
    float temporal_depth_threshold = 0.10f; ///< Neighbor depth agreement threshold (meters)
    int temporal_neighbor_radius = 2; ///< Radius of the observed-depth neighborhood
    int temporal_min_consistent_neighbors = 5; ///< Required agreeing neighbors
    float temporal_min_valid_fraction = 0.02f; ///< Minimum usable rendered/background coverage
    float temporal_min_render_opacity = 0.5f; ///< Minimum accumulated opacity for trusted rendered depth
    float temporal_prediction_max_translation = 0.5f; ///< Maximum constant-velocity increment (meters)
    float temporal_prediction_max_rotation = 0.7854f; ///< Maximum constant-velocity increment (radians)

    // ============= Mask-aware adaptive ORB extraction =============
    float adaptive_feature_relaxation = 0.5f; ///< FAST threshold reduction at a fully dynamic mask
    int adaptive_feature_min_previous_inliers = 0; ///< Enable adaptive extraction only after a weaker previous frame; 0 disables the risk gate
    int adaptive_feature_hold_frames = 1; ///< Minimum consecutive adaptive frames after a low-inlier trigger

    // ============= Prior Weights =============
    float w_sem                = 1.0f;   ///< Semantic evidence reliability
    float w_flow               = 1.0f;   ///< Residual-flow evidence reliability
    float w_geo                = 2.0f;   ///< Geometric evidence response scale

    // ============= Output Parameters =============
    float dyn_threshold        = 0.6f;   ///< P_dyn threshold for diagnostic soft fusion
    int morph_kernel           = 3;      ///< Morphological kernel size
    float soft_mapping_alpha   = 0.7f;   ///< Strength of P_dyn in mapping confidence
    float soft_mapping_min_weight = 0.3f; ///< Lower bound for mapping confidence

    // ============= Motion-marginalized pose prior =============
    bool use_motion_pose_prior = false;  ///< Reuse informative dynamic features only as a pose prior
    bool motion_pose_prior_shadow_only = false; ///< Evaluate the prior gate without applying it
    int motion_max_features = 400;
    int motion_min_features_per_object = 20;
    int motion_min_component_area = 400;
    int motion_min_track_age = 3;
    float motion_ransac_threshold = 0.05f;
    float motion_min_inlier_ratio = 0.5f;
    float motion_min_information = 5.0f;
    float motion_max_translation = 0.5f;
    float motion_max_rotation = 0.5236f;
    bool motion_use_depth_foreground_filter = true;
    float motion_foreground_depth_separation = 0.15f;
    float motion_foreground_min_fraction = 0.08f;
    float motion_min_object_translation_speed = 0.03f;
    float motion_min_object_rotation_speed = 0.05f;
    float motion_measurement_translation_sigma = 0.01f;
    float motion_measurement_rotation_sigma = 0.02f;
    float motion_camera_translation_sigma = 0.02f;
    float motion_camera_rotation_sigma = 0.02f;
    float motion_velocity_process_translation_sigma = 0.20f;
    float motion_velocity_process_rotation_sigma = 0.35f;
    float motion_max_information_eigenvalue = 1e4f;
    float motion_candidate_mahalanobis_threshold = 16.812f;
    int motion_gate_min_inlier_gain = 5;
    float motion_gate_min_inlier_gain_ratio = 0.05f;
    int motion_gate_max_static_inliers = 2147483647;
    float motion_gate_min_translation_innovation = 0.0f;
    float motion_gate_max_translation_innovation = 1e6f;
    float motion_gate_max_rotation_innovation = 3.1415927f;
    float motion_prior_information_scale = 1.0f;
    bool motion_prior_use_direct_validation = false;
    int motion_prior_direct_score_mode = 0; ///< 0=photometric, 1=depth, 2=combined
    bool motion_prior_inject_local_map = true;
    bool motion_prior_initialization_only = false;
    bool motion_prior_require_common_support_improvement = false;
    bool motion_prior_bypass_reliability_gate = false;
    int motion_prior_gate_policy = 0; ///< 0=legacy-conjunction, 1=direct-combined
    int motion_prior_shuffle_lag_frames = 0;
    int motion_prior_shadow_translation_horizon = 0;
    float motion_prior_posterior_min_score_improvement = 0.0f;
    float motion_prior_shadow_translation_blend = 1.0f;
    float motion_prior_max_static_information_leverage = 0.0f;
    int motion_prior_static_information_leverage_mode = 0; ///< 0=cap-only, 1=normalize-to-target

    // ============= Geo Hard Mask Parameters =============
    // These control when geometric evidence can add to the dynamic mask
    // Key principle: YOLO decisions are never overridden, geo can only ADD dynamic regions
    bool enable_geo_hard       = false;  ///< Enable geometric hard mask (补充 YOLO 漏掉的动态)
    float geo_dyn_thresh       = 0.6f;   ///< Geo dynamic threshold: (1-P_geo) > this = strong dynamic evidence
    float global_geo_min       = 0.6f;   ///< Minimum global_geo required to enable geo hard mask
    float vp_scale_min         = 0.5f;   ///< Minimum viewpoint_scale required to enable geo hard mask

    // ============= Adaptive Mechanism Parameters =============
    // Viewpoint Gating: Disable geometric prior during large camera motion
    float viewpoint_gating_rot_thresh    = 0.1745f;  ///< Rotation threshold (radians, ~10 degrees)
    float viewpoint_gating_trans_thresh  = 0.08f;    ///< Translation threshold (meters)

    // Global Geo Scaling: Disable geometric prior when no semantic dynamics detected
    int global_geo_warmup_frames         = 50;       ///< Frames to check for no-semantic scene
    int global_geo_no_semantic_thresh    = 45;       ///< Threshold to disable geo if no YOLO detections

    // ============= External Precomputed Mask =============
    bool use_external_mask        = false; ///< Load precomputed semantic mask from directory (replaces YOLO)
    std::string external_mask_dir = "";   ///< Dir with PNG masks named %08d.png (255=dynamic, 0=static)

    // ============= YOLO Model Parameters =============
    std::string yolo_model_path = "";    ///< Path to YOLO model file
    float yolo_conf_threshold  = 0.25f;  ///< YOLO confidence threshold
    float yolo_nms_threshold   = 0.45f;  ///< YOLO NMS threshold
    int yolo_input_width       = 640;    ///< YOLO input width
    int yolo_input_height      = 640;    ///< YOLO input height

    // ============= LK Flow Parameters =============
    int lk_win_size            = 21;     ///< LK window size
    int lk_max_level           = 3;      ///< LK pyramid max level
    int lk_grid_step           = 8;      ///< Grid step for sparse LK sampling

    // ============= Debug/Visualization =============
    bool save_debug_images     = false;  ///< Save intermediate mask images
    std::string debug_output_dir = "";   ///< Directory for debug outputs

    /**
     * @brief Load configuration from YAML file
     * @param yaml_path Path to the configuration YAML file
     * @return true if loaded successfully
     */
    bool loadFromYAML(const std::string& yaml_path);

    /**
     * @brief Save configuration to YAML file
     * @param yaml_path Path to save the configuration
     * @return true if saved successfully
     */
    bool saveToYAML(const std::string& yaml_path) const;

    /**
     * @brief Print current configuration to stdout
     */
    void print() const;
};

} // namespace DyGeoFusion
