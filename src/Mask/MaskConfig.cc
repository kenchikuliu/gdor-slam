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

#include "include/Mask/MaskConfig.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace DyGeoFusion
{

namespace
{
bool readStringNode(const cv::FileStorage& fs,
                    const char* key,
                    std::string& value)
{
    const cv::FileNode node = fs[key];
    if (node.empty()) {
        return false;
    }
    value = node.string();
    return true;
}
} // namespace

bool MaskConfig::loadFromYAML(const std::string& yaml_path)
{
    try {
        cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[MaskConfig] Cannot open config file: " << yaml_path << std::endl;
            return false;
        }

        // Feature toggles
        if (!fs["mask.use_yolo"].empty())
            fs["mask.use_yolo"] >> use_yolo;
        if (!fs["mask.use_flow"].empty())
            fs["mask.use_flow"] >> use_flow;
        if (!fs["mask.use_depth_consistency"].empty())
            fs["mask.use_depth_consistency"] >> use_depth_consistency;
        if (!fs["mask.use_soft_mapping"].empty())
            fs["mask.use_soft_mapping"] >> use_soft_mapping;
        if (!fs["mask.use_hard_mapping_mask"].empty())
            fs["mask.use_hard_mapping_mask"] >>
                use_hard_mapping_mask;
        if (!fs["mask.enable_flow_hard"].empty())
            fs["mask.enable_flow_hard"] >> enable_flow_hard;
        if (!fs["mask.use_temporal_background_refinement"].empty())
            fs["mask.use_temporal_background_refinement"] >>
                use_temporal_background_refinement;
        if (!fs["mask.use_adaptive_feature_extraction"].empty())
            fs["mask.use_adaptive_feature_extraction"] >>
                use_adaptive_feature_extraction;
        if (!fs["mask.temporal_recovery_only"].empty())
            fs["mask.temporal_recovery_only"] >>
                temporal_recovery_only;
        if (!fs["mask.temporal_conservative_mapping"].empty())
            fs["mask.temporal_conservative_mapping"] >>
                temporal_conservative_mapping;
        if (!fs["mask.temporal_recovery_flow_guard"].empty())
            fs["mask.temporal_recovery_flow_guard"] >>
                temporal_recovery_flow_guard;
        if (!fs["mask.temporal_flow_guard_safe_radius"].empty())
            fs["mask.temporal_flow_guard_safe_radius"] >>
                temporal_flow_guard_safe_radius;
        if (!fs["mask.temporal_flow_guard_adaptive_radius"].empty())
            fs["mask.temporal_flow_guard_adaptive_radius"] >>
                temporal_flow_guard_adaptive_radius;
        if (!fs["mask.temporal_flow_guard_high_confidence_scale"].empty())
            fs["mask.temporal_flow_guard_high_confidence_scale"] >>
                temporal_flow_guard_high_confidence_scale;

        // Timing parameters
        if (!fs["mask.warmup_frames"].empty())
            fs["mask.warmup_frames"] >> warmup_frames;
        if (!fs["mask.yolo_every_n_frames"].empty())
            fs["mask.yolo_every_n_frames"] >> yolo_every_n_frames;
        if (!fs["mask.temporal_background_warmup_frames"].empty())
            fs["mask.temporal_background_warmup_frames"] >>
                temporal_background_warmup_frames;

        // Threshold parameters
        if (!fs["mask.tau_flow"].empty())
            fs["mask.tau_flow"] >> tau_flow;
        if (!fs["mask.tau_depth"].empty())
            fs["mask.tau_depth"] >> tau_depth;

        if (!fs["mask.temporal_render_weight"].empty())
            fs["mask.temporal_render_weight"] >> temporal_render_weight;
        if (!fs["mask.temporal_observation_weight"].empty())
            fs["mask.temporal_observation_weight"] >>
                temporal_observation_weight;
        if (!fs["mask.temporal_depth_threshold"].empty())
            fs["mask.temporal_depth_threshold"] >>
                temporal_depth_threshold;
        if (!fs["mask.temporal_neighbor_radius"].empty())
            fs["mask.temporal_neighbor_radius"] >>
                temporal_neighbor_radius;
        if (!fs["mask.temporal_min_consistent_neighbors"].empty())
            fs["mask.temporal_min_consistent_neighbors"] >>
                temporal_min_consistent_neighbors;
        if (!fs["mask.temporal_min_valid_fraction"].empty())
            fs["mask.temporal_min_valid_fraction"] >>
                temporal_min_valid_fraction;
        if (!fs["mask.temporal_min_render_opacity"].empty())
            fs["mask.temporal_min_render_opacity"] >>
                temporal_min_render_opacity;
        if (!fs["mask.temporal_prediction_max_translation"].empty())
            fs["mask.temporal_prediction_max_translation"] >>
                temporal_prediction_max_translation;
        if (!fs["mask.temporal_prediction_max_rotation"].empty())
            fs["mask.temporal_prediction_max_rotation"] >>
                temporal_prediction_max_rotation;
        if (!fs["mask.adaptive_feature_relaxation"].empty())
            fs["mask.adaptive_feature_relaxation"] >>
                adaptive_feature_relaxation;
        if (!fs["mask.adaptive_feature_min_previous_inliers"].empty())
            fs["mask.adaptive_feature_min_previous_inliers"] >>
                adaptive_feature_min_previous_inliers;
        if (!fs["mask.adaptive_feature_hold_frames"].empty())
            fs["mask.adaptive_feature_hold_frames"] >>
                adaptive_feature_hold_frames;

        // Prior weights
        if (!fs["mask.w_sem"].empty())
            fs["mask.w_sem"] >> w_sem;
        if (!fs["mask.w_flow"].empty())
            fs["mask.w_flow"] >> w_flow;
        if (!fs["mask.w_geo"].empty())
            fs["mask.w_geo"] >> w_geo;

        // Output parameters
        if (!fs["mask.dyn_threshold"].empty())
            fs["mask.dyn_threshold"] >> dyn_threshold;
        if (!fs["mask.morph_kernel"].empty())
            fs["mask.morph_kernel"] >> morph_kernel;
        if (!fs["mask.soft_mapping_alpha"].empty())
            fs["mask.soft_mapping_alpha"] >> soft_mapping_alpha;
        if (!fs["mask.soft_mapping_min_weight"].empty())
            fs["mask.soft_mapping_min_weight"] >> soft_mapping_min_weight;

        if (!fs["mask.use_motion_pose_prior"].empty())
            fs["mask.use_motion_pose_prior"] >> use_motion_pose_prior;
        if (!fs["mask.motion_pose_prior_shadow_only"].empty())
            fs["mask.motion_pose_prior_shadow_only"] >> motion_pose_prior_shadow_only;
        if (!fs["mask.motion_max_features"].empty())
            fs["mask.motion_max_features"] >> motion_max_features;
        if (!fs["mask.motion_min_features_per_object"].empty())
            fs["mask.motion_min_features_per_object"] >> motion_min_features_per_object;
        if (!fs["mask.motion_min_component_area"].empty())
            fs["mask.motion_min_component_area"] >> motion_min_component_area;
        if (!fs["mask.motion_min_track_age"].empty())
            fs["mask.motion_min_track_age"] >> motion_min_track_age;
        if (!fs["mask.motion_ransac_threshold"].empty())
            fs["mask.motion_ransac_threshold"] >> motion_ransac_threshold;
        if (!fs["mask.motion_min_inlier_ratio"].empty())
            fs["mask.motion_min_inlier_ratio"] >> motion_min_inlier_ratio;
        if (!fs["mask.motion_min_information"].empty())
            fs["mask.motion_min_information"] >> motion_min_information;
        if (!fs["mask.motion_max_translation"].empty())
            fs["mask.motion_max_translation"] >> motion_max_translation;
        if (!fs["mask.motion_max_rotation"].empty())
            fs["mask.motion_max_rotation"] >> motion_max_rotation;
        if (!fs["mask.motion_use_depth_foreground_filter"].empty())
            fs["mask.motion_use_depth_foreground_filter"] >> motion_use_depth_foreground_filter;
        if (!fs["mask.motion_foreground_depth_separation"].empty())
            fs["mask.motion_foreground_depth_separation"] >> motion_foreground_depth_separation;
        if (!fs["mask.motion_foreground_min_fraction"].empty())
            fs["mask.motion_foreground_min_fraction"] >> motion_foreground_min_fraction;
        if (!fs["mask.motion_min_object_translation_speed"].empty())
            fs["mask.motion_min_object_translation_speed"] >> motion_min_object_translation_speed;
        if (!fs["mask.motion_min_object_rotation_speed"].empty())
            fs["mask.motion_min_object_rotation_speed"] >> motion_min_object_rotation_speed;
        if (!fs["mask.motion_measurement_translation_sigma"].empty())
            fs["mask.motion_measurement_translation_sigma"] >> motion_measurement_translation_sigma;
        if (!fs["mask.motion_measurement_rotation_sigma"].empty())
            fs["mask.motion_measurement_rotation_sigma"] >> motion_measurement_rotation_sigma;
        if (!fs["mask.motion_camera_translation_sigma"].empty())
            fs["mask.motion_camera_translation_sigma"] >> motion_camera_translation_sigma;
        if (!fs["mask.motion_camera_rotation_sigma"].empty())
            fs["mask.motion_camera_rotation_sigma"] >> motion_camera_rotation_sigma;
        if (!fs["mask.motion_velocity_process_translation_sigma"].empty())
            fs["mask.motion_velocity_process_translation_sigma"] >> motion_velocity_process_translation_sigma;
        if (!fs["mask.motion_velocity_process_rotation_sigma"].empty())
            fs["mask.motion_velocity_process_rotation_sigma"] >> motion_velocity_process_rotation_sigma;
        if (!fs["mask.motion_max_information_eigenvalue"].empty())
            fs["mask.motion_max_information_eigenvalue"] >> motion_max_information_eigenvalue;
        if (!fs["mask.motion_candidate_mahalanobis_threshold"].empty())
            fs["mask.motion_candidate_mahalanobis_threshold"] >>
                motion_candidate_mahalanobis_threshold;
        if (!fs["mask.motion_gate_min_inlier_gain"].empty())
            fs["mask.motion_gate_min_inlier_gain"] >> motion_gate_min_inlier_gain;
        if (!fs["mask.motion_gate_min_inlier_gain_ratio"].empty())
            fs["mask.motion_gate_min_inlier_gain_ratio"] >>
                motion_gate_min_inlier_gain_ratio;
        if (!fs["mask.motion_gate_max_static_inliers"].empty())
            fs["mask.motion_gate_max_static_inliers"] >>
                motion_gate_max_static_inliers;
        if (!fs["mask.motion_gate_min_translation_innovation"].empty())
            fs["mask.motion_gate_min_translation_innovation"] >>
                motion_gate_min_translation_innovation;
        if (!fs["mask.motion_gate_max_translation_innovation"].empty())
            fs["mask.motion_gate_max_translation_innovation"] >>
                motion_gate_max_translation_innovation;
        if (!fs["mask.motion_gate_max_rotation_innovation"].empty())
            fs["mask.motion_gate_max_rotation_innovation"] >>
                motion_gate_max_rotation_innovation;
        if (!fs["mask.motion_prior_information_scale"].empty())
            fs["mask.motion_prior_information_scale"] >>
                motion_prior_information_scale;
        if (!fs["mask.motion_prior_use_direct_validation"].empty())
            fs["mask.motion_prior_use_direct_validation"] >>
                motion_prior_use_direct_validation;
        if (!fs["mask.motion_prior_direct_score_mode"].empty())
            fs["mask.motion_prior_direct_score_mode"] >>
                motion_prior_direct_score_mode;
        if (!fs["mask.motion_prior_inject_local_map"].empty())
            fs["mask.motion_prior_inject_local_map"] >>
                motion_prior_inject_local_map;
        if (!fs["mask.motion_prior_initialization_only"].empty())
            fs["mask.motion_prior_initialization_only"] >>
                motion_prior_initialization_only;
        if (!fs["mask.motion_prior_require_common_support_improvement"].empty())
            fs["mask.motion_prior_require_common_support_improvement"] >>
                motion_prior_require_common_support_improvement;
        if (!fs["mask.motion_prior_bypass_reliability_gate"].empty())
            fs["mask.motion_prior_bypass_reliability_gate"] >>
                motion_prior_bypass_reliability_gate;
        if (!fs["mask.motion_prior_gate_policy"].empty())
            fs["mask.motion_prior_gate_policy"] >>
                motion_prior_gate_policy;
        if (!fs["mask.motion_prior_shuffle_lag_frames"].empty())
            fs["mask.motion_prior_shuffle_lag_frames"] >>
                motion_prior_shuffle_lag_frames;
        if (!fs["mask.motion_prior_shadow_translation_horizon"].empty())
            fs["mask.motion_prior_shadow_translation_horizon"] >>
                motion_prior_shadow_translation_horizon;
        if (!fs["mask.motion_prior_posterior_min_score_improvement"].empty())
            fs["mask.motion_prior_posterior_min_score_improvement"] >>
                motion_prior_posterior_min_score_improvement;
        if (!fs["mask.motion_prior_shadow_translation_blend"].empty())
            fs["mask.motion_prior_shadow_translation_blend"] >>
                motion_prior_shadow_translation_blend;
        if (!fs["mask.motion_prior_max_static_information_leverage"].empty())
            fs["mask.motion_prior_max_static_information_leverage"] >>
                motion_prior_max_static_information_leverage;
        if (!fs["mask.motion_prior_static_information_leverage_mode"].empty())
            fs["mask.motion_prior_static_information_leverage_mode"] >>
                motion_prior_static_information_leverage_mode;

        // Geo hard mask parameters
        if (!fs["mask.enable_geo_hard"].empty())
            fs["mask.enable_geo_hard"] >> enable_geo_hard;
        if (!fs["mask.geo_dyn_thresh"].empty())
            fs["mask.geo_dyn_thresh"] >> geo_dyn_thresh;
        if (!fs["mask.global_geo_min"].empty())
            fs["mask.global_geo_min"] >> global_geo_min;
        if (!fs["mask.vp_scale_min"].empty())
            fs["mask.vp_scale_min"] >> vp_scale_min;

        // Adaptive mechanism parameters
        if (!fs["mask.viewpoint_gating_rot_thresh"].empty())
            fs["mask.viewpoint_gating_rot_thresh"] >> viewpoint_gating_rot_thresh;
        if (!fs["mask.viewpoint_gating_trans_thresh"].empty())
            fs["mask.viewpoint_gating_trans_thresh"] >> viewpoint_gating_trans_thresh;
        if (!fs["mask.global_geo_warmup_frames"].empty())
            fs["mask.global_geo_warmup_frames"] >> global_geo_warmup_frames;
        if (!fs["mask.global_geo_no_semantic_thresh"].empty())
            fs["mask.global_geo_no_semantic_thresh"] >> global_geo_no_semantic_thresh;

        // YOLO parameters
        readStringNode(fs, "mask.yolo_model_path", yolo_model_path);
        if (!fs["mask.yolo_conf_threshold"].empty())
            fs["mask.yolo_conf_threshold"] >> yolo_conf_threshold;
        if (!fs["mask.yolo_nms_threshold"].empty())
            fs["mask.yolo_nms_threshold"] >> yolo_nms_threshold;
        if (!fs["mask.yolo_input_width"].empty())
            fs["mask.yolo_input_width"] >> yolo_input_width;
        if (!fs["mask.yolo_input_height"].empty())
            fs["mask.yolo_input_height"] >> yolo_input_height;

        // LK flow parameters
        if (!fs["mask.lk_win_size"].empty())
            fs["mask.lk_win_size"] >> lk_win_size;
        if (!fs["mask.lk_max_level"].empty())
            fs["mask.lk_max_level"] >> lk_max_level;
        if (!fs["mask.lk_grid_step"].empty())
            fs["mask.lk_grid_step"] >> lk_grid_step;

        // External precomputed mask parameters
        if (!fs["mask.use_external_mask"].empty())
            fs["mask.use_external_mask"] >> use_external_mask;
        readStringNode(fs, "mask.external_mask_dir", external_mask_dir);

        // Debug parameters
        if (!fs["mask.save_debug_images"].empty())
            fs["mask.save_debug_images"] >> save_debug_images;
        readStringNode(fs, "mask.debug_output_dir", debug_output_dir);

        yolo_every_n_frames = std::max(1, yolo_every_n_frames);
        warmup_frames = std::max(0, warmup_frames);
        temporal_background_warmup_frames =
            std::max(0, temporal_background_warmup_frames);
        tau_depth = std::max(1e-4f, tau_depth);
        temporal_render_weight =
            std::clamp(temporal_render_weight, 0.0f, 1.0f);
        temporal_observation_weight =
            std::clamp(temporal_observation_weight, 0.0f, 1.0f);
        temporal_depth_threshold =
            std::max(1e-4f, temporal_depth_threshold);
        temporal_neighbor_radius =
            std::max(0, temporal_neighbor_radius);
        const int temporal_window_width =
            2 * temporal_neighbor_radius + 1;
        temporal_min_consistent_neighbors = std::clamp(
            temporal_min_consistent_neighbors, 1,
            temporal_window_width * temporal_window_width);
        temporal_min_valid_fraction =
            std::clamp(temporal_min_valid_fraction, 0.0f, 1.0f);
        temporal_min_render_opacity =
            std::clamp(temporal_min_render_opacity, 0.0f, 1.0f);
        temporal_prediction_max_translation =
            std::max(0.0f, temporal_prediction_max_translation);
        temporal_prediction_max_rotation =
            std::max(0.0f, temporal_prediction_max_rotation);
        temporal_flow_guard_safe_radius =
            std::max(0, temporal_flow_guard_safe_radius);
        temporal_flow_guard_high_confidence_scale =
            std::max(1.0f, temporal_flow_guard_high_confidence_scale);
        adaptive_feature_relaxation =
            std::clamp(adaptive_feature_relaxation, 0.0f, 1.0f);
        adaptive_feature_min_previous_inliers =
            std::max(0, adaptive_feature_min_previous_inliers);
        adaptive_feature_hold_frames =
            std::max(1, adaptive_feature_hold_frames);
        viewpoint_gating_rot_thresh = std::max(1e-4f, viewpoint_gating_rot_thresh);
        viewpoint_gating_trans_thresh = std::max(1e-4f, viewpoint_gating_trans_thresh);
        soft_mapping_alpha = std::clamp(soft_mapping_alpha, 0.0f, 1.0f);
        soft_mapping_min_weight = std::clamp(soft_mapping_min_weight, 0.0f, 1.0f);
        motion_max_features = std::max(1, motion_max_features);
        motion_min_features_per_object = std::max(3, motion_min_features_per_object);
        motion_min_component_area = std::max(1, motion_min_component_area);
        motion_min_track_age = std::max(1, motion_min_track_age);
        motion_ransac_threshold = std::max(1e-4f, motion_ransac_threshold);
        motion_min_inlier_ratio = std::clamp(motion_min_inlier_ratio, 0.0f, 1.0f);
        motion_min_information = std::max(0.0f, motion_min_information);
        motion_max_translation = std::max(0.0f, motion_max_translation);
        motion_max_rotation = std::max(0.0f, motion_max_rotation);
        motion_foreground_depth_separation =
            std::max(0.0f, motion_foreground_depth_separation);
        motion_foreground_min_fraction =
            std::clamp(motion_foreground_min_fraction, 0.01f, 0.49f);
        motion_min_object_translation_speed =
            std::max(0.0f, motion_min_object_translation_speed);
        motion_min_object_rotation_speed =
            std::max(0.0f, motion_min_object_rotation_speed);
        motion_measurement_translation_sigma =
            std::max(1e-4f, motion_measurement_translation_sigma);
        motion_measurement_rotation_sigma =
            std::max(1e-4f, motion_measurement_rotation_sigma);
        motion_camera_translation_sigma =
            std::max(1e-4f, motion_camera_translation_sigma);
        motion_camera_rotation_sigma =
            std::max(1e-4f, motion_camera_rotation_sigma);
        motion_velocity_process_translation_sigma =
            std::max(0.0f, motion_velocity_process_translation_sigma);
        motion_velocity_process_rotation_sigma =
            std::max(0.0f, motion_velocity_process_rotation_sigma);
        motion_max_information_eigenvalue =
            std::max(motion_min_information, motion_max_information_eigenvalue);
        motion_candidate_mahalanobis_threshold =
            std::max(0.0f, motion_candidate_mahalanobis_threshold);
        motion_gate_min_inlier_gain = std::max(0, motion_gate_min_inlier_gain);
        motion_gate_min_inlier_gain_ratio =
            std::max(0.0f, motion_gate_min_inlier_gain_ratio);
        motion_gate_max_static_inliers =
            std::max(1, motion_gate_max_static_inliers);
        motion_gate_min_translation_innovation =
            std::max(0.0f, motion_gate_min_translation_innovation);
        motion_gate_max_translation_innovation =
            std::max(
                motion_gate_min_translation_innovation,
                motion_gate_max_translation_innovation);
        motion_gate_max_rotation_innovation =
            std::max(0.0f, motion_gate_max_rotation_innovation);
        motion_prior_information_scale =
            std::clamp(motion_prior_information_scale, 0.0f, 1.0f);
        motion_prior_direct_score_mode =
            std::clamp(motion_prior_direct_score_mode, 0, 2);
        motion_prior_gate_policy =
            motion_prior_gate_policy == 1 ? 1 : 0;
        motion_prior_shuffle_lag_frames =
            std::max(0, motion_prior_shuffle_lag_frames);
        motion_prior_posterior_min_score_improvement =
            std::max(0.0f, motion_prior_posterior_min_score_improvement);
        motion_prior_shadow_translation_blend =
            std::clamp(motion_prior_shadow_translation_blend, 0.0f, 1.0f);
        motion_prior_max_static_information_leverage =
            std::max(0.0f, motion_prior_max_static_information_leverage);
        motion_prior_static_information_leverage_mode = std::clamp(
            motion_prior_static_information_leverage_mode, 0, 1);

        fs.release();
        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "[MaskConfig] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
}

bool MaskConfig::saveToYAML(const std::string& yaml_path) const
{
    try {
        cv::FileStorage fs(yaml_path, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "[MaskConfig] Cannot create config file: " << yaml_path << std::endl;
            return false;
        }

        fs << "mask.use_yolo" << use_yolo;
        fs << "mask.use_flow" << use_flow;
        fs << "mask.use_depth_consistency" << use_depth_consistency;
        fs << "mask.use_soft_mapping" << use_soft_mapping;
        fs << "mask.use_hard_mapping_mask"
           << use_hard_mapping_mask;
        fs << "mask.enable_flow_hard" << enable_flow_hard;
        fs << "mask.use_temporal_background_refinement"
           << use_temporal_background_refinement;
        fs << "mask.use_adaptive_feature_extraction"
           << use_adaptive_feature_extraction;
        fs << "mask.temporal_recovery_only"
           << temporal_recovery_only;
        fs << "mask.temporal_conservative_mapping"
           << temporal_conservative_mapping;
        fs << "mask.temporal_recovery_flow_guard"
           << temporal_recovery_flow_guard;
        fs << "mask.temporal_flow_guard_safe_radius"
           << temporal_flow_guard_safe_radius;
        fs << "mask.temporal_flow_guard_adaptive_radius"
           << temporal_flow_guard_adaptive_radius;
        fs << "mask.temporal_flow_guard_high_confidence_scale"
           << temporal_flow_guard_high_confidence_scale;

        fs << "mask.warmup_frames" << warmup_frames;
        fs << "mask.yolo_every_n_frames" << yolo_every_n_frames;
        fs << "mask.temporal_background_warmup_frames"
           << temporal_background_warmup_frames;

        fs << "mask.tau_flow" << tau_flow;
        fs << "mask.tau_depth" << tau_depth;
        fs << "mask.temporal_render_weight" << temporal_render_weight;
        fs << "mask.temporal_observation_weight"
           << temporal_observation_weight;
        fs << "mask.temporal_depth_threshold"
           << temporal_depth_threshold;
        fs << "mask.temporal_neighbor_radius"
           << temporal_neighbor_radius;
        fs << "mask.temporal_min_consistent_neighbors"
           << temporal_min_consistent_neighbors;
        fs << "mask.temporal_min_valid_fraction"
           << temporal_min_valid_fraction;
        fs << "mask.temporal_min_render_opacity"
           << temporal_min_render_opacity;
        fs << "mask.temporal_prediction_max_translation"
           << temporal_prediction_max_translation;
        fs << "mask.temporal_prediction_max_rotation"
           << temporal_prediction_max_rotation;
        fs << "mask.adaptive_feature_relaxation"
           << adaptive_feature_relaxation;
        fs << "mask.adaptive_feature_min_previous_inliers"
           << adaptive_feature_min_previous_inliers;
        fs << "mask.adaptive_feature_hold_frames"
           << adaptive_feature_hold_frames;

        fs << "mask.w_sem" << w_sem;
        fs << "mask.w_flow" << w_flow;
        fs << "mask.w_geo" << w_geo;

        fs << "mask.dyn_threshold" << dyn_threshold;
        fs << "mask.morph_kernel" << morph_kernel;
        fs << "mask.soft_mapping_alpha" << soft_mapping_alpha;
        fs << "mask.soft_mapping_min_weight" << soft_mapping_min_weight;

        fs << "mask.use_motion_pose_prior" << use_motion_pose_prior;
        fs << "mask.motion_pose_prior_shadow_only" << motion_pose_prior_shadow_only;
        fs << "mask.motion_max_features" << motion_max_features;
        fs << "mask.motion_min_features_per_object" << motion_min_features_per_object;
        fs << "mask.motion_min_component_area" << motion_min_component_area;
        fs << "mask.motion_min_track_age" << motion_min_track_age;
        fs << "mask.motion_ransac_threshold" << motion_ransac_threshold;
        fs << "mask.motion_min_inlier_ratio" << motion_min_inlier_ratio;
        fs << "mask.motion_min_information" << motion_min_information;
        fs << "mask.motion_max_translation" << motion_max_translation;
        fs << "mask.motion_max_rotation" << motion_max_rotation;
        fs << "mask.motion_use_depth_foreground_filter" << motion_use_depth_foreground_filter;
        fs << "mask.motion_foreground_depth_separation" << motion_foreground_depth_separation;
        fs << "mask.motion_foreground_min_fraction" << motion_foreground_min_fraction;
        fs << "mask.motion_min_object_translation_speed" << motion_min_object_translation_speed;
        fs << "mask.motion_min_object_rotation_speed" << motion_min_object_rotation_speed;
        fs << "mask.motion_measurement_translation_sigma" << motion_measurement_translation_sigma;
        fs << "mask.motion_measurement_rotation_sigma" << motion_measurement_rotation_sigma;
        fs << "mask.motion_camera_translation_sigma" << motion_camera_translation_sigma;
        fs << "mask.motion_camera_rotation_sigma" << motion_camera_rotation_sigma;
        fs << "mask.motion_velocity_process_translation_sigma" << motion_velocity_process_translation_sigma;
        fs << "mask.motion_velocity_process_rotation_sigma" << motion_velocity_process_rotation_sigma;
        fs << "mask.motion_max_information_eigenvalue" << motion_max_information_eigenvalue;
        fs << "mask.motion_candidate_mahalanobis_threshold"
           << motion_candidate_mahalanobis_threshold;
        fs << "mask.motion_gate_min_inlier_gain" << motion_gate_min_inlier_gain;
        fs << "mask.motion_gate_min_inlier_gain_ratio"
           << motion_gate_min_inlier_gain_ratio;
        fs << "mask.motion_gate_max_static_inliers"
           << motion_gate_max_static_inliers;
        fs << "mask.motion_gate_min_translation_innovation"
           << motion_gate_min_translation_innovation;
        fs << "mask.motion_gate_max_translation_innovation"
           << motion_gate_max_translation_innovation;
        fs << "mask.motion_gate_max_rotation_innovation"
           << motion_gate_max_rotation_innovation;
        fs << "mask.motion_prior_information_scale"
           << motion_prior_information_scale;
        fs << "mask.motion_prior_use_direct_validation"
           << motion_prior_use_direct_validation;
        fs << "mask.motion_prior_direct_score_mode"
           << motion_prior_direct_score_mode;
        fs << "mask.motion_prior_inject_local_map"
           << motion_prior_inject_local_map;
        fs << "mask.motion_prior_initialization_only"
           << motion_prior_initialization_only;
        fs << "mask.motion_prior_require_common_support_improvement"
           << motion_prior_require_common_support_improvement;
        fs << "mask.motion_prior_bypass_reliability_gate"
           << motion_prior_bypass_reliability_gate;
        fs << "mask.motion_prior_gate_policy"
           << motion_prior_gate_policy;
        fs << "mask.motion_prior_shuffle_lag_frames"
           << motion_prior_shuffle_lag_frames;
        fs << "mask.motion_prior_shadow_translation_horizon"
           << motion_prior_shadow_translation_horizon;
        fs << "mask.motion_prior_posterior_min_score_improvement"
           << motion_prior_posterior_min_score_improvement;
        fs << "mask.motion_prior_shadow_translation_blend"
           << motion_prior_shadow_translation_blend;
        fs << "mask.motion_prior_max_static_information_leverage"
           << motion_prior_max_static_information_leverage;
        fs << "mask.motion_prior_static_information_leverage_mode"
           << motion_prior_static_information_leverage_mode;

        fs << "mask.enable_geo_hard" << enable_geo_hard;
        fs << "mask.geo_dyn_thresh" << geo_dyn_thresh;
        fs << "mask.global_geo_min" << global_geo_min;
        fs << "mask.vp_scale_min" << vp_scale_min;

        fs << "mask.viewpoint_gating_rot_thresh" << viewpoint_gating_rot_thresh;
        fs << "mask.viewpoint_gating_trans_thresh" << viewpoint_gating_trans_thresh;
        fs << "mask.global_geo_warmup_frames" << global_geo_warmup_frames;
        fs << "mask.global_geo_no_semantic_thresh" << global_geo_no_semantic_thresh;

        fs << "mask.use_external_mask" << use_external_mask;
        fs << "mask.external_mask_dir" << external_mask_dir;

        fs << "mask.yolo_model_path" << yolo_model_path;
        fs << "mask.yolo_conf_threshold" << yolo_conf_threshold;
        fs << "mask.yolo_nms_threshold" << yolo_nms_threshold;
        fs << "mask.yolo_input_width" << yolo_input_width;
        fs << "mask.yolo_input_height" << yolo_input_height;

        fs << "mask.lk_win_size" << lk_win_size;
        fs << "mask.lk_max_level" << lk_max_level;
        fs << "mask.lk_grid_step" << lk_grid_step;

        fs << "mask.save_debug_images" << save_debug_images;
        fs << "mask.debug_output_dir" << debug_output_dir;

        fs.release();
        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "[MaskConfig] OpenCV exception: " << e.what() << std::endl;
        return false;
    }
}

void MaskConfig::print() const
{
    std::cout << "\n========== DynamicMaskRefiner Configuration ==========" << std::endl;
    std::cout << "[Feature Toggles]" << std::endl;
    std::cout << "  use_yolo:              " << (use_yolo ? "true" : "false") << std::endl;
    std::cout << "  use_flow:              " << (use_flow ? "true" : "false") << std::endl;
    std::cout << "  use_depth_consistency: " << (use_depth_consistency ? "true" : "false") << std::endl;
    std::cout << "  use_soft_mapping:      " << (use_soft_mapping ? "true" : "false") << std::endl;
    std::cout << "  use_hard_mapping_mask: "
              << (use_hard_mapping_mask ? "true" : "false")
              << std::endl;
    std::cout << "  enable_flow_hard:      " << (enable_flow_hard ? "true" : "false") << std::endl;
    std::cout << "  temporal_background:  "
              << (use_temporal_background_refinement ? "true" : "false")
              << std::endl;
    std::cout << "  adaptive_features:    "
              << (use_adaptive_feature_extraction ? "true" : "false")
              << std::endl;
    std::cout << "  temporal_recovery:    "
              << (temporal_recovery_only ? "true" : "false")
              << std::endl;
    std::cout << "  conservative_mapping: "
              << (temporal_conservative_mapping ? "true" : "false")
              << std::endl;
    std::cout << "  recovery_flow_guard:  "
              << (temporal_recovery_flow_guard ? "true" : "false")
              << std::endl;
    std::cout << "  flow_guard_radius:    "
              << temporal_flow_guard_safe_radius << std::endl;
    std::cout << "  adaptive_guard_radius:"
              << (temporal_flow_guard_adaptive_radius ? " true" : " false")
              << std::endl;
    std::cout << "  guard_high_conf_scale:"
              << temporal_flow_guard_high_confidence_scale << std::endl;

    std::cout << "[Timing Parameters]" << std::endl;
    std::cout << "  warmup_frames:         " << warmup_frames << std::endl;
    std::cout << "  yolo_every_n_frames:   " << yolo_every_n_frames << std::endl;
    std::cout << "  temporal_warmup:       "
              << temporal_background_warmup_frames << std::endl;

    std::cout << "[Threshold Parameters]" << std::endl;
    std::cout << "  tau_flow:              " << std::fixed << std::setprecision(2) << tau_flow << " px" << std::endl;
    std::cout << "  tau_depth:             " << std::fixed << std::setprecision(3) << tau_depth << " m" << std::endl;

    std::cout << "[Prior-image Background Refinement]" << std::endl;
    std::cout << "  render_weight:         " << temporal_render_weight << std::endl;
    std::cout << "  observation_weight:    " << temporal_observation_weight << std::endl;
    std::cout << "  depth_threshold:       "
              << temporal_depth_threshold << " m" << std::endl;
    std::cout << "  neighbor_radius:       "
              << temporal_neighbor_radius << std::endl;
    std::cout << "  min_neighbors:         "
              << temporal_min_consistent_neighbors << std::endl;
    std::cout << "  min_valid_fraction:    "
              << temporal_min_valid_fraction << std::endl;
    std::cout << "  min_render_opacity:    "
              << temporal_min_render_opacity << std::endl;
    std::cout << "  prediction_max_trans:  "
              << temporal_prediction_max_translation << " m" << std::endl;
    std::cout << "  prediction_max_rot:    "
              << temporal_prediction_max_rotation << " rad" << std::endl;
    std::cout << "  flow_guard_radius:     "
              << temporal_flow_guard_safe_radius << " px" << std::endl;
    std::cout << "  adaptive_guard_radius: "
              << (temporal_flow_guard_adaptive_radius ? "true" : "false")
              << std::endl;
    std::cout << "  guard_high_conf_scale: "
              << temporal_flow_guard_high_confidence_scale << std::endl;
    std::cout << "  feature_relaxation:    "
              << adaptive_feature_relaxation << std::endl;
    std::cout << "  feature_min_inliers:   "
              << adaptive_feature_min_previous_inliers << std::endl;
    std::cout << "  feature_hold_frames:   "
              << adaptive_feature_hold_frames << std::endl;

    std::cout << "[Prior Weights]" << std::endl;
    std::cout << "  w_sem:                 " << std::fixed << std::setprecision(2) << w_sem << std::endl;
    std::cout << "  w_flow:                " << std::fixed << std::setprecision(2) << w_flow << std::endl;
    std::cout << "  w_geo:                 " << std::fixed << std::setprecision(2) << w_geo << std::endl;

    std::cout << "[Output Parameters]" << std::endl;
    std::cout << "  dyn_threshold:         " << std::fixed << std::setprecision(2) << dyn_threshold << std::endl;
    std::cout << "  morph_kernel:          " << morph_kernel << std::endl;
    std::cout << "  soft_mapping_alpha:    " << std::fixed << std::setprecision(2) << soft_mapping_alpha << std::endl;
    std::cout << "  soft_mapping_min:      " << std::fixed << std::setprecision(2) << soft_mapping_min_weight << std::endl;

    std::cout << "[Motion-marginalized Pose Prior]" << std::endl;
    std::cout << "  enabled:               " << (use_motion_pose_prior ? "true" : "false") << std::endl;
    std::cout << "  shadow_only:           "
              << (motion_pose_prior_shadow_only ? "true" : "false") << std::endl;
    std::cout << "  max_features:          " << motion_max_features << std::endl;
    std::cout << "  min_features/object:   " << motion_min_features_per_object << std::endl;
    std::cout << "  min_track_age:         " << motion_min_track_age << std::endl;
    std::cout << "  min_information:       " << motion_min_information << std::endl;
    std::cout << "  depth_foreground:      "
              << (motion_use_depth_foreground_filter ? "true" : "false")
              << std::endl;
    std::cout << "  min_translation_speed: "
              << motion_min_object_translation_speed << " m/s" << std::endl;
    std::cout << "  min_rotation_speed:    "
              << motion_min_object_rotation_speed << " rad/s" << std::endl;
    std::cout << "  candidate_mahalanobis: "
              << motion_candidate_mahalanobis_threshold << std::endl;
    std::cout << "  gate_min_inlier_gain:  "
              << motion_gate_min_inlier_gain << std::endl;
    std::cout << "  gate_min_gain_ratio:   "
              << motion_gate_min_inlier_gain_ratio << std::endl;
    std::cout << "  gate_max_static:       "
              << motion_gate_max_static_inliers << std::endl;
    std::cout << "  gate_min_translation:  "
              << std::fixed << std::setprecision(4)
              << motion_gate_min_translation_innovation << " m" << std::endl;
    std::cout << "  gate_max_translation:  "
              << std::fixed << std::setprecision(4)
              << motion_gate_max_translation_innovation << " m" << std::endl;
    std::cout << "  gate_max_rotation:     "
              << std::fixed << std::setprecision(6)
              << motion_gate_max_rotation_innovation << " rad" << std::endl;
    std::cout << "  prior_information_scale: "
              << motion_prior_information_scale << std::endl;
    std::cout << "  direct_validation:      "
              << (motion_prior_use_direct_validation ? "true" : "false")
              << std::endl;
    std::cout << "  direct_score_mode:      "
              << motion_prior_direct_score_mode << std::endl;
    std::cout << "  inject_local_map:      "
              << (motion_prior_inject_local_map ? "true" : "false")
              << std::endl;
    std::cout << "  initialization_only:   "
              << (motion_prior_initialization_only ? "true" : "false")
              << std::endl;
    std::cout << "  require_common_support:"
              << (motion_prior_require_common_support_improvement
                      ? " true" : " false")
              << std::endl;
    std::cout << "  bypass_reliability_gate:"
              << (motion_prior_bypass_reliability_gate
                      ? " true" : " false")
              << std::endl;
    std::cout << "  gate_policy:           "
              << (motion_prior_gate_policy == 1
                      ? "direct-combined" : "legacy-conjunction")
              << std::endl;
    std::cout << "  shuffle_lag_frames:    "
              << motion_prior_shuffle_lag_frames << std::endl;
    std::cout << "  shadow_translation_horizon: "
              << motion_prior_shadow_translation_horizon << std::endl;
    std::cout << "  posterior_min_score_improvement: "
              << motion_prior_posterior_min_score_improvement << std::endl;
    std::cout << "  shadow_translation_blend: "
              << motion_prior_shadow_translation_blend << std::endl;
    std::cout << "  max_static_information_leverage: "
              << motion_prior_max_static_information_leverage << std::endl;
    std::cout << "  static_information_leverage_mode: "
              << (motion_prior_static_information_leverage_mode == 1
                      ? "normalize-to-target" : "cap-only")
              << std::endl;

    std::cout << "[Geo Hard Mask Parameters]" << std::endl;
    std::cout << "  enable_geo_hard:       " << (enable_geo_hard ? "true" : "false") << std::endl;
    std::cout << "  geo_dyn_thresh:        " << std::fixed << std::setprecision(2) << geo_dyn_thresh << std::endl;
    std::cout << "  global_geo_min:        " << std::fixed << std::setprecision(2) << global_geo_min << std::endl;
    std::cout << "  vp_scale_min:          " << std::fixed << std::setprecision(2) << vp_scale_min << std::endl;

    std::cout << "[Adaptive Mechanism Parameters]" << std::endl;
    std::cout << "  viewpoint_gating_rot_thresh:    " << std::fixed << std::setprecision(4) << viewpoint_gating_rot_thresh << " rad" << std::endl;
    std::cout << "  viewpoint_gating_trans_thresh:  " << std::fixed << std::setprecision(2) << viewpoint_gating_trans_thresh << " m" << std::endl;
    std::cout << "  global_geo_warmup_frames:       " << global_geo_warmup_frames << std::endl;
    std::cout << "  global_geo_no_semantic_thresh:  " << global_geo_no_semantic_thresh << std::endl;

    std::cout << "[External Mask Parameters]" << std::endl;
    std::cout << "  use_external_mask:     " << (use_external_mask ? "true" : "false") << std::endl;
    std::cout << "  external_mask_dir:     " << external_mask_dir << std::endl;

    std::cout << "[YOLO Parameters]" << std::endl;
    std::cout << "  yolo_model_path:       " << yolo_model_path << std::endl;
    std::cout << "  yolo_conf_threshold:   " << std::fixed << std::setprecision(2) << yolo_conf_threshold << std::endl;
    std::cout << "  yolo_input_size:       " << yolo_input_width << "x" << yolo_input_height << std::endl;

    std::cout << "[LK Flow Parameters]" << std::endl;
    std::cout << "  lk_win_size:           " << lk_win_size << std::endl;
    std::cout << "  lk_max_level:          " << lk_max_level << std::endl;
    std::cout << "  lk_grid_step:          " << lk_grid_step << std::endl;

    std::cout << "======================================================\n" << std::endl;
}

} // namespace DyGeoFusion
