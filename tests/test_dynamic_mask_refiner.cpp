#include <gtest/gtest.h>

#include "Mask/DynamicMaskRefiner.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

TEST(DynamicMaskRefinerTest, DefaultsShadowTranslationInterventionOff) {
    const DyGeoFusion::MaskConfig config;

    EXPECT_EQ(config.motion_prior_shadow_translation_horizon, 0);
    EXPECT_FLOAT_EQ(
        config.motion_prior_max_static_information_leverage, 0.0f);
    EXPECT_EQ(config.motion_prior_static_information_leverage_mode, 0);
    EXPECT_EQ(config.adaptive_feature_min_previous_inliers, 0);
    EXPECT_EQ(config.adaptive_feature_hold_frames, 1);
    EXPECT_EQ(config.temporal_flow_guard_safe_radius, 0);
    EXPECT_FALSE(config.temporal_flow_guard_adaptive_radius);
    EXPECT_FLOAT_EQ(
        config.temporal_flow_guard_high_confidence_scale, 2.0f);
}

TEST(DynamicMaskRefinerTest, RejectsEmptyRequiredYoloEngine) {
    DyGeoFusion::MaskConfig config;
    config.use_yolo = true;
    config.use_external_mask = false;
    config.yolo_model_path.clear();

    EXPECT_THROW(
        DyGeoFusion::DynamicMaskRefiner(config, torch::kCPU),
        std::invalid_argument);
}

TEST(DynamicMaskRefinerTest, RejectsMissingRequiredYoloEngine) {
    DyGeoFusion::MaskConfig config;
    config.use_yolo = true;
    config.use_external_mask = false;
    config.yolo_model_path =
        (std::filesystem::temp_directory_path() /
         "dynags_missing_yolo_engine.engine").string();

    EXPECT_THROW(
        DyGeoFusion::DynamicMaskRefiner(config, torch::kCPU),
        std::runtime_error);
}

TEST(DynamicMaskRefinerTest, LoadsMotionAblationControls) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "dynags_motion_ablation_controls.yaml";
    {
        std::ofstream storage(path);
        ASSERT_TRUE(storage.is_open());
        storage << "%YAML:1.0\n"
                << "mask.motion_prior_bypass_reliability_gate: 1\n"
                << "mask.motion_prior_gate_policy: 1\n"
                << "mask.motion_prior_shuffle_lag_frames: 30\n"
                << "mask.motion_prior_shadow_translation_horizon: 10\n"
                << "mask.motion_prior_max_static_information_leverage: 0.25\n"
                << "mask.motion_prior_static_information_leverage_mode: 1\n";
    }

    DyGeoFusion::MaskConfig config;
    ASSERT_TRUE(config.loadFromYAML(path.string()));
    std::filesystem::remove(path);

    EXPECT_TRUE(config.motion_prior_bypass_reliability_gate);
    EXPECT_EQ(config.motion_prior_gate_policy, 1);
    EXPECT_EQ(config.motion_prior_shuffle_lag_frames, 30);
    EXPECT_EQ(config.motion_prior_shadow_translation_horizon, 10);
    EXPECT_FLOAT_EQ(
        config.motion_prior_max_static_information_leverage, 0.25f);
    EXPECT_EQ(config.motion_prior_static_information_leverage_mode, 1);
}

TEST(DynamicMaskRefinerTest, LoadsDyPhoCompatibleControls) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "dynags_dypho_compatible_controls.yaml";
    {
        std::ofstream storage(path);
        ASSERT_TRUE(storage.is_open());
        storage << "%YAML:1.0\n"
                << "mask.use_temporal_background_refinement: 1\n"
                << "mask.use_adaptive_feature_extraction: 1\n"
                << "mask.temporal_recovery_flow_guard: 1\n"
                << "mask.temporal_flow_guard_safe_radius: 2\n"
                << "mask.temporal_flow_guard_adaptive_radius: 1\n"
                << "mask.temporal_flow_guard_high_confidence_scale: 1.75\n"
                << "mask.temporal_background_warmup_frames: 7\n"
                << "mask.temporal_render_weight: 0.7\n"
                << "mask.temporal_observation_weight: 0.1\n"
                << "mask.temporal_depth_threshold: 0.08\n"
                << "mask.temporal_neighbor_radius: 3\n"
                << "mask.temporal_min_consistent_neighbors: 9\n"
                << "mask.temporal_min_valid_fraction: 0.15\n"
                << "mask.temporal_min_render_opacity: 0.55\n"
                << "mask.temporal_prediction_max_translation: 0.4\n"
                << "mask.temporal_prediction_max_rotation: 0.6\n"
                << "mask.adaptive_feature_relaxation: 0.4\n"
                << "mask.adaptive_feature_min_previous_inliers: 120\n"
                << "mask.adaptive_feature_hold_frames: 4\n";
    }

    DyGeoFusion::MaskConfig config;
    ASSERT_TRUE(config.loadFromYAML(path.string()));
    std::filesystem::remove(path);

    EXPECT_TRUE(config.use_temporal_background_refinement);
    EXPECT_TRUE(config.use_adaptive_feature_extraction);
    EXPECT_TRUE(config.temporal_recovery_flow_guard);
    EXPECT_EQ(config.temporal_flow_guard_safe_radius, 2);
    EXPECT_TRUE(config.temporal_flow_guard_adaptive_radius);
    EXPECT_FLOAT_EQ(
        config.temporal_flow_guard_high_confidence_scale, 1.75f);
    EXPECT_EQ(config.temporal_background_warmup_frames, 7);
    EXPECT_FLOAT_EQ(config.temporal_render_weight, 0.7f);
    EXPECT_FLOAT_EQ(config.temporal_observation_weight, 0.1f);
    EXPECT_FLOAT_EQ(config.temporal_depth_threshold, 0.08f);
    EXPECT_EQ(config.temporal_neighbor_radius, 3);
    EXPECT_EQ(config.temporal_min_consistent_neighbors, 9);
    EXPECT_FLOAT_EQ(config.temporal_min_valid_fraction, 0.15f);
    EXPECT_FLOAT_EQ(config.temporal_min_render_opacity, 0.55f);
    EXPECT_FLOAT_EQ(config.temporal_prediction_max_translation, 0.4f);
    EXPECT_FLOAT_EQ(config.temporal_prediction_max_rotation, 0.6f);
    EXPECT_FLOAT_EQ(config.adaptive_feature_relaxation, 0.4f);
    EXPECT_EQ(config.adaptive_feature_min_previous_inliers, 120);
    EXPECT_EQ(config.adaptive_feature_hold_frames, 4);
}

TEST(DynamicMaskRefinerTest, LoadsFrozenDyPhoPresetMatrix) {
    const std::filesystem::path source_dir =
        std::filesystem::path(__FILE__).parent_path().parent_path();
    struct ExpectedPreset {
        const char* filename;
        bool temporal;
        bool adaptive;
        bool flow_hard;
    };
    const std::vector<ExpectedPreset> presets = {
        {"mask_config_dypho_raw.yaml", false, false, true},
        {"mask_config_dypho_temporal.yaml", true, false, true},
        {"mask_config_dypho_feature.yaml", false, true, true},
        {"mask_config_dypho_compatible.yaml", true, true, true},
        {"mask_config_dypho_decoupled.yaml", true, true, true},
        {"mask_config_dypho_flow_guarded.yaml", true, true, true},
        {"mask_config_dypho_flow_exact.yaml", true, true, true},
        {"mask_config_dypho_flow_adaptive.yaml", true, true, true},
    };

    for (const ExpectedPreset& expected : presets) {
        DyGeoFusion::MaskConfig config;
        ASSERT_TRUE(config.loadFromYAML(
            (source_dir / "config" / expected.filename).string()))
            << expected.filename;
        EXPECT_EQ(
            config.use_temporal_background_refinement,
            expected.temporal) << expected.filename;
        EXPECT_EQ(
            config.use_adaptive_feature_extraction,
            expected.adaptive) << expected.filename;
        EXPECT_EQ(config.enable_flow_hard, expected.flow_hard)
            << expected.filename;
        EXPECT_TRUE(config.use_hard_mapping_mask)
            << expected.filename;
        EXPECT_FALSE(config.use_motion_pose_prior)
            << expected.filename;
        if (expected.temporal) {
            EXPECT_FLOAT_EQ(config.temporal_depth_threshold, 0.2f)
                << expected.filename;
            EXPECT_EQ(config.temporal_min_consistent_neighbors, 9)
                << expected.filename;
            EXPECT_FLOAT_EQ(config.temporal_min_render_opacity, 0.5f)
                << expected.filename;
        }
        if (expected.adaptive) {
            EXPECT_FLOAT_EQ(config.adaptive_feature_relaxation, 0.9f)
                << expected.filename;
        }
        if (std::string(expected.filename) ==
                "mask_config_dypho_decoupled.yaml" ||
            std::string(expected.filename) ==
                "mask_config_dypho_flow_guarded.yaml" ||
            std::string(expected.filename) ==
                "mask_config_dypho_flow_exact.yaml" ||
            std::string(expected.filename) ==
                "mask_config_dypho_flow_adaptive.yaml") {
            EXPECT_TRUE(config.temporal_recovery_only);
            EXPECT_TRUE(config.temporal_conservative_mapping);
        }
        const bool flow_guard =
            std::string(expected.filename) ==
                "mask_config_dypho_flow_guarded.yaml" ||
            std::string(expected.filename) ==
                "mask_config_dypho_flow_exact.yaml" ||
            std::string(expected.filename) ==
                "mask_config_dypho_flow_adaptive.yaml";
        EXPECT_EQ(
            config.temporal_recovery_flow_guard,
            flow_guard);
        if (flow_guard) {
            EXPECT_EQ(
                config.adaptive_feature_min_previous_inliers, 150);
            EXPECT_EQ(config.adaptive_feature_hold_frames, 5);
            const bool exact_guard =
                std::string(expected.filename) ==
                    "mask_config_dypho_flow_exact.yaml";
            EXPECT_EQ(
                config.temporal_flow_guard_safe_radius,
                exact_guard ? 0 : 1);
        } else {
            EXPECT_EQ(config.temporal_flow_guard_safe_radius, 0);
        }
        const bool adaptive_guard =
            std::string(expected.filename) ==
                "mask_config_dypho_flow_adaptive.yaml";
        EXPECT_EQ(
            config.temporal_flow_guard_adaptive_radius,
            adaptive_guard);
        EXPECT_FLOAT_EQ(
            config.temporal_flow_guard_high_confidence_scale, 2.0f);
    }
}

TEST(DynamicMaskRefinerTest, AdaptiveRadiusExpandsOnlyHighConfidenceGuard) {
    cv::Mat exact_guard = cv::Mat::zeros(5, 5, CV_8UC1);
    cv::Mat high_confidence_guard =
        cv::Mat::zeros(5, 5, CV_8UC1);
    exact_guard.at<uchar>(2, 3) = 1;

    EXPECT_TRUE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            exact_guard, high_confidence_guard, 2, 2, 1, true));
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            exact_guard, high_confidence_guard, 3, 2, 1, true));

    high_confidence_guard.at<uchar>(2, 3) = 1;
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            exact_guard, high_confidence_guard, 2, 2, 1, true));
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            exact_guard, cv::Mat(), 2, 2, 1, true));
}

TEST(DynamicMaskRefinerTest, RecoveryFlowGuardSafeRadiusRejectsNeighbors) {
    cv::Mat guard = cv::Mat::zeros(5, 5, CV_8UC1);
    guard.at<uchar>(2, 3) = 1;

    EXPECT_TRUE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            guard, 2, 2, 0));
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            guard, 2, 2, 1));
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            cv::Mat(), 2, 2, 1));
    EXPECT_FALSE(
        DyGeoFusion::DynamicMaskRefiner::recoveryFlowGuardAllowsPixel(
            guard, -1, 2, 0));
}

TEST(DynamicMaskRefinerTest, TemporalBackgroundCorrectsBothMaskDirections) {
    const std::filesystem::path mask_dir =
        std::filesystem::temp_directory_path() /
        "dynags_temporal_background_masks";
    std::filesystem::create_directories(mask_dir);

    cv::Mat first_mask = cv::Mat::zeros(9, 9, CV_8UC1);
    cv::Mat second_mask = cv::Mat::zeros(9, 9, CV_8UC1);
    second_mask(cv::Rect(1, 3, 3, 3)).setTo(255);
    second_mask(cv::Rect(5, 3, 3, 3)).setTo(255);
    ASSERT_TRUE(cv::imwrite(
        (mask_dir / "00000000.png").string(), first_mask));
    ASSERT_TRUE(cv::imwrite(
        (mask_dir / "00000001.png").string(), second_mask));

    DyGeoFusion::MaskConfig config;
    config.use_yolo = false;
    config.use_external_mask = true;
    config.external_mask_dir = mask_dir.string();
    config.use_flow = false;
    config.use_temporal_background_refinement = true;
    config.temporal_background_warmup_frames = 0;
    config.temporal_render_weight = 0.6f;
    config.temporal_observation_weight = 0.2f;
    config.temporal_depth_threshold = 0.1f;
    config.temporal_neighbor_radius = 1;
    config.temporal_min_consistent_neighbors = 5;
    config.temporal_min_valid_fraction = 0.01f;
    config.morph_kernel = 1;

    DyGeoFusion::DynamicMaskRefiner refiner(config, torch::kCPU);
    const cv::Mat rgb = cv::Mat::zeros(9, 9, CV_8UC3);
    const cv::Mat background_depth(9, 9, CV_32FC1, cv::Scalar(2.0f));
    refiner.compute(
        rgb, background_depth, Sophus::SE3f(), nullptr, 0.0, true);

    cv::Mat observed_depth = background_depth.clone();
    observed_depth(cv::Rect(5, 3, 3, 3)).setTo(1.0f);
    const cv::Mat static_mask = refiner.compute(
        rgb, observed_depth, Sophus::SE3f(),
        &background_depth, 1.0, true);

    EXPECT_TRUE(refiner.temporalRefinementApplied());
    EXPECT_EQ(static_mask.at<uchar>(4, 2), 1);
    EXPECT_EQ(static_mask.at<uchar>(4, 6), 0);
    EXPECT_GT(refiner.temporalRecoveredStaticPixels(), 0);
    EXPECT_GT(refiner.temporalAddedDynamicPixels(), 0);

    std::filesystem::remove_all(mask_dir);
}

TEST(DynamicMaskRefinerTest, TemporalRecoveryOnlyNeverAddsExclusions) {
    const std::filesystem::path mask_dir =
        std::filesystem::temp_directory_path() /
        "dynags_temporal_recovery_only_masks";
    std::filesystem::create_directories(mask_dir);

    cv::Mat first_mask = cv::Mat::zeros(9, 9, CV_8UC1);
    cv::Mat second_mask = cv::Mat::zeros(9, 9, CV_8UC1);
    second_mask(cv::Rect(1, 3, 3, 3)).setTo(255);
    ASSERT_TRUE(cv::imwrite(
        (mask_dir / "00000000.png").string(), first_mask));
    ASSERT_TRUE(cv::imwrite(
        (mask_dir / "00000001.png").string(), second_mask));

    DyGeoFusion::MaskConfig config;
    config.use_yolo = false;
    config.use_external_mask = true;
    config.external_mask_dir = mask_dir.string();
    config.use_flow = false;
    config.use_temporal_background_refinement = true;
    config.temporal_recovery_only = true;
    config.temporal_background_warmup_frames = 0;
    config.temporal_render_weight = 0.6f;
    config.temporal_observation_weight = 0.2f;
    config.temporal_depth_threshold = 0.1f;
    config.temporal_neighbor_radius = 1;
    config.temporal_min_consistent_neighbors = 5;
    config.temporal_min_valid_fraction = 0.01f;
    config.morph_kernel = 1;

    DyGeoFusion::DynamicMaskRefiner refiner(config, torch::kCPU);
    const cv::Mat rgb = cv::Mat::zeros(9, 9, CV_8UC3);
    const cv::Mat background_depth(9, 9, CV_32FC1, cv::Scalar(2.0f));
    refiner.compute(
        rgb, background_depth, Sophus::SE3f(), nullptr, 0.0, true);

    cv::Mat observed_depth = background_depth.clone();
    observed_depth(cv::Rect(5, 3, 3, 3)).setTo(1.0f);
    const cv::Mat static_mask = refiner.compute(
        rgb, observed_depth, Sophus::SE3f(),
        &background_depth, 1.0, true);

    EXPECT_TRUE(refiner.temporalRefinementApplied());
    EXPECT_EQ(static_mask.at<uchar>(4, 2), 1);
    EXPECT_EQ(static_mask.at<uchar>(4, 6), 1);
    EXPECT_GT(refiner.temporalRecoveredStaticPixels(), 0);
    EXPECT_EQ(refiner.temporalAddedDynamicPixels(), 0);
    EXPECT_EQ(refiner.getRawDynamicMask().at<uchar>(4, 2), 1);

    std::filesystem::remove_all(mask_dir);
}

}  // namespace
