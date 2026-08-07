#include <gtest/gtest.h>

#include "Motion3D/BackgroundSupportRouter.h"

#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

#include <algorithm>
#include <cmath>

namespace {

cv::Mat texturedImage(const cv::Size& size)
{
    cv::Mat image(size, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uchar>(y, x) = static_cast<uchar>(
                (x * 17 + y * 31 + ((x / 3 + y / 5) % 2) * 70) % 255);
        }
    }
    return image;
}

cv::Mat curvedDepth(const cv::Size& size)
{
    cv::Mat depth(size, CV_32FC1);
    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            depth.at<float>(y, x) =
                1.2f +
                0.08f * std::sin(0.055f * static_cast<float>(x)) +
                0.06f * std::cos(0.047f * static_cast<float>(y)) +
                0.03f * std::sin(
                    0.031f * static_cast<float>(x + y));
        }
    }
    return depth;
}

cv::Mat translateDepth(
    const cv::Mat& model_depth,
    const Motion3D::GaussianBackgroundPoseRefiner::Intrinsics& intrinsics,
    const Eigen::Vector3f& translation)
{
    cv::Mat current(
        model_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    for (int y = 0; y < model_depth.rows; ++y) {
        for (int x = 0; x < model_depth.cols; ++x) {
            const float depth = model_depth.at<float>(y, x);
            const Eigen::Vector3f point(
                (static_cast<float>(x) - intrinsics.cx) *
                    depth / intrinsics.fx,
                (static_cast<float>(y) - intrinsics.cy) *
                    depth / intrinsics.fy,
                depth);
            const Eigen::Vector3f transformed = point + translation;
            if (transformed.z() <= 0.0f) {
                continue;
            }
            const int projected_x = static_cast<int>(std::lround(
                intrinsics.fx * transformed.x() / transformed.z() +
                intrinsics.cx));
            const int projected_y = static_cast<int>(std::lround(
                intrinsics.fy * transformed.y() / transformed.z() +
                intrinsics.cy));
            if (projected_x < 0 || projected_y < 0 ||
                projected_x >= current.cols ||
                projected_y >= current.rows) {
                continue;
            }
            float& value = current.at<float>(projected_y, projected_x);
            if (value == 0.0f || transformed.z() < value) {
                value = transformed.z();
            }
        }
    }
    cv::Mat filled;
    cv::dilate(current, filled, cv::Mat());
    filled.copyTo(current, current == 0.0f);
    return current;
}

}  // namespace

TEST(BackgroundSupportRouterTest, SelectsOnlyStaticRingSupport)
{
    Motion3D::BackgroundSupportRouter::Config config;
    config.ring_radius = 2;
    config.sample_stride = 1;
    config.min_support_points = 10;
    config.max_support_points = 200;
    config.min_gradient = 0.01f;
    Motion3D::BackgroundSupportRouter router(config);

    const cv::Size size(80, 60);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat rendered(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(30, 20, 20, 18)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);
    const cv::Mat dynamic_before = dynamic.clone();
    const cv::Mat static_before = static_mask.clone();

    const auto result = router.select(
        image, depth, dynamic, static_mask, rendered);
    ASSERT_TRUE(result.valid);
    EXPECT_GT(result.ring_pixels, 0);
    EXPECT_EQ(result.selected_points, result.candidate_points);
    EXPECT_EQ(cv::countNonZero(result.support_mask & dynamic), 0);
    EXPECT_GT(cv::countNonZero(result.support_mask), 0);
    cv::Mat diff;
    cv::absdiff(dynamic, dynamic_before, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
    cv::absdiff(static_mask, static_before, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(BackgroundSupportRouterTest, RejectsDepthAndFlowGuardFailures)
{
    Motion3D::BackgroundSupportRouter::Config config;
    config.ring_radius = 2;
    config.sample_stride = 1;
    config.min_support_points = 1;
    config.max_support_points = 200;
    config.min_gradient = 0.0f;
    config.max_depth_residual = 0.01f;
    config.depth_residual_scale = 0.0f;
    config.flow_guard_radius = 1;
    Motion3D::BackgroundSupportRouter router(config);

    const cv::Size size(50, 40);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat rendered(size, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(20, 15, 10, 10)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    cv::Mat guard = cv::Mat::zeros(size, CV_8UC1);
    guard.at<uchar>(15, 19) = 255;
    const auto rejected = router.select(
        image, depth, dynamic, static_mask, rendered, guard);
    EXPECT_FALSE(rejected.valid);
    EXPECT_EQ(rejected.selected_points, 0);

    const cv::Mat consistent(size, CV_32FC1, cv::Scalar(1.0f));
    const auto accepted = router.select(
        image, depth, dynamic, static_mask, consistent);
    EXPECT_TRUE(accepted.valid);
    EXPECT_GT(accepted.selected_points, 0);
}

TEST(BackgroundSupportRouterTest, RequiresRenderedDepthByDefault)
{
    Motion3D::BackgroundSupportRouter router;
    const cv::Size size(20, 20);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(7, 7, 6, 6)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto result = router.select(
        image, depth, dynamic, static_mask);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.selected_points, 0);
}

TEST(BackgroundSupportRouterTest, CapsSupportDeterministically)
{
    Motion3D::BackgroundSupportRouter::Config config;
    config.ring_radius = 3;
    config.sample_stride = 1;
    config.min_support_points = 1;
    config.max_support_points = 7;
    config.min_gradient = 0.0f;
    Motion3D::BackgroundSupportRouter router(config);

    const cv::Size size(60, 50);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat rendered(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(20, 15, 20, 15)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto first = router.select(
        image, depth, dynamic, static_mask, rendered);
    const auto second = router.select(
        image, depth, dynamic, static_mask, rendered);
    ASSERT_TRUE(first.valid);
    ASSERT_TRUE(second.valid);
    ASSERT_EQ(first.selected_points, 7);
    ASSERT_EQ(second.selected_points, 7);
    ASSERT_EQ(first.points.size(), second.points.size());
    for (std::size_t index = 0; index < first.points.size(); ++index) {
        EXPECT_EQ(first.points[index].pixel, second.points[index].pixel);
        EXPECT_FLOAT_EQ(first.points[index].score, second.points[index].score);
    }
}

TEST(BackgroundSupportPoseRefinerTest, RejectsInvalidIntrinsics)
{
    Motion3D::BackgroundSupportPoseRefiner::Config config;
    Motion3D::BackgroundSupportPoseRefiner::Intrinsics intrinsics;
    intrinsics.fx = 0.0f;
    intrinsics.fy = 50.0f;
    intrinsics.cx = 25.0f;
    intrinsics.cy = 20.0f;

    EXPECT_THROW(
        Motion3D::BackgroundSupportPoseRefiner(config, intrinsics),
        std::invalid_argument);
}

TEST(BackgroundSupportPoseRefinerTest, FailsClosedWithoutRenderedDepth)
{
    Motion3D::BackgroundSupportPoseRefiner::Config config;
    config.router.ring_radius = 2;
    config.router.sample_stride = 1;
    config.router.min_support_points = 1;
    config.router.max_support_points = 100;
    config.router.min_gradient = 0.0f;
    config.router.require_rendered_depth = true;
    Motion3D::BackgroundSupportPoseRefiner::Intrinsics intrinsics;
    intrinsics.fx = 50.0f;
    intrinsics.fy = 50.0f;
    intrinsics.cx = 25.0f;
    intrinsics.cy = 20.0f;
    Motion3D::BackgroundSupportPoseRefiner refiner(config, intrinsics);

    const cv::Size size(50, 40);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(20, 15, 10, 10)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto result = refiner.refine(
        image, image, depth, depth, static_mask, static_mask,
        dynamic, cv::Mat(), cv::Mat(), Sophus::SE3f());

    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.improved);
    EXPECT_EQ(result.routed_support, 0);
}

TEST(BackgroundSupportRouterTest, PoseRefinerRejectsExactParent)
{
    Motion3D::BackgroundSupportPoseRefiner::Config config;
    config.router.ring_radius = 2;
    config.router.sample_stride = 1;
    config.router.min_support_points = 10;
    config.router.max_support_points = 200;
    config.router.min_gradient = 0.0f;
    config.min_rmse_improvement = 0.01f;
    config.translation_only = true;
    Motion3D::BackgroundSupportPoseRefiner refiner(
        config,
        Motion3D::BackgroundSupportPoseRefiner::Intrinsics{
            80.0f, 80.0f, 40.0f, 30.0f});

    const cv::Size size(80, 60);
    const cv::Mat image = texturedImage(size);
    const cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat rendered(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(30, 20, 20, 18)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto result = refiner.refine(
        image, image, depth, depth, static_mask, static_mask,
        dynamic, rendered, cv::Mat(), Sophus::SE3f());
    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.improved);
    EXPECT_GT(result.geometric_support, 0);
    EXPECT_NEAR(result.initial_rmse, 0.0f, 1e-5f);
    EXPECT_NEAR(result.final_rmse, 0.0f, 1e-5f);
    EXPECT_NEAR(result.rotation_correction, 0.0f, 1e-6f);
}

TEST(
    GaussianBackgroundPoseRefinerTest,
    RecoversTranslationWithDisjointValidation)
{
    const cv::Size size(160, 120);
    const Motion3D::GaussianBackgroundPoseRefiner::Intrinsics intrinsics{
        140.0f, 140.0f, 79.5f, 59.5f};
    Motion3D::GaussianBackgroundPoseRefiner::Config config;
    config.sample_stride = 2;
    config.split_tile_size = 12;
    config.dynamic_dilation_radius = 3;
    config.min_proposal_support = 300;
    config.min_validation_support = 300;
    config.minimum_observable_rank = 2;
    config.max_correspondence_distance = 0.08f;
    config.min_proposal_relative_improvement = 0.001f;
    config.min_validation_relative_improvement = 0.001f;
    config.max_translation_correction = 0.02f;
    Motion3D::GaussianBackgroundPoseRefiner refiner(config, intrinsics);

    const cv::Mat model_depth = curvedDepth(size);
    const Eigen::Vector3f expected_translation(0.006f, -0.004f, 0.005f);
    const cv::Mat current_depth = translateDepth(
        model_depth, intrinsics, expected_translation);
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(65, 45, 30, 25)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto result = refiner.refine(
        current_depth, static_mask, dynamic, model_depth,
        Sophus::SE3f());

    ASSERT_TRUE(result.candidate_available);
    ASSERT_TRUE(result.proposal_improved);
    ASSERT_TRUE(result.validation_valid);
    ASSERT_TRUE(result.validation_prefers_candidate);
    ASSERT_TRUE(result.gate_pass);
    EXPECT_EQ(result.proposal_validation_overlap, 0);
    EXPECT_GE(result.observable_rank, 2);
    EXPECT_GT(result.proposal_support, config.min_proposal_support);
    EXPECT_GT(
        result.validation_common_support,
        config.min_validation_support);
    EXPECT_LT(
        result.proposal_candidate_loss,
        result.proposal_parent_loss);
    EXPECT_LT(
        result.validation_candidate_loss,
        result.validation_parent_loss);
    EXPECT_NEAR(result.rotation_correction, 0.0f, 1e-6f);
    EXPECT_LE(
        result.translation_correction,
        config.max_translation_correction + 1e-6f);
    EXPECT_LT(
        (result.refined_tcw.translation() - expected_translation).norm(),
        0.006f);
}

TEST(
    GaussianBackgroundPoseRefinerTest,
    RejectsExactParentAndDegeneratePlane)
{
    const cv::Size size(120, 90);
    const Motion3D::GaussianBackgroundPoseRefiner::Intrinsics intrinsics{
        100.0f, 100.0f, 59.5f, 44.5f};
    Motion3D::GaussianBackgroundPoseRefiner::Config config;
    config.sample_stride = 2;
    config.split_tile_size = 10;
    config.dynamic_dilation_radius = 2;
    config.min_proposal_support = 100;
    config.min_validation_support = 100;
    config.minimum_observable_rank = 2;
    config.min_proposal_relative_improvement = 0.001f;
    config.min_validation_relative_improvement = 0.001f;
    Motion3D::GaussianBackgroundPoseRefiner refiner(config, intrinsics);

    const cv::Mat plane(size, CV_32FC1, cv::Scalar(1.2f));
    cv::Mat dynamic = cv::Mat::zeros(size, CV_8UC1);
    dynamic(cv::Rect(50, 35, 20, 20)).setTo(255);
    cv::Mat static_mask = cv::Mat::ones(size, CV_8UC1) * 255;
    static_mask.setTo(0, dynamic);

    const auto result = refiner.refine(
        plane, static_mask, dynamic, plane, Sophus::SE3f());
    EXPECT_FALSE(result.candidate_available);
    EXPECT_FALSE(result.gate_pass);
    EXPECT_LT(result.observable_rank, config.minimum_observable_rank);
    EXPECT_EQ(result.proposal_validation_overlap, 0);
}
