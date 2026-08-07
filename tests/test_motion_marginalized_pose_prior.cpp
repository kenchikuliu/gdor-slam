#include <gtest/gtest.h>

#include "Motion3D/MotionMarginalizedPosePrior.h"
#include "Utils/ImageColor.h"
#include "Utils/ImageGeometry.h"

#include <Eigen/Cholesky>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <set>
#include <vector>

using MotionPrior = Motion3D::MotionMarginalizedPosePrior;

namespace {

std::vector<Eigen::Vector3f> makeNonDegeneratePoints()
{
    std::vector<Eigen::Vector3f> points;
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 4; ++x) {
                points.emplace_back(
                    0.08f * x - 0.12f,
                    0.07f * y - 0.07f,
                    0.9f + 0.08f * z + 0.01f * x * y);
            }
        }
    }
    return points;
}

Eigen::Vector3f residualForPoint(
    const Eigen::Vector3f& source, const Eigen::Vector3f& target,
    const Sophus::SE3f& camera, const Sophus::SE3f& object)
{
    return camera * (object * source) - target;
}

struct NumericNormalEquations {
    MotionPrior::Matrix6f Hcc = MotionPrior::Matrix6f::Zero();
    MotionPrior::Matrix6f Hco = MotionPrior::Matrix6f::Zero();
    MotionPrior::Matrix6f Hoo = MotionPrior::Matrix6f::Zero();
    MotionPrior::Vector6f bc = MotionPrior::Vector6f::Zero();
    MotionPrior::Vector6f bo = MotionPrior::Vector6f::Zero();
};

NumericNormalEquations finiteDifferenceNormalEquations(
    const std::vector<Eigen::Vector3f>& source,
    const std::vector<Eigen::Vector3f>& target,
    const Sophus::SE3f& camera,
    const Sophus::SE3f& object,
    float variance)
{
    NumericNormalEquations result;
    constexpr float epsilon = 2e-4f;
    for (std::size_t point_index = 0; point_index < source.size();
         ++point_index) {
        Eigen::Matrix<float, 3, 6> Jc;
        Eigen::Matrix<float, 3, 6> Jo;
        for (int axis = 0; axis < 6; ++axis) {
            MotionPrior::Vector6f perturbation =
                MotionPrior::Vector6f::Zero();
            perturbation[axis] = epsilon;
            Jc.col(axis) =
                (residualForPoint(
                     source[point_index], target[point_index],
                     Sophus::SE3f::exp(perturbation) * camera, object) -
                 residualForPoint(
                     source[point_index], target[point_index],
                     Sophus::SE3f::exp(-perturbation) * camera, object)) /
                (2.0f * epsilon);
            Jo.col(axis) =
                (residualForPoint(
                     source[point_index], target[point_index], camera,
                     Sophus::SE3f::exp(perturbation) * object) -
                 residualForPoint(
                     source[point_index], target[point_index], camera,
                     Sophus::SE3f::exp(-perturbation) * object)) /
                (2.0f * epsilon);
        }
        const Eigen::Vector3f residual = residualForPoint(
            source[point_index], target[point_index], camera, object);
        result.Hcc += Jc.transpose() * Jc / variance;
        result.Hco += Jc.transpose() * Jo / variance;
        result.Hoo += Jo.transpose() * Jo / variance;
        result.bc += Jc.transpose() * residual / variance;
        result.bo += Jo.transpose() * residual / variance;
    }
    return result;
}

cv::Mat texturedFrame(
    const cv::Mat& texture, const cv::Rect& destination,
    const cv::Size& image_size)
{
    cv::Mat image(image_size, CV_8UC1, cv::Scalar(0));
    texture.copyTo(image(destination));
    return image;
}

MotionPrior::Result runMotionPriorWithFinalDt(double final_timestamp)
{
    MotionPrior::Config config;
    config.max_features = 180;
    config.min_features_per_object = 8;
    config.min_component_area = 100;
    config.min_track_age = 2;
    config.ransac_threshold = 0.03f;
    config.min_inlier_ratio = 0.7f;
    config.min_information = 1e-5f;
    config.max_information_eigenvalue = 1e8f;
    config.use_depth_foreground_filter = false;
    config.min_object_translation_speed = 0.0f;
    config.min_object_rotation_speed = 0.0f;
    config.seed = 19;
    MotionPrior prior(
        config, MotionPrior::CameraIntrinsics{100.0f, 100.0f, 80.0f, 60.0f});

    cv::Mat texture(60, 60, CV_8UC1);
    cv::RNG rng(7);
    rng.fill(texture, cv::RNG::UNIFORM, 0, 255);
    const cv::Size size(160, 120);
    cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));
    MotionPrior::Result result;
    const std::array<double, 3> timestamps = {0.0, 0.1, final_timestamp};
    for (int frame = 0; frame < 3; ++frame) {
        const cv::Rect box(30 + 2 * frame, 30, 60, 60);
        const cv::Mat image = texturedFrame(texture, box, size);
        cv::Mat mask(size, CV_8UC1, cv::Scalar(0));
        mask(box).setTo(255);
        result = prior.prepare(image, depth, mask, timestamps[frame]);
        prior.commitCameraPose(Sophus::SE3f(), true);
    }
    return result;
}

}  // namespace

TEST(MotionMarginalizedPosePriorTest, RigidAlignmentRejectsOutliers)
{
    std::vector<Eigen::Vector3f> source = makeNonDegeneratePoints();
    const Sophus::SE3f expected(
        Sophus::SO3f::exp(Eigen::Vector3f(0.02f, -0.03f, 0.04f)),
        Eigen::Vector3f(0.05f, -0.02f, 0.01f));
    std::vector<Eigen::Vector3f> target;
    target.reserve(source.size());
    for (const Eigen::Vector3f& point : source) {
        target.push_back(expected * point);
    }
    target[3] += Eigen::Vector3f(0.5f, 0.5f, 0.5f);
    target[17] += Eigen::Vector3f(-0.4f, 0.3f, 0.2f);

    std::mt19937 rng(7);
    Sophus::SE3f estimated;
    std::vector<unsigned char> inliers;
    float rmse = 0.0f;
    ASSERT_TRUE(MotionPrior::estimateRigidTransform(
        source, target, 0.02f, 200, rng, estimated, inliers, rmse));
    EXPECT_GE(std::count(inliers.begin(), inliers.end(), 1), 22);
    EXPECT_LT(
        (estimated * expected.inverse()).translation().norm(), 1e-4f);
    EXPECT_LT(
        (estimated * expected.inverse()).so3().log().norm(), 1e-4f);
    EXPECT_LT(rmse, 1e-4f);
}

TEST(MotionMarginalizedPosePriorTest, JointJacobiansMatchFiniteDifferences)
{
    const std::vector<Eigen::Vector3f> source = makeNonDegeneratePoints();
    const Sophus::SE3f camera = Sophus::SE3f::exp(
        (MotionPrior::Vector6f() <<
             0.03f, -0.02f, 0.01f, 0.02f, -0.01f, 0.03f)
            .finished());
    const Sophus::SE3f object = Sophus::SE3f::exp(
        (MotionPrior::Vector6f() <<
             -0.01f, 0.04f, 0.02f, -0.03f, 0.02f, 0.01f)
            .finished());
    std::vector<Eigen::Vector3f> target;
    for (const Eigen::Vector3f& point : source) {
        target.push_back(
            camera * (object * point) +
            Eigen::Vector3f(0.002f, -0.001f, 0.003f));
    }
    const float variance = 4e-4f;
    const std::vector<unsigned char> inliers(source.size(), 1);
    const auto analytic = MotionPrior::linearizeAndMarginalize(
        source, target, inliers, camera, object, variance,
        MotionPrior::Matrix6f::Identity() * 10.0f);
    const NumericNormalEquations numeric =
        finiteDifferenceNormalEquations(
            source, target, camera, object, variance);

    EXPECT_LT((analytic.Hcc - numeric.Hcc).norm() / numeric.Hcc.norm(), 2e-3f);
    EXPECT_LT((analytic.Hco - numeric.Hco).norm() / numeric.Hco.norm(), 2e-3f);
    EXPECT_LT((analytic.Hoo - numeric.Hoo).norm() / numeric.Hoo.norm(), 2e-3f);
    EXPECT_LT((analytic.bc - numeric.bc).norm() / numeric.bc.norm(), 3e-3f);
    EXPECT_LT((analytic.bo - numeric.bo).norm() / numeric.bo.norm(), 3e-3f);
}

TEST(MotionMarginalizedPosePriorTest, SchurMatchesFullJointSolve)
{
    const std::vector<Eigen::Vector3f> source = makeNonDegeneratePoints();
    const Sophus::SE3f true_camera = Sophus::SE3f::exp(
        (MotionPrior::Vector6f() <<
             0.04f, -0.01f, 0.02f, 0.01f, 0.02f, -0.03f)
            .finished());
    const Sophus::SE3f true_object = Sophus::SE3f::exp(
        (MotionPrior::Vector6f() <<
             -0.02f, 0.03f, 0.01f, -0.01f, 0.03f, 0.02f)
            .finished());
    std::vector<Eigen::Vector3f> target;
    for (const Eigen::Vector3f& point : source) {
        target.push_back(true_camera * (true_object * point));
    }

    MotionPrior::Vector6f camera_offset;
    camera_offset << 0.004f, -0.003f, 0.002f, 0.002f, -0.001f, 0.003f;
    MotionPrior::Vector6f object_offset;
    object_offset << -0.003f, 0.002f, -0.002f, -0.002f, 0.003f, 0.001f;
    const Sophus::SE3f camera =
        Sophus::SE3f::exp(camera_offset) * true_camera;
    const Sophus::SE3f object =
        Sophus::SE3f::exp(object_offset) * true_object;
    const MotionPrior::Matrix6f object_prior =
        MotionPrior::Matrix6f::Identity() * 40.0f;
    const std::vector<unsigned char> inliers(source.size(), 1);
    const auto schur = MotionPrior::linearizeAndMarginalize(
        source, target, inliers, camera, object, 2.5e-4f, object_prior);
    ASSERT_TRUE(schur.valid);

    Eigen::Matrix<float, 12, 12> joint =
        Eigen::Matrix<float, 12, 12>::Zero();
    joint.block<6, 6>(0, 0) = schur.Hcc;
    joint.block<6, 6>(0, 6) = schur.Hco;
    joint.block<6, 6>(6, 0) = schur.Hco.transpose();
    joint.block<6, 6>(6, 6) = schur.Hoo + object_prior;
    Eigen::Matrix<float, 12, 1> gradient;
    gradient << schur.bc, schur.bo;
    const Eigen::Matrix<float, 12, 1> full_increment =
        -joint.ldlt().solve(gradient);

    EXPECT_LT(
        (schur.camera_increment - full_increment.head<6>()).norm(), 2e-5f);

    const Sophus::SE3f corrected_camera =
        Sophus::SE3f::exp(schur.camera_increment) * camera;
    const auto corrected = MotionPrior::linearizeAndMarginalize(
        source, target, inliers, corrected_camera, object, 2.5e-4f,
        object_prior);
    ASSERT_TRUE(corrected.valid);
    EXPECT_LT(
        corrected.camera_gradient.norm(),
        schur.camera_gradient.norm());
}

TEST(MotionMarginalizedPosePriorTest, CovarianceTransportMatchesConjugation)
{
    MotionPrior::Matrix6f covariance = MotionPrior::Matrix6f::Zero();
    covariance.diagonal() <<
        0.01f, 0.02f, 0.03f, 0.004f, 0.005f, 0.006f;
    const Sophus::SE3f transform(
        Sophus::SO3f::exp(Eigen::Vector3f(0.2f, -0.1f, 0.15f)),
        Eigen::Vector3f(0.4f, -0.2f, 0.1f));
    MotionPrior::Matrix6f numeric_adjoint;
    constexpr float epsilon = 1e-4f;
    for (int axis = 0; axis < 6; ++axis) {
        MotionPrior::Vector6f tangent = MotionPrior::Vector6f::Zero();
        tangent[axis] = epsilon;
        numeric_adjoint.col(axis) =
            (transform * Sophus::SE3f::exp(tangent) * transform.inverse())
                .log() /
            epsilon;
    }
    const MotionPrior::Matrix6f expected =
        numeric_adjoint * covariance * numeric_adjoint.transpose();
    const MotionPrior::Matrix6f actual =
        MotionPrior::transformCovariance(covariance, transform);
    EXPECT_LT((actual - expected).norm(), 2e-5f);
}

TEST(MotionMarginalizedPosePriorTest, ExpAndLogJacobiansAreLocallyConsistent)
{
    MotionPrior::Vector6f tangent;
    tangent << 0.08f, -0.04f, 0.03f, 0.12f, -0.07f, 0.05f;
    const MotionPrior::Matrix6f exp_jacobian =
        MotionPrior::leftExpJacobian(tangent);
    const MotionPrior::Matrix6f log_jacobian =
        MotionPrior::leftLogJacobian(Sophus::SE3f::exp(tangent));
    EXPECT_LT(
        (log_jacobian * exp_jacobian -
         MotionPrior::Matrix6f::Identity()).norm(),
        5e-3f);
}

TEST(MotionMarginalizedPosePriorTest,
     CandidateConsensusRejectsContradictoryObjects)
{
    const MotionPrior::Matrix6f information =
        MotionPrior::Matrix6f::Identity() * 1000.0f;
    std::vector<MotionPrior::CameraCandidate> candidates{
        {Sophus::SE3f(), information},
        {Sophus::SE3f(
             Sophus::SO3f(),
             Eigen::Vector3f(0.01f, 0.0f, 0.0f)),
         information},
        {Sophus::SE3f(
             Sophus::SO3f(),
             Eigen::Vector3f(-0.25f, 0.0f, 0.0f)),
         information},
    };

    EXPECT_LT(
        MotionPrior::candidateInnovationMahalanobis(
            candidates[0], candidates[1]),
        16.812f);
    EXPECT_GT(
        MotionPrior::candidateInnovationMahalanobis(
            candidates[0], candidates[2]),
        16.812f);
    const std::vector<std::size_t> selected =
        MotionPrior::selectConsistentCandidates(candidates, 16.812f);
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], 0u);
    EXPECT_EQ(selected[1], 1u);
}

TEST(MotionMarginalizedPosePriorTest, SplitAndMergeKeepObjectIdsUnique)
{
    MotionPrior::Config config;
    config.max_features = 240;
    config.min_features_per_object = 5;
    config.min_component_area = 80;
    config.min_track_age = 2;
    config.ransac_threshold = 0.03f;
    config.min_inlier_ratio = 0.5f;
    config.use_depth_foreground_filter = false;
    MotionPrior prior(
        config, MotionPrior::CameraIntrinsics{120.0f, 120.0f, 70.0f, 55.0f});

    const cv::Size size(140, 110);
    cv::Mat texture(80, 60, CV_8UC1);
    cv::RNG rng(17);
    rng.fill(texture, cv::RNG::UNIFORM, 0, 255);
    cv::Mat image(size, CV_8UC1, cv::Scalar(0));
    texture.copyTo(image(cv::Rect(30, 15, 60, 80)));
    cv::Mat depth(size, CV_32FC1, cv::Scalar(1.0f));

    cv::Mat merged(size, CV_8UC1, cv::Scalar(0));
    merged(cv::Rect(30, 15, 60, 80)).setTo(255);
    prior.prepare(image, depth, merged, 0.0);
    prior.commitCameraPose(Sophus::SE3f(), true);
    ASSERT_EQ(prior.activeObjectIds().size(), 1u);
    const int original_id = prior.activeObjectIds().front();

    cv::Mat split(size, CV_8UC1, cv::Scalar(0));
    split(cv::Rect(30, 15, 27, 80)).setTo(255);
    split(cv::Rect(63, 15, 27, 80)).setTo(255);
    prior.prepare(image, depth, split, 0.1);
    prior.commitCameraPose(Sophus::SE3f(), true);
    const std::vector<int> split_ids = prior.activeObjectIds();
    ASSERT_EQ(split_ids.size(), 2u);
    EXPECT_EQ(std::set<int>(split_ids.begin(), split_ids.end()).size(), 2u);
    EXPECT_EQ(std::count(split_ids.begin(), split_ids.end(), original_id), 1);

    prior.prepare(image, depth, merged, 0.2);
    prior.commitCameraPose(Sophus::SE3f(), true);
    const std::vector<int> merged_ids = prior.activeObjectIds();
    ASSERT_EQ(merged_ids.size(), 1u);
    EXPECT_EQ(std::set<int>(merged_ids.begin(), merged_ids.end()).size(), 1u);
}

TEST(MotionMarginalizedPosePriorTest, LongerPredictionIntervalWeakensInformation)
{
    const MotionPrior::Result short_interval =
        runMotionPriorWithFinalDt(0.2);
    const MotionPrior::Result long_interval =
        runMotionPriorWithFinalDt(1.1);
    ASSERT_TRUE(short_interval.valid);
    EXPECT_TRUE(
        !long_interval.valid ||
        short_interval.information_score >
            long_interval.information_score);
}

TEST(MotionMarginalizedPosePriorTest, ReusesMotionButDoesNotModifyMasks)
{
    MotionPrior::Config config;
    config.max_features = 120;
    config.min_features_per_object = 8;
    config.min_component_area = 100;
    config.min_track_age = 2;
    config.ransac_threshold = 0.03f;
    config.min_inlier_ratio = 0.7f;
    config.min_information = 0.1f;
    config.use_depth_foreground_filter = false;
    config.seed = 11;
    MotionPrior prior(
        config, MotionPrior::CameraIntrinsics{100.0f, 100.0f, 60.0f, 50.0f});

    cv::Mat texture(60, 60, CV_8UC1);
    cv::RNG rng(5);
    rng.fill(texture, cv::RNG::UNIFORM, 0, 255);
    cv::Mat depth(100, 120, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat static_mask(100, 120, CV_8UC1, cv::Scalar(1));

    MotionPrior::Result result;
    for (int frame = 0; frame < 3; ++frame) {
        cv::Mat gray(100, 120, CV_8UC1, cv::Scalar(0));
        cv::Mat dynamic_mask(100, 120, CV_8UC1, cv::Scalar(0));
        const cv::Rect object_box(20 + 2 * frame, 20, 60, 60);
        texture.copyTo(gray(object_box));
        dynamic_mask(object_box).setTo(1);
        const cv::Mat dynamic_before = dynamic_mask.clone();
        const cv::Mat static_before = static_mask.clone();

        result = prior.prepare(gray, depth, dynamic_mask, 0.1 * frame);
        prior.commitCameraPose(Sophus::SE3f(), true);

        EXPECT_EQ(cv::countNonZero(dynamic_mask != dynamic_before), 0);
        EXPECT_EQ(cv::countNonZero(static_mask != static_before), 0);
    }

    EXPECT_TRUE(result.valid);
    EXPECT_GE(result.objects_used, 1);
    EXPECT_LT(result.relative_pose.translation().norm(), 0.01f);
    EXPECT_LT(result.relative_pose.so3().log().norm(), 0.01f);
}

TEST(MotionMarginalizedPosePriorTest, StaticMaskProducesNoPrior)
{
    MotionPrior::Config config;
    config.min_component_area = 10;
    MotionPrior prior(
        config, MotionPrior::CameraIntrinsics{100.0f, 100.0f, 20.0f, 20.0f});
    cv::Mat image(40, 40, CV_8UC1, cv::Scalar(127));
    cv::Mat depth(40, 40, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat no_dynamic_pixels(40, 40, CV_8UC1, cv::Scalar(0));

    EXPECT_FALSE(
        prior.prepare(image, depth, no_dynamic_pixels, 0.0).valid);
    prior.commitCameraPose(Sophus::SE3f(), true);
    EXPECT_FALSE(
        prior.prepare(image, depth, no_dynamic_pixels, 0.1).valid);
}

TEST(ImageColorTest, ConvertsRgbSentinelToOpenCvBgr)
{
    cv::Mat rgb(1, 1, CV_8UC3);
    rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(11, 22, 33);
    const cv::Mat bgr = DyGeoFusion::rgbToOpenCvBgr(rgb);
    EXPECT_EQ(bgr.at<cv::Vec3b>(0, 0), cv::Vec3b(33, 22, 11));
}

TEST(ImageColorTest, ConvertsOpenCvBgrSentinelToRgb)
{
    cv::Mat bgr(1, 1, CV_8UC3);
    bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(33, 22, 11);
    const cv::Mat rgb = DyGeoFusion::openCvBgrToRgb(bgr);
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 0), cv::Vec3b(11, 22, 33));
}

TEST(ImageGeometryTest, ResizesRgbAndDepthToCalibratedCameraSize)
{
    cv::Mat rgb(2, 3, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat depth(2, 2, CV_16UC1);
    depth.at<unsigned short>(0, 0) = 100;
    depth.at<unsigned short>(0, 1) = 200;
    depth.at<unsigned short>(1, 0) = 300;
    depth.at<unsigned short>(1, 1) = 400;

    ASSERT_TRUE(DyGeoFusion::resizeRgbdToCameraSize(
        rgb, depth, cv::Size(4, 4)));
    EXPECT_EQ(rgb.size(), cv::Size(4, 4));
    EXPECT_EQ(depth.size(), cv::Size(4, 4));
    EXPECT_EQ(depth.at<unsigned short>(0, 0), 100);
    EXPECT_EQ(depth.at<unsigned short>(0, 3), 200);
    EXPECT_EQ(depth.at<unsigned short>(3, 0), 300);
    EXPECT_EQ(depth.at<unsigned short>(3, 3), 400);
}
