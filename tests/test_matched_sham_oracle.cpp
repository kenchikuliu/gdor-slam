#include <gtest/gtest.h>

#include "Motion3D/MatchedShamOracle.h"

#include <limits>

namespace {

Sophus::SE3f tcwFromCameraCenter(float x, float y, float z) {
    return Sophus::SE3f(
        Sophus::SO3f(), Eigen::Vector3f(-x, -y, -z));
}

}  // namespace

TEST(MatchedShamOracleTest, SelectsLowerCameraCenterTranslationError)
{
    const Sophus::SE3f ground_truth = tcwFromCameraCenter(1.0f, 0.0f, 0.0f);
    const auto result = Motion3D::evaluateMatchedShamOracle(
        tcwFromCameraCenter(0.8f, 0.0f, 0.0f),
        tcwFromCameraCenter(0.95f, 0.0f, 0.0f),
        ground_truth);

    ASSERT_TRUE(result.valid);
    EXPECT_TRUE(result.prefers_dynamic);
    EXPECT_NEAR(result.static_translation_error_m, 0.2f, 1e-6f);
    EXPECT_NEAR(result.dynamic_translation_error_m, 0.05f, 1e-6f);
}

TEST(MatchedShamOracleTest, TiesAndSubThresholdChangesChooseStatic)
{
    const Sophus::SE3f ground_truth = tcwFromCameraCenter(0.0f, 0.0f, 0.0f);
    const auto tie = Motion3D::evaluateMatchedShamOracle(
        tcwFromCameraCenter(0.1f, 0.0f, 0.0f),
        tcwFromCameraCenter(-0.1f, 0.0f, 0.0f),
        ground_truth);
    ASSERT_TRUE(tie.valid);
    EXPECT_FALSE(tie.prefers_dynamic);

    const auto sub_threshold = Motion3D::evaluateMatchedShamOracle(
        tcwFromCameraCenter(0.1f, 0.0f, 0.0f),
        tcwFromCameraCenter(0.0999995f, 0.0f, 0.0f),
        ground_truth, 1e-6f);
    ASSERT_TRUE(sub_threshold.valid);
    EXPECT_FALSE(sub_threshold.prefers_dynamic);
}

TEST(MatchedShamOracleTest, RejectsInvalidInputs)
{
    Sophus::SE3f invalid;
    invalid.translation().x() = std::numeric_limits<float>::quiet_NaN();
    const auto result = Motion3D::evaluateMatchedShamOracle(
        Sophus::SE3f(), Sophus::SE3f(), invalid);
    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.prefers_dynamic);
}
