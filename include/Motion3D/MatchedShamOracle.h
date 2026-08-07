#pragma once

#include <sophus/se3.hpp>

namespace Motion3D {

struct MatchedShamOracleResult {
    bool valid = false;
    bool prefers_dynamic = false;
    float static_translation_error_m = -1.0f;
    float dynamic_translation_error_m = -1.0f;
    float static_rotation_error_rad = -1.0f;
    float dynamic_rotation_error_rad = -1.0f;
};

MatchedShamOracleResult evaluateMatchedShamOracle(
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw,
    const Sophus::SE3f& ground_truth_tcw,
    float minimum_translation_improvement_m = 1e-6f);

}  // namespace Motion3D
