#include "Motion3D/MatchedShamOracle.h"

#include <cmath>

namespace Motion3D {

MatchedShamOracleResult evaluateMatchedShamOracle(
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw,
    const Sophus::SE3f& ground_truth_tcw,
    float minimum_translation_improvement_m) {
    MatchedShamOracleResult result;
    if (!static_tcw.matrix().allFinite() ||
        !dynamic_tcw.matrix().allFinite() ||
        !ground_truth_tcw.matrix().allFinite() ||
        !std::isfinite(minimum_translation_improvement_m) ||
        minimum_translation_improvement_m < 0.0f) {
        return result;
    }

    const Sophus::SE3f static_twc = static_tcw.inverse();
    const Sophus::SE3f dynamic_twc = dynamic_tcw.inverse();
    const Sophus::SE3f ground_truth_twc = ground_truth_tcw.inverse();
    result.static_translation_error_m =
        (static_twc.translation() - ground_truth_twc.translation()).norm();
    result.dynamic_translation_error_m =
        (dynamic_twc.translation() - ground_truth_twc.translation()).norm();
    result.static_rotation_error_rad =
        (static_twc.so3() * ground_truth_twc.so3().inverse()).log().norm();
    result.dynamic_rotation_error_rad =
        (dynamic_twc.so3() * ground_truth_twc.so3().inverse()).log().norm();
    result.valid =
        std::isfinite(result.static_translation_error_m) &&
        std::isfinite(result.dynamic_translation_error_m) &&
        std::isfinite(result.static_rotation_error_rad) &&
        std::isfinite(result.dynamic_rotation_error_rad);
    result.prefers_dynamic = result.valid &&
        result.static_translation_error_m -
            result.dynamic_translation_error_m >
                minimum_translation_improvement_m;
    return result;
}

}  // namespace Motion3D
