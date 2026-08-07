#include "Motion3D/FrozenReplayPacket.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path output;
    std::filesystem::path episode_manifest;
    std::string selection = "would-use";
    std::string prior_subspace = "full";
    std::string intervention_projection = "full";
    std::vector<int> horizons{1, 5, 10};
    int min_inliers = 10;
    float information_scale_multiplier = 1.0f;
    bool posterior_local_map = false;
    int lag_frames = 30;
    float translation_blend = -1.0f;
    float posterior_min_score_improvement = -1.0f;
    float max_static_information_leverage = 0.0f;
    Motion3D::TranslationLeverageMode static_information_leverage_mode =
        Motion3D::TranslationLeverageMode::CapOnly;
    bool gate_policy_override = false;
    bool static_scan_only = false;
    Motion3D::ReplayGatePolicy gate_policy =
        Motion3D::ReplayGatePolicy::LegacyConjunction;
    std::vector<std::string> packet_paths;
};

struct Rollout {
    bool success = false;
    int failure_step = -1;
    int min_matches = std::numeric_limits<int>::max();
    int min_inliers = std::numeric_limits<int>::max();
    int first_matches = -1;
    int first_inliers = -1;
    int final_matches = -1;
    int final_inliers = -1;
    Sophus::SE3f final_tcw;
    bool candidate = false;
    bool posterior_valid = false;
    bool posterior_pass = false;
    int posterior_static_inliers = -1;
    int posterior_dynamic_inliers = -1;
    int posterior_common_support = 0;
    float posterior_static_score = -1.0f;
    float posterior_dynamic_score = -1.0f;
    float posterior_score_improvement = -1.0f;
    float posterior_translation_innovation = -1.0f;
    Motion3D::TranslationLeverageDiagnostics leverage;
    std::string error;
};

enum class RolloutMode {
    Static,
    DynamicPropagated,
    DynamicVelocityNeutral,
};

enum class MethodMode {
    Semantic,
    Schur,
    Lag,
};

enum class PosteriorForkMode {
    Static,
    DynamicVelocityNeutral,
};

struct Episode {
    std::string id;
    std::string checkpoint;
    int horizon = 0;
    bool emitted = false;
};

std::vector<std::string> splitCsvRow(const std::string& row) {
    std::vector<std::string> values;
    std::stringstream stream(row);
    std::string value;
    while (std::getline(stream, value, ',')) {
        values.push_back(value);
    }
    if (!row.empty() && row.back() == ',') {
        values.emplace_back();
    }
    return values;
}

std::string episodeKey(const std::string& checkpoint, int horizon) {
    return checkpoint + "\n" + std::to_string(horizon);
}

bool loadEpisodes(
    const std::filesystem::path& path,
    std::unordered_map<std::string, Episode>& episodes,
    std::string& error) {
    if (path.empty()) {
        return true;
    }
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "Cannot open episode manifest: " + path.string();
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        error = "Episode manifest is empty";
        return false;
    }
    const std::vector<std::string> header = splitCsvRow(line);
    const auto column = [&header](const std::string& name) {
        const auto found = std::find(header.begin(), header.end(), name);
        return found == header.end()
            ? -1 : static_cast<int>(std::distance(header.begin(), found));
    };
    const int idColumn = column("event_id");
    const int checkpointColumn = column("checkpoint");
    const int horizonColumn = column("horizon");
    if (idColumn < 0 || checkpointColumn < 0 || horizonColumn < 0) {
        error = "Episode manifest requires event_id, checkpoint, and horizon";
        return false;
    }
    std::unordered_map<std::string, bool> ids;
    int lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> values = splitCsvRow(line);
        if (values.size() != header.size()) {
            error = "Malformed episode manifest row " +
                std::to_string(lineNumber);
            return false;
        }
        Episode episode;
        episode.id = values[idColumn];
        episode.checkpoint = values[checkpointColumn];
        try {
            episode.horizon = std::stoi(values[horizonColumn]);
        } catch (const std::exception&) {
            error = "Invalid episode horizon on row " +
                std::to_string(lineNumber);
            return false;
        }
        if (episode.id.empty() || episode.checkpoint.empty() ||
            episode.horizon <= 0 || !ids.emplace(episode.id, true).second) {
            error = "Invalid or duplicate episode on row " +
                std::to_string(lineNumber);
            return false;
        }
        const std::string key = episodeKey(
            episode.checkpoint, episode.horizon);
        if (!episodes.emplace(key, std::move(episode)).second) {
            error = "Duplicate checkpoint/horizon in episode manifest";
            return false;
        }
    }
    if (episodes.empty()) {
        error = "Episode manifest contains no events";
        return false;
    }
    return true;
}

bool parseHorizons(
    const std::string& value,
    std::vector<int>& horizons) {
    horizons.clear();
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            const int horizon = std::stoi(item);
            if (horizon <= 0) {
                return false;
            }
            horizons.push_back(horizon);
        } catch (const std::exception&) {
            return false;
        }
    }
    std::sort(horizons.begin(), horizons.end());
    horizons.erase(
        std::unique(horizons.begin(), horizons.end()),
        horizons.end());
    return !horizons.empty();
}

bool parseOptions(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output") {
            if (++index >= argc) {
                return false;
            }
            options.output = argv[index];
        } else if (argument == "--episode-manifest") {
            if (++index >= argc) {
                return false;
            }
            options.episode_manifest = argv[index];
        } else if (argument == "--static-scan-only") {
            options.static_scan_only = true;
        } else if (argument == "--selection") {
            if (++index >= argc) {
                return false;
            }
            options.selection = argv[index];
        } else if (argument == "--prior-subspace") {
            if (++index >= argc) {
                return false;
            }
            options.prior_subspace = argv[index];
        } else if (argument == "--intervention-projection") {
            if (++index >= argc) {
                return false;
            }
            options.intervention_projection = argv[index];
        } else if (argument == "--horizons") {
            if (++index >= argc ||
                !parseHorizons(argv[index], options.horizons)) {
                return false;
            }
        } else if (argument == "--min-inliers") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.min_inliers = std::stoi(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (options.min_inliers <= 0) {
                return false;
            }
        } else if (argument == "--information-scale-multiplier") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.information_scale_multiplier =
                    std::stof(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (!std::isfinite(options.information_scale_multiplier) ||
                options.information_scale_multiplier < 0.0f) {
                return false;
            }
        } else if (argument == "--posterior-local-map") {
            options.posterior_local_map = true;
        } else if (argument == "--lag-frames") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.lag_frames = std::stoi(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (options.lag_frames <= 0) {
                return false;
            }
        } else if (argument == "--translation-blend") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.translation_blend = std::stof(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (!std::isfinite(options.translation_blend) ||
                options.translation_blend < 0.0f ||
                options.translation_blend > 1.0f) {
                return false;
            }
        } else if (
            argument == "--posterior-min-score-improvement") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.posterior_min_score_improvement =
                    std::stof(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (!std::isfinite(
                    options.posterior_min_score_improvement) ||
                options.posterior_min_score_improvement < 0.0f) {
                return false;
            }
        } else if (
            argument == "--max-static-information-leverage") {
            if (++index >= argc) {
                return false;
            }
            try {
                options.max_static_information_leverage =
                    std::stof(argv[index]);
            } catch (const std::exception&) {
                return false;
            }
            if (!std::isfinite(
                    options.max_static_information_leverage) ||
                options.max_static_information_leverage < 0.0f) {
                return false;
            }
        } else if (
            argument == "--static-information-leverage-mode") {
            if (++index >= argc ||
                !Motion3D::parseTranslationLeverageMode(
                    argv[index],
                    options.static_information_leverage_mode)) {
                return false;
            }
        } else if (argument == "--gate-policy") {
            if (++index >= argc ||
                !Motion3D::parseReplayGatePolicy(
                    argv[index], options.gate_policy)) {
                return false;
            }
            options.gate_policy_override = true;
        } else if (argument.rfind("--", 0) == 0) {
            return false;
        } else {
            options.packet_paths.push_back(argument);
        }
    }
    return !options.output.empty() &&
        !options.packet_paths.empty() &&
        (options.selection == "all" ||
         options.selection == "would-use" ||
         options.selection == "consensus") &&
        (options.prior_subspace == "full" ||
         options.prior_subspace == "translation") &&
        (options.intervention_projection == "full" ||
         options.intervention_projection == "translation" ||
         options.intervention_projection == "common-score" ||
         options.intervention_projection == "shadow-anchor" ||
         options.intervention_projection == "shadow-translation") &&
        (options.static_information_leverage_mode !=
             Motion3D::TranslationLeverageMode::NormalizeToTarget ||
         options.max_static_information_leverage > 0.0f);
}

bool selected(
    const Motion3D::FrozenReplayPacket& packet,
    const std::string& selection) {
    if (selection == "all") {
        return true;
    }
    if (selection == "would-use") {
        return packet.expected_would_use;
    }
    return packet.expected_consensus_pass;
}

bool packetsAreContiguous(
    const std::vector<Motion3D::FrozenReplayPacket>& packets,
    std::size_t start,
    int horizon) {
    for (int offset = 1; offset < horizon; ++offset) {
        const Motion3D::FrozenReplayPacket& previous =
            packets[start + offset - 1];
        const Motion3D::FrozenReplayPacket& current =
            packets[start + offset];
        if (std::abs(
                current.previous_timestamp -
                previous.current_timestamp) > 1e-6 ||
            current.freeze.epoch != previous.freeze.epoch ||
            current.state_identity.map_id !=
                previous.state_identity.map_id ||
            current.state_identity.map_generation !=
                previous.state_identity.map_generation) {
            return false;
        }
    }
    return true;
}

Rollout runRollout(
    const std::vector<Motion3D::FrozenReplayPacket>& packets,
    std::size_t start,
    int horizon,
    RolloutMode mode,
    int minimumInliers,
    const std::string& interventionProjection) {
    Rollout rollout;
    Motion3D::FrozenReplayFrameState previousState;
    Sophus::SE3f relativeVelocity;
    bool hasVelocity = false;
    Motion3D::FrozenReplayFrameState shadowState;
    Sophus::SE3f shadowVelocity;
    Sophus::SO3f shadowRotationOffset;
    bool shadowAnchorActive =
        (interventionProjection == "shadow-anchor" ||
         interventionProjection == "shadow-translation") &&
        mode != RolloutMode::Static;
    bool shadowRotationOffsetInitialized = false;
    for (int offset = 0; offset < horizon; ++offset) {
        const Motion3D::FrozenReplayPacket& packet =
            packets[start + offset];
        Sophus::SE3f initialTcw;
        if (offset == 0) {
            initialTcw = mode == RolloutMode::Static
                ? packet.static_initial_tcw
                : packet.dynamic_initial_tcw;
        } else {
            initialTcw = relativeVelocity * previousState.tcw;
        }
        const Sophus::SE3f previousTcw = previousState.valid
            ? previousState.tcw
            : packet.previous_tcw;
        Motion3D::FrozenReplayBranchStateResult step =
            Motion3D::runFrozenReplayBranch(
                packet, previousState, initialTcw,
                offset == 0 && mode != RolloutMode::Static);
        if (!step.valid) {
            rollout.failure_step = offset + 1;
            rollout.error = step.error;
            return rollout;
        }
        if (shadowAnchorActive) {
            const Sophus::SE3f shadowPreviousTcw = shadowState.valid
                ? shadowState.tcw
                : packet.previous_tcw;
            const Sophus::SE3f shadowInitialTcw = offset == 0
                ? packet.static_initial_tcw
                : shadowVelocity * shadowState.tcw;
            Motion3D::FrozenReplayBranchStateResult shadowStep =
                Motion3D::runFrozenReplayBranch(
                    packet, shadowState, shadowInitialTcw, false);
            const bool shadowUsable =
                shadowStep.valid &&
                shadowStep.branch.matches >= packet.minimum_matches &&
                shadowStep.branch.inliers >= minimumInliers;
            if (!shadowUsable) {
                shadowAnchorActive = false;
            } else {
                const Sophus::SE3f shadowTwc =
                    shadowStep.branch.optimized_tcw.inverse();
                Sophus::SE3f dynamicTwc =
                    step.branch.optimized_tcw.inverse();
                if (!shadowRotationOffsetInitialized) {
                    const bool retainDynamicRotation =
                        interventionProjection == "shadow-anchor" &&
                        packet.expected_common_support > 0 &&
                        std::isfinite(
                            packet.expected_static_common_score) &&
                        std::isfinite(
                            packet.expected_dynamic_common_score) &&
                        packet.expected_dynamic_common_score <
                            packet.expected_static_common_score;
                    shadowRotationOffset = retainDynamicRotation
                        ? dynamicTwc.so3() * shadowTwc.so3().inverse()
                        : Sophus::SO3f();
                    shadowRotationOffsetInitialized = true;
                }
                dynamicTwc = Sophus::SE3f(
                    shadowRotationOffset * shadowTwc.so3(),
                    dynamicTwc.translation());
                step.branch.optimized_tcw = dynamicTwc.inverse();
                step.frame.tcw = step.branch.optimized_tcw;

                shadowVelocity =
                    shadowStep.branch.optimized_tcw *
                    shadowPreviousTcw.inverse();
                shadowState = std::move(shadowStep.frame);
            }
        }
        if (offset == 0 &&
            mode != RolloutMode::Static &&
            (interventionProjection == "translation" ||
             interventionProjection == "common-score")) {
            Motion3D::FrozenReplayBranchStateResult staticReference =
                Motion3D::runFrozenReplayBranch(
                    packet, previousState,
                    packet.static_initial_tcw, false);
            if (!staticReference.valid) {
                rollout.failure_step = 1;
                rollout.error =
                    "static projection reference failed: " +
                    staticReference.error;
                return rollout;
            }
            const bool retainDynamicRotation =
                interventionProjection == "common-score" &&
                packet.expected_common_support > 0 &&
                std::isfinite(packet.expected_static_common_score) &&
                std::isfinite(packet.expected_dynamic_common_score) &&
                packet.expected_dynamic_common_score <
                    packet.expected_static_common_score;
            if (!retainDynamicRotation) {
                step.branch.optimized_tcw =
                    Motion3D::projectCameraInterventionToTranslation(
                        staticReference.branch.optimized_tcw,
                        step.branch.optimized_tcw);
                step.frame.tcw = step.branch.optimized_tcw;
            }
        }
        rollout.min_matches =
            std::min(rollout.min_matches, step.branch.matches);
        rollout.min_inliers =
            std::min(rollout.min_inliers, step.branch.inliers);
        if (offset == 0) {
            rollout.first_matches = step.branch.matches;
            rollout.first_inliers = step.branch.inliers;
        }
        rollout.final_matches = step.branch.matches;
        rollout.final_inliers = step.branch.inliers;
        rollout.final_tcw = step.branch.optimized_tcw;
        if (step.branch.matches < packet.minimum_matches ||
            step.branch.inliers < minimumInliers) {
            rollout.failure_step = offset + 1;
            rollout.error = "tracking support fell below threshold";
            return rollout;
        }
        if (offset == 0 &&
            mode == RolloutMode::DynamicVelocityNeutral) {
            relativeVelocity = packet.velocity;
        } else {
            relativeVelocity =
                step.branch.optimized_tcw * previousTcw.inverse();
        }
        hasVelocity = true;
        previousState = std::move(step.frame);
    }
    rollout.success = hasVelocity;
    return rollout;
}

std::uint64_t currentFrameId(
    const Motion3D::FrozenReplayPacket& packet) {
    return packet.state_identity.previous_frame_id + 1;
}

Rollout runMethodRollout(
    const std::vector<Motion3D::FrozenReplayPacket>& packets,
    const std::unordered_map<std::uint64_t, std::size_t>& packetByFrame,
    std::size_t start,
    int horizon,
    MethodMode mode,
    int minimumInliers,
    int lagFrames,
    float translationBlendOverride,
    float posteriorMinScoreImprovementOverride,
    float maxStaticInformationLeverage,
    Motion3D::TranslationLeverageMode staticInformationLeverageMode) {
    Rollout rollout;
    const Motion3D::FrozenReplayPacket& checkpoint = packets[start];
    Motion3D::FrozenReplayPacket gatePacket = checkpoint;
    bool candidateAvailable = true;
    if (mode == MethodMode::Lag) {
        const std::uint64_t frame = currentFrameId(checkpoint);
        if (frame < static_cast<std::uint64_t>(lagFrames)) {
            candidateAvailable = false;
        } else {
            const auto found = packetByFrame.find(frame - lagFrames);
            if (found == packetByFrame.end()) {
                candidateAvailable = false;
            } else {
                const Motion3D::FrozenReplayPacket& source =
                    packets[found->second];
                const Sophus::SE3f laggedRelative =
                    source.dynamic_initial_tcw *
                    source.previous_tcw.inverse();
                gatePacket.dynamic_initial_tcw =
                    laggedRelative * checkpoint.previous_tcw;
                gatePacket.dynamic_pose_information =
                    source.dynamic_pose_information;
                gatePacket.dynamic_information_scale =
                    checkpoint.dynamic_information_scale;
            }
        }
    }

    const Motion3D::FrozenReplayResult gate =
        Motion3D::runFrozenReplay(
            candidateAvailable ? gatePacket : checkpoint);
    if (!gate.valid) {
        rollout.error = "pre-gate replay failed: " + gate.error;
        return rollout;
    }
    rollout.candidate =
        mode != MethodMode::Semantic &&
        candidateAvailable && gate.would_use;

    const float blend = translationBlendOverride >= 0.0f
        ? translationBlendOverride
        : checkpoint.shadow_translation_blend;
    const float minimumScoreImprovement =
        posteriorMinScoreImprovementOverride >= 0.0f
        ? posteriorMinScoreImprovementOverride
        : checkpoint.posterior_min_score_improvement;
    const Sophus::SE3f dynamicPrior =
        Motion3D::reuseShadowCameraTranslation(
            gate.static_branch.optimized_tcw,
            gate.dynamic_branch.optimized_tcw,
            blend);
    const Eigen::Matrix<float, 6, 6> information =
        Motion3D::isotropicTranslationInformation(
            candidateAvailable
                ? gatePacket.dynamic_pose_information
                : checkpoint.dynamic_pose_information);
    const Motion3D::FrozenReplayPosteriorResult posterior =
        Motion3D::runFrozenReplayPosterior(
            checkpoint,
            gate.static_branch.optimized_tcw,
            dynamicPrior,
            information,
            checkpoint.dynamic_information_scale,
            minimumScoreImprovement,
            maxStaticInformationLeverage,
            staticInformationLeverageMode);
    if (!posterior.valid) {
        rollout.error =
            "posterior local-map replay failed: " + posterior.error;
        return rollout;
    }
    rollout.posterior_valid =
        candidateAvailable && posterior.gate_valid;
    rollout.posterior_pass =
        rollout.candidate && posterior.pass;
    rollout.posterior_static_inliers =
        posterior.static_branch.inliers;
    rollout.posterior_dynamic_inliers =
        posterior.dynamic_branch.inliers;
    rollout.posterior_common_support = posterior.common_support;
    rollout.posterior_static_score = posterior.static_score;
    rollout.posterior_dynamic_score = posterior.dynamic_score;
    if (posterior.gate_valid) {
        rollout.posterior_score_improvement =
            posterior.static_score - posterior.dynamic_score;
    }
    rollout.posterior_translation_innovation =
        Motion3D::cameraCenterTranslationInnovation(
            posterior.static_branch.optimized_tcw,
            posterior.dynamic_branch.optimized_tcw);
    rollout.leverage = posterior.leverage;

    const bool commitDynamic =
        mode != MethodMode::Semantic && rollout.posterior_pass;
    Motion3D::FrozenReplayFrameState previousState =
        commitDynamic
        ? posterior.dynamic_frame
        : posterior.static_frame;
    const Motion3D::FrozenReplayBranchResult& firstBranch =
        commitDynamic
        ? posterior.dynamic_branch
        : posterior.static_branch;
    rollout.min_matches = firstBranch.matches;
    rollout.min_inliers = firstBranch.inliers;
    rollout.first_matches = firstBranch.matches;
    rollout.first_inliers = firstBranch.inliers;
    rollout.final_matches = firstBranch.matches;
    rollout.final_inliers = firstBranch.inliers;
    rollout.final_tcw = firstBranch.optimized_tcw;
    if (firstBranch.matches < checkpoint.minimum_matches ||
        firstBranch.inliers < minimumInliers) {
        rollout.failure_step = 1;
        rollout.error =
            "posterior tracking support fell below threshold";
        return rollout;
    }

    Sophus::SE3f relativeVelocity = commitDynamic
        ? checkpoint.velocity
        : firstBranch.optimized_tcw *
            checkpoint.previous_tcw.inverse();
    for (int offset = 1; offset < horizon; ++offset) {
        const Motion3D::FrozenReplayPacket& packet =
            packets[start + offset];
        const Sophus::SE3f initialTcw =
            relativeVelocity * previousState.tcw;
        const Sophus::SE3f previousTcw = previousState.tcw;
        Motion3D::FrozenReplayBranchStateResult step =
            Motion3D::runFrozenReplayBranch(
                packet, previousState, initialTcw, false);
        if (!step.valid) {
            rollout.failure_step = offset + 1;
            rollout.error = step.error;
            return rollout;
        }
        rollout.min_matches =
            std::min(rollout.min_matches, step.branch.matches);
        rollout.min_inliers =
            std::min(rollout.min_inliers, step.branch.inliers);
        rollout.final_matches = step.branch.matches;
        rollout.final_inliers = step.branch.inliers;
        rollout.final_tcw = step.branch.optimized_tcw;
        if (step.branch.matches < packet.minimum_matches ||
            step.branch.inliers < minimumInliers) {
            rollout.failure_step = offset + 1;
            rollout.error =
                "tracking support fell below threshold";
            return rollout;
        }
        relativeVelocity =
            step.branch.optimized_tcw * previousTcw.inverse();
        previousState = std::move(step.frame);
    }
    rollout.success = true;
    return rollout;
}

Rollout runPosteriorForkRollout(
    const std::vector<Motion3D::FrozenReplayPacket>& packets,
    std::size_t start,
    int horizon,
    PosteriorForkMode mode,
    int minimumInliers,
    float maxStaticInformationLeverage,
    Motion3D::TranslationLeverageMode staticInformationLeverageMode) {
    Rollout rollout;
    const Motion3D::FrozenReplayPacket& checkpoint = packets[start];
    const bool dynamic = mode == PosteriorForkMode::DynamicVelocityNeutral;
    Sophus::SE3f staticInitialTcw =
        checkpoint.expected_static.optimized_tcw;
    Sophus::SE3f dynamicPriorTcw = staticInitialTcw;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    float dynamicInformationScale = 0.0f;
    if (dynamic) {
        const Motion3D::FrozenReplayResult gate =
            Motion3D::runFrozenReplay(checkpoint);
        if (!gate.valid) {
            rollout.error = "pre-gate replay failed: " + gate.error;
            return rollout;
        }
        staticInitialTcw = gate.static_branch.optimized_tcw;
        dynamicPriorTcw = Motion3D::reuseShadowCameraTranslation(
            gate.static_branch.optimized_tcw,
            gate.dynamic_branch.optimized_tcw,
            checkpoint.shadow_translation_blend);
        dynamicInformation =
            Motion3D::isotropicTranslationInformation(
                checkpoint.dynamic_pose_information);
        dynamicInformationScale = checkpoint.dynamic_information_scale;
    }
    const Motion3D::FrozenReplayPosteriorResult posterior =
        Motion3D::runFrozenReplayPosterior(
            checkpoint,
            staticInitialTcw,
            dynamicPriorTcw,
            dynamicInformation,
            dynamicInformationScale,
            checkpoint.posterior_min_score_improvement,
            maxStaticInformationLeverage,
            staticInformationLeverageMode,
            dynamic);
    if (!posterior.valid) {
        rollout.error = "posterior fork failed: " + posterior.error;
        return rollout;
    }

    const Motion3D::FrozenReplayBranchResult& firstBranch = dynamic
        ? posterior.dynamic_branch : posterior.static_branch;
    Motion3D::FrozenReplayFrameState previousState = dynamic
        ? posterior.dynamic_frame : posterior.static_frame;
    rollout.first_matches = firstBranch.matches;
    rollout.first_inliers = firstBranch.inliers;
    rollout.final_matches = firstBranch.matches;
    rollout.final_inliers = firstBranch.inliers;
    rollout.min_matches = firstBranch.matches;
    rollout.min_inliers = firstBranch.inliers;
    rollout.final_tcw = firstBranch.optimized_tcw;
    rollout.leverage = posterior.leverage;
    if (firstBranch.matches < checkpoint.minimum_matches ||
        firstBranch.inliers < minimumInliers) {
        rollout.failure_step = 1;
        rollout.error = "posterior tracking support fell below threshold";
        return rollout;
    }

    Sophus::SE3f relativeVelocity = dynamic
        ? checkpoint.velocity
        : firstBranch.optimized_tcw * checkpoint.previous_tcw.inverse();
    for (int offset = 1; offset < horizon; ++offset) {
        const Motion3D::FrozenReplayPacket& packet = packets[start + offset];
        const Sophus::SE3f initialTcw =
            relativeVelocity * previousState.tcw;
        const Sophus::SE3f previousTcw = previousState.tcw;
        Motion3D::FrozenReplayBranchStateResult step =
            Motion3D::runFrozenReplayBranch(
                packet, previousState, initialTcw, false);
        if (!step.valid) {
            rollout.failure_step = offset + 1;
            rollout.error = step.error;
            return rollout;
        }
        rollout.min_matches =
            std::min(rollout.min_matches, step.branch.matches);
        rollout.min_inliers =
            std::min(rollout.min_inliers, step.branch.inliers);
        rollout.final_matches = step.branch.matches;
        rollout.final_inliers = step.branch.inliers;
        rollout.final_tcw = step.branch.optimized_tcw;
        if (step.branch.matches < packet.minimum_matches ||
            step.branch.inliers < minimumInliers) {
            rollout.failure_step = offset + 1;
            rollout.error = "tracking support fell below threshold";
            return rollout;
        }
        relativeVelocity =
            step.branch.optimized_tcw * previousTcw.inverse();
        previousState = std::move(step.frame);
    }
    rollout.success = true;
    return rollout;
}

void writePose(std::ostream& stream, const Sophus::SE3f& pose) {
    const Eigen::Quaternionf quaternion = pose.unit_quaternion();
    stream << ',' << pose.translation().x()
           << ',' << pose.translation().y()
           << ',' << pose.translation().z()
           << ',' << quaternion.x()
           << ',' << quaternion.y()
           << ',' << quaternion.z()
           << ',' << quaternion.w();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        std::cerr
            << "Usage: " << argv[0]
            << " --output forks.csv"
            << " [--episode-manifest episodes.csv]"
            << " [--static-scan-only]"
            << " [--selection all|would-use|consensus]"
            << " [--prior-subspace full|translation]"
            << " [--intervention-projection"
            << " full|translation|common-score|shadow-anchor|"
            << "shadow-translation]"
            << " [--horizons 1,5,10] [--min-inliers 10]"
            << " [--information-scale-multiplier S]"
            << " [--posterior-local-map] [--lag-frames 30]"
            << " [--translation-blend B]"
            << " [--posterior-min-score-improvement S]"
            << " [--max-static-information-leverage L]"
            << " [--static-information-leverage-mode"
            << " cap-only|normalize-to-target]"
            << " [--gate-policy legacy-conjunction|direct-combined]"
            << " packet.yml.gz [packet.yml.gz ...]\n";
        return 2;
    }

    std::unordered_map<std::string, Episode> episodes;
    std::string episodeError;
    if (!loadEpisodes(
            options.episode_manifest, episodes, episodeError)) {
        std::cerr << episodeError << '\n';
        return 2;
    }

    std::vector<Motion3D::FrozenReplayPacket> packets;
    packets.reserve(options.packet_paths.size());
    for (const std::string& path : options.packet_paths) {
        Motion3D::FrozenReplayPacket packet;
        std::string error;
        if (!Motion3D::readFrozenReplayPacket(path, packet, &error)) {
            std::cerr << path << ": " << error << '\n';
            return 1;
        }
        packet.dynamic_information_scale *=
            options.information_scale_multiplier;
        if (options.gate_policy_override) {
            packet.gate.policy = options.gate_policy;
        }
        if (options.prior_subspace == "translation") {
            Motion3D::projectPosePriorToTranslationSubspace(
                packet.static_initial_tcw,
                packet.dynamic_initial_tcw,
                packet.dynamic_pose_information);
        }
        packets.push_back(std::move(packet));
    }
    std::vector<std::size_t> packetOrder(packets.size());
    for (std::size_t index = 0; index < packetOrder.size(); ++index) {
        packetOrder[index] = index;
    }
    std::sort(
        packetOrder.begin(),
        packetOrder.end(),
        [&packets, &options](std::size_t left, std::size_t right) {
            if (packets[left].current_timestamp !=
                packets[right].current_timestamp) {
                return packets[left].current_timestamp <
                    packets[right].current_timestamp;
            }
            return options.packet_paths[left] <
                options.packet_paths[right];
        });
    std::vector<Motion3D::FrozenReplayPacket> sortedPackets;
    std::vector<std::string> sortedPaths;
    sortedPackets.reserve(packets.size());
    sortedPaths.reserve(options.packet_paths.size());
    for (std::size_t index : packetOrder) {
        if (!sortedPackets.empty() &&
            packets[index].current_timestamp <=
                sortedPackets.back().current_timestamp) {
            std::cerr
                << "Packet timestamps must be unique and strictly increasing: "
                << sortedPaths.back() << " and "
                << options.packet_paths[index] << '\n';
            return 1;
        }
        sortedPackets.push_back(std::move(packets[index]));
        sortedPaths.push_back(std::move(options.packet_paths[index]));
    }
    packets = std::move(sortedPackets);
    options.packet_paths = std::move(sortedPaths);
    std::unordered_map<std::uint64_t, std::size_t> packetByFrame;
    for (std::size_t index = 0; index < packets.size(); ++index) {
        const std::uint64_t frame = currentFrameId(packets[index]);
        if (!packetByFrame.emplace(frame, index).second) {
            std::cerr
                << "Packet frame IDs must be unique: " << frame << '\n';
            return 1;
        }
    }

    std::ofstream output(options.output);
    if (!output.is_open()) {
        std::cerr << "Cannot open output: " << options.output << '\n';
        return 1;
    }
    output
        << "event_id,checkpoint,static_scan_only,static_executed,"
        << "dynamic_executed,matched_sham_static_executed,"
        << "matched_sham_dynamic_executed,matched_sham_selected_branch,"
        << "selection,gate_policy,prior_subspace,"
        << "intervention_projection,"
        << "information_scale_multiplier,max_static_information_leverage,"
        << "static_information_leverage_mode,"
        << "horizon,"
        << "start_timestamp,end_timestamp,"
        << "static_success,dynamic_success,static_failure_step,"
        << "dynamic_failure_step,static_min_matches,dynamic_min_matches,"
        << "static_min_inliers,dynamic_min_inliers,"
        << "static_first_matches,static_first_inliers,"
        << "static_final_matches,static_final_inliers,"
        << "dynamic_first_matches,dynamic_first_inliers,"
        << "dynamic_final_matches,dynamic_final_inliers,"
        << "velocity_neutral_success,velocity_neutral_failure_step,"
        << "velocity_neutral_min_matches,velocity_neutral_min_inliers,"
        << "velocity_neutral_first_matches,velocity_neutral_first_inliers,"
        << "velocity_neutral_final_matches,velocity_neutral_final_inliers,"
        << "replay_mode,shadow_success,shadow_failure_step,"
        << "lag30_success,lag30_failure_step,"
        << "lag30_min_matches,lag30_min_inliers,"
        << "schur_candidate,schur_posterior_valid,"
        << "schur_posterior_pass,schur_posterior_static_inliers,"
        << "schur_posterior_dynamic_inliers,"
        << "schur_posterior_common_support,"
        << "schur_posterior_static_score,"
        << "schur_posterior_dynamic_score,"
        << "schur_posterior_score_improvement,"
        << "schur_posterior_translation_innovation,"
        << "schur_static_translation_information_trace,"
        << "schur_dynamic_translation_information_trace_raw,"
        << "schur_dynamic_translation_information_trace_bounded,"
        << "schur_max_generalized_leverage_raw,"
        << "schur_max_generalized_leverage_bounded,"
        << "schur_leverage_normalization_scale,"
        << "lag30_candidate,lag30_posterior_valid,"
        << "lag30_posterior_pass,"
        << "lag30_posterior_static_inliers,"
        << "lag30_posterior_dynamic_inliers,"
        << "lag30_posterior_common_support,"
        << "lag30_posterior_static_score,"
        << "lag30_posterior_dynamic_score,"
        << "lag30_posterior_score_improvement,"
        << "lag30_posterior_translation_innovation,"
        << "lag30_static_translation_information_trace,"
        << "lag30_dynamic_translation_information_trace_raw,"
        << "lag30_dynamic_translation_information_trace_bounded,"
        << "lag30_max_generalized_leverage_raw,"
        << "lag30_max_generalized_leverage_bounded,"
        << "lag30_leverage_normalization_scale,"
        << "lag_frames,translation_blend,"
        << "posterior_min_score_improvement,"
        << "start_tcw_tx,start_tcw_ty,start_tcw_tz,start_tcw_qx,"
        << "start_tcw_qy,start_tcw_qz,start_tcw_qw,"
        << "static_tcw_tx,static_tcw_ty,static_tcw_tz,static_tcw_qx,"
        << "static_tcw_qy,static_tcw_qz,static_tcw_qw,"
        << "dynamic_tcw_tx,dynamic_tcw_ty,dynamic_tcw_tz,dynamic_tcw_qx,"
        << "dynamic_tcw_qy,dynamic_tcw_qz,dynamic_tcw_qw,"
        << "velocity_neutral_tcw_tx,velocity_neutral_tcw_ty,"
        << "velocity_neutral_tcw_tz,velocity_neutral_tcw_qx,"
        << "velocity_neutral_tcw_qy,velocity_neutral_tcw_qz,"
        << "velocity_neutral_tcw_qw,"
        << "shadow_tcw_tx,shadow_tcw_ty,shadow_tcw_tz,shadow_tcw_qx,"
        << "shadow_tcw_qy,shadow_tcw_qz,shadow_tcw_qw,"
        << "lag30_tcw_tx,lag30_tcw_ty,lag30_tcw_tz,lag30_tcw_qx,"
        << "lag30_tcw_qy,lag30_tcw_qz,lag30_tcw_qw\n";
    output << std::scientific << std::setprecision(9);

    int checkpoints = 0;
    int rows = 0;
    for (std::size_t start = 0; start < packets.size(); ++start) {
        const Motion3D::FrozenReplayPacket& checkpoint = packets[start];
        if (!selected(checkpoint, options.selection)) {
            continue;
        }
        ++checkpoints;
        for (int horizon : options.horizons) {
            if (start + static_cast<std::size_t>(horizon) >
                    packets.size() ||
                !packetsAreContiguous(packets, start, horizon)) {
                continue;
            }
            Episode* episode = nullptr;
            if (!episodes.empty()) {
                const std::string key = episodeKey(
                    std::filesystem::path(
                        options.packet_paths[start]).filename().string(),
                    horizon);
                const auto found = episodes.find(key);
                if (found == episodes.end()) {
                    continue;
                }
                episode = &found->second;
            }
            Rollout staticRollout;
            Rollout dynamicRollout;
            Rollout velocityNeutralRollout;
            Rollout shadowRollout;
            Rollout lagRollout;
            if (options.static_scan_only) {
                staticRollout = options.posterior_local_map
                    ? runPosteriorForkRollout(
                          packets, start, horizon,
                          PosteriorForkMode::Static,
                          options.min_inliers,
                          options.max_static_information_leverage,
                          options.static_information_leverage_mode)
                    : runRollout(
                          packets, start, horizon, RolloutMode::Static,
                          options.min_inliers, "full");
            } else if (options.posterior_local_map && !episodes.empty()) {
                if (!checkpoint.local_map_snapshot_valid)
                    continue;
                staticRollout = runPosteriorForkRollout(
                    packets, start, horizon,
                    PosteriorForkMode::Static,
                    options.min_inliers,
                    options.max_static_information_leverage,
                    options.static_information_leverage_mode);
                dynamicRollout = runPosteriorForkRollout(
                    packets, start, horizon,
                    PosteriorForkMode::DynamicVelocityNeutral,
                    options.min_inliers,
                    options.max_static_information_leverage,
                    options.static_information_leverage_mode);
                velocityNeutralRollout = dynamicRollout;
                shadowRollout = staticRollout;
            } else if (options.posterior_local_map) {
                if (!checkpoint.local_map_snapshot_valid)
                    continue;
                staticRollout = runMethodRollout(
                    packets, packetByFrame, start, horizon,
                    MethodMode::Semantic, options.min_inliers,
                    options.lag_frames, options.translation_blend,
                    options.posterior_min_score_improvement,
                    options.max_static_information_leverage,
                    options.static_information_leverage_mode);
                dynamicRollout = runMethodRollout(
                    packets, packetByFrame, start, horizon,
                    MethodMode::Schur, options.min_inliers,
                    options.lag_frames, options.translation_blend,
                    options.posterior_min_score_improvement,
                    options.max_static_information_leverage,
                    options.static_information_leverage_mode);
                velocityNeutralRollout = dynamicRollout;
                shadowRollout = staticRollout;
                shadowRollout.candidate = dynamicRollout.candidate;
                shadowRollout.posterior_valid =
                    dynamicRollout.posterior_valid;
                shadowRollout.posterior_pass =
                    dynamicRollout.posterior_pass;
                lagRollout = runMethodRollout(
                    packets, packetByFrame, start, horizon,
                    MethodMode::Lag, options.min_inliers,
                    options.lag_frames, options.translation_blend,
                    options.posterior_min_score_improvement,
                    options.max_static_information_leverage,
                    options.static_information_leverage_mode);
            } else {
                staticRollout = runRollout(
                    packets, start, horizon, RolloutMode::Static,
                    options.min_inliers, "full");
                dynamicRollout = runRollout(
                    packets, start, horizon,
                    RolloutMode::DynamicPropagated,
                    options.min_inliers,
                    options.intervention_projection);
                velocityNeutralRollout = runRollout(
                    packets, start, horizon,
                    RolloutMode::DynamicVelocityNeutral,
                    options.min_inliers,
                    options.intervention_projection);
                shadowRollout = staticRollout;
            }
            const float effectiveBlend =
                options.translation_blend >= 0.0f
                ? options.translation_blend
                : checkpoint.shadow_translation_blend;
            const float effectivePosteriorScore =
                options.posterior_min_score_improvement >= 0.0f
                ? options.posterior_min_score_improvement
                : checkpoint.posterior_min_score_improvement;
            output
                << (episode ? episode->id : "")
                << ','
                << std::filesystem::path(
                    options.packet_paths[start]).filename().string()
                << ',' << (options.static_scan_only ? 1 : 0)
                << ",1,"
                << (options.static_scan_only ? 0 : 1)
                << ',' << (options.static_scan_only ? 0 : 1)
                << ',' << (options.static_scan_only ? 0 : 1)
                << ',' << (options.static_scan_only ? "none" : "static")
                << ',' << options.selection
                << ','
                << Motion3D::replayGatePolicyName(
                    checkpoint.gate.policy)
                << ',' << options.prior_subspace
                << ',' << options.intervention_projection
                << ',' << options.information_scale_multiplier
                << ',' << options.max_static_information_leverage
                << ','
                << Motion3D::translationLeverageModeName(
                    options.static_information_leverage_mode)
                << ',' << horizon
                << std::setprecision(17)
                << ',' << checkpoint.previous_timestamp
                << ',' << packets[start + horizon - 1].current_timestamp
                << std::setprecision(9)
                << ',' << (staticRollout.success ? 1 : 0)
                << ',' << (dynamicRollout.success ? 1 : 0)
                << ',' << staticRollout.failure_step
                << ',' << dynamicRollout.failure_step
                << ',' << staticRollout.min_matches
                << ',' << dynamicRollout.min_matches
                << ',' << staticRollout.min_inliers
                << ',' << dynamicRollout.min_inliers
                << ',' << staticRollout.first_matches
                << ',' << staticRollout.first_inliers
                << ',' << staticRollout.final_matches
                << ',' << staticRollout.final_inliers
                << ',' << dynamicRollout.first_matches
                << ',' << dynamicRollout.first_inliers
                << ',' << dynamicRollout.final_matches
                << ',' << dynamicRollout.final_inliers
                << ',' << (velocityNeutralRollout.success ? 1 : 0)
                << ',' << velocityNeutralRollout.failure_step
                << ',' << velocityNeutralRollout.min_matches
                << ',' << velocityNeutralRollout.min_inliers
                << ',' << velocityNeutralRollout.first_matches
                << ',' << velocityNeutralRollout.first_inliers
                << ',' << velocityNeutralRollout.final_matches
                << ',' << velocityNeutralRollout.final_inliers
                << ',' << (options.posterior_local_map
                                ? "posterior-local-map-v1"
                                : "legacy-motion-model-v1")
                << ',' << (shadowRollout.success ? 1 : 0)
                << ',' << shadowRollout.failure_step
                << ',' << (lagRollout.success ? 1 : 0)
                << ',' << lagRollout.failure_step
                << ',' << lagRollout.min_matches
                << ',' << lagRollout.min_inliers
                << ',' << (dynamicRollout.candidate ? 1 : 0)
                << ',' << (dynamicRollout.posterior_valid ? 1 : 0)
                << ',' << (dynamicRollout.posterior_pass ? 1 : 0)
                << ',' << dynamicRollout.posterior_static_inliers
                << ',' << dynamicRollout.posterior_dynamic_inliers
                << ',' << dynamicRollout.posterior_common_support
                << ',' << dynamicRollout.posterior_static_score
                << ',' << dynamicRollout.posterior_dynamic_score
                << ',' << dynamicRollout.posterior_score_improvement
                << ',' << dynamicRollout.posterior_translation_innovation
                << ',' << dynamicRollout.leverage.static_translation_trace
                << ',' << dynamicRollout.leverage.dynamic_translation_trace_raw
                << ','
                << dynamicRollout.leverage.dynamic_translation_trace_bounded
                << ','
                << dynamicRollout.leverage.max_generalized_leverage_raw
                << ','
                << dynamicRollout.leverage.max_generalized_leverage_bounded
                << ',' << dynamicRollout.leverage.normalization_scale
                << ',' << (lagRollout.candidate ? 1 : 0)
                << ',' << (lagRollout.posterior_valid ? 1 : 0)
                << ',' << (lagRollout.posterior_pass ? 1 : 0)
                << ',' << lagRollout.posterior_static_inliers
                << ',' << lagRollout.posterior_dynamic_inliers
                << ',' << lagRollout.posterior_common_support
                << ',' << lagRollout.posterior_static_score
                << ',' << lagRollout.posterior_dynamic_score
                << ',' << lagRollout.posterior_score_improvement
                << ',' << lagRollout.posterior_translation_innovation
                << ',' << lagRollout.leverage.static_translation_trace
                << ',' << lagRollout.leverage.dynamic_translation_trace_raw
                << ','
                << lagRollout.leverage.dynamic_translation_trace_bounded
                << ',' << lagRollout.leverage.max_generalized_leverage_raw
                << ','
                << lagRollout.leverage.max_generalized_leverage_bounded
                << ',' << lagRollout.leverage.normalization_scale
                << ',' << options.lag_frames
                << ',' << effectiveBlend
                << ',' << effectivePosteriorScore;
            writePose(output, checkpoint.previous_tcw);
            writePose(output, staticRollout.final_tcw);
            writePose(output, dynamicRollout.final_tcw);
            writePose(output, velocityNeutralRollout.final_tcw);
            writePose(output, shadowRollout.final_tcw);
            writePose(output, lagRollout.final_tcw);
            output << '\n';
            if (episode) {
                episode->emitted = true;
            }
            ++rows;
            if (!staticRollout.error.empty()) {
                std::cerr << options.packet_paths[start]
                          << " static horizon " << horizon
                          << ": " << staticRollout.error << '\n';
            }
            if (!dynamicRollout.error.empty()) {
                std::cerr << options.packet_paths[start]
                          << " dynamic horizon " << horizon
                          << ": " << dynamicRollout.error << '\n';
            }
            if (!velocityNeutralRollout.error.empty()) {
                std::cerr << options.packet_paths[start]
                          << " velocity-neutral horizon " << horizon
                          << ": " << velocityNeutralRollout.error << '\n';
            }
            if (options.posterior_local_map &&
                !lagRollout.error.empty()) {
                std::cerr << options.packet_paths[start]
                          << " Lag-" << options.lag_frames
                          << " horizon " << horizon
                          << ": " << lagRollout.error << '\n';
            }
        }
    }
    output.close();
    for (const auto& item : episodes) {
        if (!item.second.emitted) {
            std::cerr << "Episode was not replayed: "
                      << item.second.id << " ("
                      << item.second.checkpoint << ", H="
                      << item.second.horizon << ")\n";
            return 1;
        }
    }
    std::cout << "checkpointed-fork rows=" << rows
              << " checkpoints=" << checkpoints
              << " selection=" << options.selection
              << " gate_policy="
              << (options.gate_policy_override
                      ? Motion3D::replayGatePolicyName(
                            options.gate_policy)
                      : "packet")
              << " prior_subspace=" << options.prior_subspace
              << " intervention_projection="
              << options.intervention_projection
              << " information_scale_multiplier="
              << options.information_scale_multiplier
              << " max_static_information_leverage="
              << options.max_static_information_leverage
              << " static_information_leverage_mode="
              << Motion3D::translationLeverageModeName(
                  options.static_information_leverage_mode)
              << " replay_mode="
              << (options.posterior_local_map
                      ? "posterior-local-map-v1"
                      : "legacy-motion-model-v1")
              << " static_scan_only="
              << (options.static_scan_only ? 1 : 0)
              << " episode_manifest="
              << (options.episode_manifest.empty()
                      ? "none" : options.episode_manifest.string())
              << " lag_frames=" << options.lag_frames << '\n';
    return 0;
}
