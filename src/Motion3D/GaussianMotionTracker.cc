#include "Motion3D/GaussianMotionTracker.h"
#include "gaussian_model.h"
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace Motion3D {

GaussianMotionTracker::GaussianMotionTracker(const Config& config)
    : config_(config), stats_{0, 0, 0, 0.0f} {}

uint64_t GaussianMotionTracker::mortonEncode3D(int32_t x, int32_t y, int32_t z) {
    auto expandBits = [](uint32_t v) -> uint64_t {
        uint64_t x = v & 0x1FFFFF;
        x = (x | x << 32) & 0x1F00000000FFFF;
        x = (x | x << 16) & 0x1F0000FF0000FF;
        x = (x | x << 8) & 0x100F00F00F00F00F;
        x = (x | x << 4) & 0x10C30C30C30C30C3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    };

    uint32_t ux = static_cast<uint32_t>(x);
    uint32_t uy = static_cast<uint32_t>(y);
    uint32_t uz = static_cast<uint32_t>(z);

    return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

int GaussianMotionTracker::assignGaussianID(const Eigen::Vector3f& position) {
    int x_bucket = static_cast<int>(std::round(position.x() / config_.quantization_res));
    int y_bucket = static_cast<int>(std::round(position.y() / config_.quantization_res));
    int z_bucket = static_cast<int>(std::round(position.z() / config_.quantization_res));

    uint64_t morton = mortonEncode3D(x_bucket, y_bucket, z_bucket);
    return static_cast<int>(morton & 0x7FFFFFFF);
}

std::unordered_map<int, std::vector<int>> GaussianMotionTracker::buildSpatialIndex(
    const std::vector<Eigen::Vector3f>& positions
) {
    std::unordered_map<int, std::vector<int>> spatial_index;

    for (size_t i = 0; i < positions.size(); ++i) {
        int grid_id = assignGaussianID(positions[i]);
        spatial_index[grid_id].push_back(i);
    }

    return spatial_index;
}

float GaussianMotionTracker::computeSimilarity(
    const Eigen::Vector3f& pos1,
    const Eigen::Vector3f& rgb1,
    const Eigen::Vector3f& pos2,
    const Eigen::Vector3f& rgb2
) const {
    float dist = (pos1 - pos2).norm();

    if (!config_.use_rgb_similarity) {
        return 1.0f / (1.0f + dist);
    }

    float rgb_dist = (rgb1 - rgb2).norm();
    float similarity = 1.0f / (1.0f + dist + config_.rgb_weight * rgb_dist);

    return similarity;
}

void GaussianMotionTracker::updateGaussians(
    std::shared_ptr<GaussianModel> gaussians,
    double timestamp
) {
    stats_ = Stats{0, 0, 0, 0.0f};

    auto xyz_tensor = gaussians->getXYZ().to(torch::kCPU).contiguous();
    int N = xyz_tensor.size(0);

    if (N == 0) {
        current_state_.clear();
        return;
    }

    auto xyz_data = xyz_tensor.accessor<float, 2>();

    std::vector<Eigen::Vector3f> positions(N);
    std::vector<Eigen::Vector3f> colors(N);

    auto features_dc_tensor = gaussians->getFeatures().to(torch::kCPU).contiguous();
    const bool has_colors = features_dc_tensor.defined() &&
        features_dc_tensor.dim() == 3 && features_dc_tensor.size(0) >= N &&
        features_dc_tensor.size(1) > 0 && features_dc_tensor.size(2) >= 3;

    for (int i = 0; i < N; ++i) {
        positions[i] = Eigen::Vector3f(xyz_data[i][0], xyz_data[i][1], xyz_data[i][2]);
        if (has_colors) {
            colors[i] = Eigen::Vector3f(
                features_dc_tensor[i][0][0].item<float>(),
                features_dc_tensor[i][0][1].item<float>(),
                features_dc_tensor[i][0][2].item<float>());
        } else {
            colors[i] = Eigen::Vector3f::Zero();
        }
    }

    auto spatial_index = buildSpatialIndex(positions);

    std::unordered_map<int, GaussianState> new_state;
    std::unordered_map<int, bool> matched;

    for (int i = 0; i < N; ++i) {
        const auto& pos = positions[i];
        int id = assignGaussianID(pos);

        Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
        float confidence = 0.0f;

        if (current_state_.count(id)) {
            const auto& prev = current_state_[id];
            float dt = timestamp - prev.timestamp;
            if (dt > 1e-6) {
                velocity = (pos - prev.position) / dt;
                confidence = 1.0f;
                stats_.num_tracked++;
                matched[id] = true;
            }
        } else {
            bool found_match = false;
            float best_similarity = 0.0f;
            const GaussianState* best_prev = nullptr;

            for (int dx = -1; dx <= 1 && !found_match; ++dx) {
                for (int dy = -1; dy <= 1 && !found_match; ++dy) {
                    for (int dz = -1; dz <= 1 && !found_match; ++dz) {
                        Eigen::Vector3f offset_pos = pos + Eigen::Vector3f(
                            dx * config_.quantization_res,
                            dy * config_.quantization_res,
                            dz * config_.quantization_res
                        );
                        int neighbor_id = assignGaussianID(offset_pos);

                        if (current_state_.count(neighbor_id)) {
                            const auto& prev = current_state_[neighbor_id];
                            float distance = (pos - prev.position).norm();

                            if (distance < config_.matching_radius) {
                                float sim = computeSimilarity(pos, colors[i],
                                                             prev.position, Eigen::Vector3f::Zero());
                                if (sim > best_similarity) {
                                    best_similarity = sim;
                                    best_prev = &prev;
                                    id = neighbor_id;
                                }
                            }
                        }
                    }
                }
            }

            if (best_prev) {
                float dt = timestamp - best_prev->timestamp;
                if (dt > 1e-6) {
                    velocity = (pos - best_prev->position) / dt;
                    confidence = best_similarity;
                    stats_.num_tracked++;
                    matched[id] = true;
                    found_match = true;
                }
            }

            if (!found_match) {
                stats_.num_new++;
            }
        }

        GaussianState state{id, pos, velocity, timestamp, confidence};
        new_state[id] = state;

        temporal_buffer_[id].push_back(state);
        if (temporal_buffer_[id].size() > static_cast<size_t>(config_.temporal_window_size)) {
            temporal_buffer_[id].pop_front();
        }
    }

    for (const auto& [old_id, old_state] : current_state_) {
        if (!matched[old_id]) {
            stats_.num_lost++;
            if (ttl_counter_[old_id]++ < config_.ttl_frames) {
                new_state[old_id] = old_state;
            } else {
                ttl_counter_.erase(old_id);
            }
        } else {
            ttl_counter_[old_id] = 0;
        }
    }

    current_state_ = std::move(new_state);

    float total_velocity = 0.0f;
    int velocity_count = 0;
    for (const auto& [id, state] : current_state_) {
        if (state.confidence > 0.0f) {
            total_velocity += state.velocity.norm();
            velocity_count++;
        }
    }
    stats_.avg_velocity = velocity_count > 0 ? total_velocity / velocity_count : 0.0f;

    pruneOldStates(timestamp);
}

std::vector<Eigen::Vector3f> GaussianMotionTracker::getMotionVectors() const {
    std::vector<Eigen::Vector3f> motion_vectors;
    motion_vectors.reserve(current_state_.size());

    for (const auto& [id, state] : current_state_) {
        if (state.confidence > 0.0f) {
            motion_vectors.push_back(state.velocity);
        }
    }

    return motion_vectors;
}

std::vector<GaussianMotionTracker::GaussianState> GaussianMotionTracker::getGaussianStates() const {
    std::vector<GaussianState> states;
    states.reserve(current_state_.size());

    for (const auto& [id, state] : current_state_) {
        states.push_back(state);
    }

    return states;
}

void GaussianMotionTracker::reset() {
    current_state_.clear();
    temporal_buffer_.clear();
    ttl_counter_.clear();
    stats_ = Stats{0, 0, 0, 0.0f};
}

GaussianMotionTracker::Stats GaussianMotionTracker::getStats() const {
    return stats_;
}

void GaussianMotionTracker::pruneOldStates(double current_time) {
    std::vector<int> to_erase;

    for (auto& [id, buffer] : temporal_buffer_) {
        while (!buffer.empty() &&
               (current_time - buffer.front().timestamp) > config_.temporal_window_size * 1.0) {
            buffer.pop_front();
        }

        if (buffer.empty()) {
            to_erase.push_back(id);
        }
    }

    for (int id : to_erase) {
        temporal_buffer_.erase(id);
    }
}

}  // namespace Motion3D
