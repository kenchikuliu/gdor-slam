#pragma once

#include <Eigen/Core>
#include <memory>
#include <unordered_map>
#include <deque>
#include <vector>

class GaussianModel;

namespace Motion3D {

class GaussianMotionTracker {
public:
    struct GaussianState {
        int gaussian_id;
        Eigen::Vector3f position;
        Eigen::Vector3f velocity;
        double timestamp;
        float confidence;
    };

    struct Config {
        int temporal_window_size;
        float matching_radius;
        bool use_rgb_similarity;
        float rgb_weight;
        float quantization_res;
        int ttl_frames;

        Config()
            : temporal_window_size(5),
              matching_radius(0.05f),
              use_rgb_similarity(true),
              rgb_weight(0.3f),
              quantization_res(0.02f),
              ttl_frames(2) {}
    };

    explicit GaussianMotionTracker(const Config& config = Config());

    void updateGaussians(
        std::shared_ptr<GaussianModel> gaussians,
        double timestamp
    );

    std::vector<Eigen::Vector3f> getMotionVectors() const;
    std::vector<GaussianState> getGaussianStates() const;
    void reset();

    struct Stats {
        int num_tracked;
        int num_lost;
        int num_new;
        float avg_velocity;
    };
    Stats getStats() const;

private:
    Config config_;
    std::unordered_map<int, GaussianState> current_state_;
    std::unordered_map<int, std::deque<GaussianState>> temporal_buffer_;
    std::unordered_map<int, int> ttl_counter_;
    Stats stats_;

    int assignGaussianID(const Eigen::Vector3f& position);
    uint64_t mortonEncode3D(int32_t x, int32_t y, int32_t z);
    std::unordered_map<int, std::vector<int>> buildSpatialIndex(
        const std::vector<Eigen::Vector3f>& positions
    );
    float computeSimilarity(
        const Eigen::Vector3f& pos1,
        const Eigen::Vector3f& rgb1,
        const Eigen::Vector3f& pos2,
        const Eigen::Vector3f& rgb2
    ) const;
    void pruneOldStates(double current_time);
};

}  // namespace Motion3D
