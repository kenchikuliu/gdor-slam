#include <gtest/gtest.h>
#include "Motion3D/GaussianMotionTracker.h"
#include "gaussian_model.h"
#include <torch/torch.h>
#include <memory>
#include <cmath>

using namespace Motion3D;

std::shared_ptr<GaussianModel> createMockGaussians(torch::Tensor xyz, torch::Tensor features) {
    auto model = std::make_shared<GaussianModel>(0);

    int N = xyz.size(0);
    model->xyz_ = xyz.to(torch::kCUDA);
    model->features_dc_ = features.to(torch::kCUDA);

    return model;
}

class GaussianMotionTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = GaussianMotionTracker::Config();
    }

    GaussianMotionTracker::Config config_;
};

TEST_F(GaussianMotionTrackerTest, Initialization) {
    GaussianMotionTracker tracker(config_);
    auto stats = tracker.getStats();

    EXPECT_EQ(stats.num_tracked, 0);
    EXPECT_EQ(stats.num_lost, 0);
    EXPECT_EQ(stats.num_new, 0);
    EXPECT_FLOAT_EQ(stats.avg_velocity, 0.0f);
}

TEST_F(GaussianMotionTrackerTest, StaticScenario) {
    GaussianMotionTracker tracker(config_);

    torch::Tensor xyz1 = torch::tensor({
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features1 = torch::zeros({3, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians1 = createMockGaussians(xyz1, features1);

    tracker.updateGaussians(gaussians1, 0.0);
    auto stats1 = tracker.getStats();
    EXPECT_EQ(stats1.num_new, 3);
    EXPECT_EQ(stats1.num_tracked, 0);

    torch::Tensor xyz2 = torch::tensor({
        {0.001f, 0.0f, 0.0f},
        {1.0f, 0.001f, 0.0f},
        {0.001f, 1.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    auto gaussians2 = createMockGaussians(xyz2, features1);
    tracker.updateGaussians(gaussians2, 0.1);

    auto states = tracker.getGaussianStates();
    EXPECT_EQ(states.size(), 3);

    for (const auto& state : states) {
        EXPECT_LT(state.velocity.norm(), 0.05f)
            << "Velocity should be near zero for static scenario, got: " << state.velocity.norm();
    }

    auto stats2 = tracker.getStats();
    EXPECT_GT(stats2.num_tracked, 0);
    EXPECT_LT(stats2.avg_velocity, 0.05f);
}

TEST_F(GaussianMotionTrackerTest, LinearMotion) {
    GaussianMotionTracker tracker(config_);

    torch::Tensor xyz1 = torch::tensor({
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features1 = torch::zeros({2, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians1 = createMockGaussians(xyz1, features1);

    tracker.updateGaussians(gaussians1, 0.0);

    torch::Tensor xyz2 = torch::tensor({
        {0.01f, 0.0f, 0.0f},
        {1.01f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    auto gaussians2 = createMockGaussians(xyz2, features1);
    tracker.updateGaussians(gaussians2, 1.0);

    auto states = tracker.getGaussianStates();
    EXPECT_GE(states.size(), 2);

    int num_tracked = 0;
    float total_velocity_x = 0.0f;
    for (const auto& state : states) {
        if (state.confidence > 0.0f) {
            total_velocity_x += state.velocity.x();
            num_tracked++;
        }
    }

    EXPECT_GT(num_tracked, 0);
    if (num_tracked > 0) {
        float avg_vx = total_velocity_x / num_tracked;
        EXPECT_NEAR(avg_vx, 0.01f, 0.015f)
            << "Expected average velocity ~0.01 m/s in x direction";
    }
}

TEST_F(GaussianMotionTrackerTest, NewGaussianCreation) {
    GaussianMotionTracker tracker(config_);

    torch::Tensor xyz1 = torch::tensor({
        {0.0f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features1 = torch::zeros({1, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians1 = createMockGaussians(xyz1, features1);

    tracker.updateGaussians(gaussians1, 0.0);
    auto stats1 = tracker.getStats();
    EXPECT_EQ(stats1.num_new, 1);
    EXPECT_EQ(stats1.num_tracked, 0);

    auto states1 = tracker.getGaussianStates();
    EXPECT_EQ(states1.size(), 1);
    EXPECT_FLOAT_EQ(states1[0].confidence, 0.0f);
    EXPECT_FLOAT_EQ(states1[0].velocity.norm(), 0.0f);

    torch::Tensor xyz2 = torch::tensor({
        {0.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 5.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features2 = torch::zeros({2, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians2 = createMockGaussians(xyz2, features2);

    tracker.updateGaussians(gaussians2, 0.1);
    auto stats2 = tracker.getStats();
    EXPECT_EQ(stats2.num_new, 1);
    EXPECT_EQ(stats2.num_tracked, 1);

    auto states2 = tracker.getGaussianStates();
    EXPECT_EQ(states2.size(), 2);

    bool found_new = false;
    bool found_tracked = false;
    for (const auto& state : states2) {
        if (state.confidence == 0.0f) {
            found_new = true;
        } else if (state.confidence > 0.0f) {
            found_tracked = true;
        }
    }
    EXPECT_TRUE(found_new);
    EXPECT_TRUE(found_tracked);
}

TEST_F(GaussianMotionTrackerTest, MatchingRobustness) {
    GaussianMotionTracker tracker(config_);

    torch::Tensor xyz1 = torch::tensor({
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features1 = torch::zeros({3, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians1 = createMockGaussians(xyz1, features1);

    tracker.updateGaussians(gaussians1, 0.0);

    torch::Tensor noise = torch::randn({3, 3}) * 0.005f;
    torch::Tensor xyz2 = xyz1 + noise;

    auto gaussians2 = createMockGaussians(xyz2, features1);
    tracker.updateGaussians(gaussians2, 0.1);

    auto stats = tracker.getStats();
    EXPECT_GT(stats.num_tracked, 0);

    auto states = tracker.getGaussianStates();
    EXPECT_EQ(states.size(), 3);

    for (const auto& state : states) {
        if (state.confidence > 0.0f) {
            EXPECT_LT(state.velocity.norm(), 0.3f);
        }
    }
}

TEST_F(GaussianMotionTrackerTest, TTLMechanism) {
    config_.ttl_frames = 2;
    GaussianMotionTracker tracker(config_);

    torch::Tensor xyz1 = torch::tensor({
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features1 = torch::zeros({2, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians1 = createMockGaussians(xyz1, features1);

    tracker.updateGaussians(gaussians1, 0.0);
    EXPECT_EQ(tracker.getGaussianStates().size(), 2);

    torch::Tensor xyz2 = torch::tensor({
        {1.0f, 0.0f, 0.0f}
    }, torch::dtype(torch::kFloat32));

    torch::Tensor features2 = torch::zeros({1, 1, 3}, torch::dtype(torch::kFloat32));
    auto gaussians2 = createMockGaussians(xyz2, features2);

    tracker.updateGaussians(gaussians2, 0.1);
    auto stats1 = tracker.getStats();
    EXPECT_EQ(stats1.num_lost, 1);
    EXPECT_EQ(tracker.getGaussianStates().size(), 2);

    tracker.updateGaussians(gaussians2, 0.2);
    auto stats2 = tracker.getStats();
    EXPECT_EQ(stats2.num_lost, 1);
    EXPECT_EQ(tracker.getGaussianStates().size(), 2);

    tracker.updateGaussians(gaussians2, 0.3);
    EXPECT_EQ(tracker.getGaussianStates().size(), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
