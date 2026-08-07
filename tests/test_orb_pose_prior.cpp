#include <gtest/gtest.h>

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

TEST(OrbPosePriorTest, InformationReorderingPreservesQuadraticForm)
{
    Eigen::Matrix<float, 6, 6> factor =
        Eigen::Matrix<float, 6, 6>::Random();
    const Eigen::Matrix<float, 6, 6> sophus_information =
        factor.transpose() * factor;
    Eigen::Matrix<float, 6, 1> sophus_tangent;
    sophus_tangent << 0.1f, -0.2f, 0.3f, 0.04f, -0.05f, 0.06f;
    Eigen::Matrix<double, 6, 1> g2o_tangent;
    g2o_tangent <<
        sophus_tangent[3], sophus_tangent[4], sophus_tangent[5],
        sophus_tangent[0], sophus_tangent[1], sophus_tangent[2];

    const Eigen::Matrix<double, 6, 6> g2o_information =
        ORB_SLAM3::Optimizer::PosePriorInformationForG2O(
            sophus_information);
    const double sophus_cost =
        static_cast<double>(
            sophus_tangent.transpose() * sophus_information *
            sophus_tangent);
    const double g2o_cost =
        g2o_tangent.transpose() * g2o_information * g2o_tangent;
    EXPECT_NEAR(sophus_cost, g2o_cost, 1e-7);

    const Eigen::Matrix<float, 6, 6> round_trip =
        ORB_SLAM3::Optimizer::PoseInformationFromG2O(
            g2o_information);
    EXPECT_TRUE(round_trip.isApprox(sophus_information, 1e-6f));
}

TEST(OrbPosePriorTest, EdgeMovesPoseToPrior)
{
    g2o::SparseOptimizer optimizer;
    auto* linear_solver =
        new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    auto* block_solver = new g2o::BlockSolver_6_3(linear_solver);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(block_solver));

    auto* current = new g2o::VertexSE3Expmap();
    current->setId(0);
    current->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(current);

    const Sophus::SE3f prior =
        Sophus::SE3f::exp(
            (Eigen::Matrix<float, 6, 1>() <<
                 0.15f, -0.08f, 0.04f, 0.03f, -0.02f, 0.05f)
                .finished());
    auto* fixed_prior = new g2o::VertexSE3Expmap();
    fixed_prior->setId(1);
    fixed_prior->setEstimate(g2o::SE3Quat(
        prior.unit_quaternion().cast<double>(),
        prior.translation().cast<double>()));
    fixed_prior->setFixed(true);
    optimizer.addVertex(fixed_prior);

    auto* edge = new g2o::EdgeSE3();
    edge->setVertex(0, current);
    edge->setVertex(1, fixed_prior);
    edge->setMeasurement(g2o::SE3Quat());
    edge->setInformation(
        ORB_SLAM3::Optimizer::PosePriorInformationForG2O(
            Eigen::Matrix<float, 6, 6>::Identity() * 100.0f));
    optimizer.addEdge(edge);

    optimizer.initializeOptimization();
    ASSERT_GT(optimizer.optimize(10), 0);
    const g2o::SE3Quat optimized = current->estimate();
    const Sophus::SE3f optimized_pose(
        optimized.rotation().cast<float>(),
        optimized.translation().cast<float>());
    EXPECT_LT(
        (optimized_pose * prior.inverse()).translation().norm(), 1e-5f);
    EXPECT_LT(
        (optimized_pose * prior.inverse()).so3().log().norm(), 1e-5f);
}
