#include <gtest/gtest.h>

#include "LocalMapping.h"

namespace {

TEST(LocalMappingIdleTest, RequiresKeyFrameAcceptance)
{
    ORB_SLAM3::LocalMapping mapper(
        nullptr, nullptr, false, false);

    EXPECT_TRUE(mapper.IsIdle());

    mapper.SetAcceptKeyFrames(false);
    EXPECT_FALSE(mapper.IsIdle());

    mapper.SetAcceptKeyFrames(true);
    EXPECT_TRUE(mapper.IsIdle());
}

TEST(LocalMappingIdleTest, StoppedMapperIsIdle)
{
    ORB_SLAM3::LocalMapping mapper(
        nullptr, nullptr, false, false);

    mapper.SetAcceptKeyFrames(false);
    mapper.RequestStop();
    ASSERT_TRUE(mapper.Stop());
    EXPECT_TRUE(mapper.IsIdle());
}

}  // namespace
