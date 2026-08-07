/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SYSTEM_H
#define SYSTEM_H


#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<cstdint>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Motion3D/MatchedShamOracle.h"
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"
#include "Motion3D/DirectRgbdPoseValidator.h"
#include "Motion3D/FrozenReplayPacket.h"
#include "Motion3D/ReplayStateContract.h"


namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;

class System
{
public:
    struct DeterministicReplayFreezeState {
        bool requested = false;
        bool local_mapping_idle_ack = false;
        bool local_mapping_stopped_ack = false;
        bool loop_closing_idle_ack = false;
        bool loop_closing_stopped_ack = false;
        bool gba_stopped_ack = false;
        std::uint64_t epoch = 0;

        bool complete() const
        {
            return requested &&
                local_mapping_idle_ack &&
                local_mapping_stopped_ack &&
                loop_closing_idle_ack &&
                loop_closing_stopped_ack &&
                gba_stopped_ack;
        }
    };

    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    // File type
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const int initFr = 0, const string &strSequence = std::string());

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Input staticMask (optional): CV_8UC1 mask where 1=static, 0=dynamic pixels
    // Input staticWeight (optional): CV_32FC1 soft weight where 1=static, 0=dynamic (for 3DGS)
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackRGBD(
        const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp,
        const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(),
        string filename="", const cv::Mat &staticMask = cv::Mat(),
        const cv::Mat &staticWeight = cv::Mat(),
        const Sophus::SE3f* relativePosePrior = nullptr,
        const Eigen::Matrix<float, 6, 6>* relativePoseInformation = nullptr,
        bool externalPosePriorShadowOnly = false,
        const Sophus::SE3f* externalPosePriorOracleTcw = nullptr,
        bool externalPosePriorOracleApply = false);

    void ConfigureAdaptiveMaskFeatures(bool enabled, float relaxation);
    float GetLastStaticMaskRatio() const;
    int GetLastAdaptiveFastThreshold() const;

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Wait until asynchronous local mapping has consumed its queue and is idle.
    // Intended for deterministic offline evaluation, not real-time operation.
    bool WaitForLocalMappingIdle(double timeoutSeconds);
    bool WaitForLoopClosingIdle(double timeoutSeconds);

    // Permanently freezes map mutation for deterministic replay capture.
    bool FreezeMapForDeterministicReplay(double timeoutSeconds);
    DeterministicReplayFreezeState GetDeterministicReplayFreezeState();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();
    bool isShutDown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    int GetMatchesInliers();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();
    cv::Mat GetTrackedDescriptors();
    std::vector<std::uint64_t> GetTrackedMapPointIds();
    std::vector<unsigned char> GetTrackedOutlierFlags();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    float GetImageScale();

    eSensor getSensorType() { return mSensor; }

    cv::Mat preprocessImage(const cv::Mat &src);

#ifdef REGISTER_TIMES
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

public:
    unsigned long GetNumKeyframes();
    bool CurrentFrameIsKeyFrame();
    bool WasExternalPosePriorUsed();
    bool WasExternalPosePriorCandidate();
    bool WouldUseExternalPosePrior();
    bool ExternalPosePriorConsensusPass();
    bool ExternalPosePriorDirectPass();
    bool ExternalPosePriorCommonSupportPass();
    bool HasExternalPosePriorHypotheses();
    bool ExternalPosePriorShadowTranslationActive();
    bool ExternalPosePriorShadowTranslationApplied();
    int ExternalPosePriorShadowTranslationRemaining();
    float ExternalPosePriorShadowTranslationInnovation();
    std::string ExternalPosePriorShadowTranslationStopReason();
    bool ExternalPosePriorShadowTranslationFactorInjected();
    bool ExternalPosePriorPosteriorValid();
    bool ExternalPosePriorPosteriorPass();
    int ExternalPosePriorPosteriorStaticInliers();
    int ExternalPosePriorPosteriorDynamicInliers();
    int ExternalPosePriorPosteriorCommonSupport();
    float ExternalPosePriorPosteriorStaticScore();
    float ExternalPosePriorPosteriorDynamicScore();
    bool ExternalPosePriorVelocityNeutralized();
    Motion3D::MatchedShamOracleResult ExternalPosePriorOracleResult();
    bool ExternalPosePriorOracleApplied();
    Sophus::SE3f ExternalPosePriorStaticPose();
    Sophus::SE3f ExternalPosePriorDynamicPose();
    Sophus::SE3f ExternalPosePriorPreviousPose();
    double ExternalPosePriorPreviousTimestamp();
    void ConfigureExternalPosePriorGate(
        int minInlierGain, float minInlierGainRatio,
        int maxStaticInliers, float minTranslationInnovation,
        float maxTranslationInnovation,
        float maxRotationInnovation, float priorInformationScale,
        bool useDirectValidation, int directValidationScoreMode,
        bool injectLocalMap, bool initializationOnly,
        bool requireCommonSupportImprovement,
        bool bypassReliabilityGate, int gatePolicy);
    void ConfigureExternalPosePriorShadowTranslation(
        int horizon, float posteriorMinScoreImprovement,
        float translationBlend, float maxStaticInformationLeverage,
        int staticInformationLeverageMode);
    int ExternalPosePriorStaticMatches();
    int ExternalPosePriorDynamicMatches();
    int ExternalPosePriorStaticInliers();
    int ExternalPosePriorDynamicInliers();
    int ExternalPosePriorCommonSupportCount();
    float ExternalPosePriorStaticCommonScore();
    float ExternalPosePriorDynamicCommonScore();
    Motion3D::DirectRgbdPoseValidator::Result
        ExternalPosePriorDirectValidation();
    Motion3D::ReplayStateIdentity
        ExternalPosePriorReplayStateIdentity();
    void EnableExternalPosePriorReplayCapture(bool enable);
    Motion3D::FrozenReplayPacket ExternalPosePriorReplayPacket();
    void RefreshExternalPosePriorReplayExecutionState();
    Motion3D::ReplayExecutionState
        ExternalPosePriorReplayExecutionState();
    Atlas* getAtlas();
    Tracking* getTracker();
    LocalMapping* getLocalMapper();
    LoopClosing* getLoopCloser();
    Settings* getSettings();
    FrameDrawer* getFrameDrawer();
    MapDrawer* getMapDrawer();

private:

    void SaveAtlas(int type);
    bool LoadAtlas(int type);

    string CalculateCheckSum(string filename, int type);

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;
    DeterministicReplayFreezeState mDeterministicReplayFreezeState;

    // Shutdown flag
    bool mbShutDown;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    cv::Mat mTrackedDescriptors;
    std::vector<std::uint64_t> mTrackedMapPointIds;
    std::vector<unsigned char> mTrackedOutlierFlags;
    std::mutex mMutexState;

    //
    string mStrLoadAtlasFromFile;
    string mStrSaveAtlasToFile;

    string mStrVocabularyFilePath;

    Settings* settings_;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
