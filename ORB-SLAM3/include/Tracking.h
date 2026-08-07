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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/types_c.h>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"
#include "Motion3D/DirectRgbdPoseValidator.h"
#include "Motion3D/FrozenReplayPacket.h"
#include "Motion3D/MatchedShamOracle.h"
#include "Motion3D/ReplayStateContract.h"

#include "GeometricCamera.h"

#include <mutex>
#include <unordered_set>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;
class ORBmatcher;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(
        const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp,
        string filename, const cv::Mat &staticMask = cv::Mat(),
        const cv::Mat &staticWeight = cv::Mat(),
        const Sophus::SE3f* relativePosePrior = nullptr,
        const Eigen::Matrix<float, 6, 6>* relativePoseInformation = nullptr,
        bool externalPosePriorShadowOnly = false,
        const Sophus::SE3f* externalPosePriorOracleTcw = nullptr,
        bool externalPosePriorOracleApply = false);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();
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
    void ConfigureAdaptiveMaskFeatures(bool enabled, float relaxation);
    float LastStaticMaskRatio() const;
    int LastAdaptiveFastThreshold() const;

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }
    bool CurrentFrameIsKeyFrame() const
    {
        return mpLastKeyFrame && mpLastKeyFrame->mnFrameId == mCurrentFrame.mnId;
    }
    bool WasExternalPosePriorUsed() const
    {
        return mbExternalPriorUsedLastFrame;
    }
    bool WasExternalPosePriorCandidate() const
    {
        return mbExternalPriorCandidateLastFrame;
    }
    bool WouldUseExternalPosePrior() const
    {
        return mbExternalPriorWouldUseLastFrame;
    }
    bool ExternalPosePriorConsensusPass() const
    {
        return mbExternalPriorConsensusPassLastFrame;
    }
    bool ExternalPosePriorDirectPass() const
    {
        return mbExternalPriorDirectPassLastFrame;
    }
    bool ExternalPosePriorCommonSupportPass() const
    {
        return mbExternalPriorCommonSupportPassLastFrame;
    }
    bool HasExternalPosePriorHypotheses() const
    {
        return mbExternalPriorHypothesesAvailableLastFrame;
    }
    bool ExternalPosePriorShadowTranslationActive() const
    {
        return mbExternalPriorShadowTranslationActive;
    }
    bool ExternalPosePriorShadowTranslationApplied() const
    {
        return mbExternalPriorShadowTranslationAppliedLastFrame;
    }
    int ExternalPosePriorShadowTranslationRemaining() const
    {
        return mnExternalPriorShadowTranslationRemaining;
    }
    float ExternalPosePriorShadowTranslationInnovation() const
    {
        return mfExternalPriorShadowTranslationInnovationLastFrame;
    }
    const std::string&
    ExternalPosePriorShadowTranslationStopReason() const
    {
        return mExternalPriorShadowTranslationStopReasonLastFrame;
    }
    bool ExternalPosePriorShadowTranslationFactorInjected() const
    {
        return mbExternalPriorShadowTranslationFactorInjectedLastFrame;
    }
    bool ExternalPosePriorPosteriorValid() const
    {
        return mbExternalPriorPosteriorValidLastFrame;
    }
    bool ExternalPosePriorPosteriorPass() const
    {
        return mbExternalPriorPosteriorPassLastFrame;
    }
    int ExternalPosePriorPosteriorStaticInliers() const
    {
        return mnExternalPriorPosteriorStaticInliersLastFrame;
    }
    int ExternalPosePriorPosteriorDynamicInliers() const
    {
        return mnExternalPriorPosteriorDynamicInliersLastFrame;
    }
    int ExternalPosePriorPosteriorCommonSupport() const
    {
        return mnExternalPriorPosteriorCommonSupportLastFrame;
    }
    float ExternalPosePriorPosteriorStaticScore() const
    {
        return mfExternalPriorPosteriorStaticScoreLastFrame;
    }
    float ExternalPosePriorPosteriorDynamicScore() const
    {
        return mfExternalPriorPosteriorDynamicScoreLastFrame;
    }
    bool ExternalPosePriorVelocityNeutralized() const
    {
        return mbExternalPriorVelocityNeutralizedLastFrame;
    }
    const Motion3D::MatchedShamOracleResult&
    ExternalPosePriorOracleResult() const
    {
        return mExternalPriorOracleResultLastFrame;
    }
    bool ExternalPosePriorOracleApplied() const
    {
        return mbExternalPriorOracleEnabled &&
            mbExternalPriorOracleApply &&
            mExternalPriorOracleResultLastFrame.valid &&
            mExternalPriorOracleResultLastFrame.prefers_dynamic &&
            mbExternalPriorUsedLastFrame;
    }
    Sophus::SE3f ExternalPosePriorStaticPose() const
    {
        return mExternalPriorStaticOptimizedPose;
    }
    Sophus::SE3f ExternalPosePriorDynamicPose() const
    {
        return mExternalPriorDynamicOptimizedPose;
    }
    Sophus::SE3f ExternalPosePriorPreviousPose() const
    {
        return mExternalPriorPreviousPose;
    }
    double ExternalPosePriorPreviousTimestamp() const
    {
        return mExternalPriorPreviousTimestamp;
    }
    int ExternalPosePriorStaticMatches() const
    {
        return mnExternalPriorStaticMatchesLastFrame;
    }
    int ExternalPosePriorDynamicMatches() const
    {
        return mnExternalPriorDynamicMatchesLastFrame;
    }
    int ExternalPosePriorStaticInliers() const
    {
        return mnExternalPriorStaticInliersLastFrame;
    }
    int ExternalPosePriorDynamicInliers() const
    {
        return mnExternalPriorDynamicInliersLastFrame;
    }
    int ExternalPosePriorCommonSupportCount() const
    {
        return mnExternalPriorCommonSupportCountLastFrame;
    }
    float ExternalPosePriorStaticCommonScore() const
    {
        return mfExternalPriorStaticCommonScoreLastFrame;
    }
    float ExternalPosePriorDynamicCommonScore() const
    {
        return mfExternalPriorDynamicCommonScoreLastFrame;
    }
    const Motion3D::DirectRgbdPoseValidator::Result&
    ExternalPosePriorDirectValidation() const
    {
        return mExternalPriorDirectValidationLastFrame;
    }
    const Motion3D::ReplayStateIdentity&
    ExternalPosePriorReplayStateIdentity() const
    {
        return mExternalPriorReplayStateIdentityLastFrame;
    }
    void EnableExternalPosePriorReplayCapture(bool enable)
    {
        mbExternalPriorReplayCaptureEnabled = enable;
    }
    void SetDeterministicReplayFreezeState(
        const Motion3D::ReplayFreezeState& state)
    {
        mDeterministicReplayFreezeState = state;
    }
    const Motion3D::FrozenReplayPacket&
    ExternalPosePriorReplayPacket() const
    {
        return mExternalPriorReplayPacketLastFrame;
    }
    const Motion3D::ReplayExecutionState&
    ExternalPosePriorReplayExecutionState() const
    {
        return mExternalPriorReplayExecutionStateLastFrame;
    }
    void RefreshExternalPosePriorReplayExecutionStateAfterFreeze();

    void SetInlierRatioLogPath(const std::string &log_path)
    {
        if (f_inlier_ratio.is_open())
            f_inlier_ratio.close();
        f_inlier_ratio.open(log_path.c_str());
        if (f_inlier_ratio.is_open()) {
            f_inlier_ratio << std::fixed << std::setprecision(6);
            f_inlier_ratio << "# timestamp inlier_ratio num_inliers num_total_matches\n";
        } else {
            std::cerr << "Cannot open inlier ratio log: " << log_path << std::endl;
        }
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;
    cv::Mat mImRGB;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<Eigen::Vector3f> mvIniColorRGB;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool ExternalPosePriorPassesGate(
        int staticInliers, int dynamicInliers,
        const Sophus::SE3f& staticPose,
        const Sophus::SE3f& dynamicPose) const;
    void ClearExternalPosePriorShadowTranslation();
    void StopExternalPosePriorShadowTranslation(
        const std::string& reason);
    void ActivateExternalPosePriorShadowTranslation(
        const Frame& staticFrame,
        const Sophus::SE3f& dynamicPose,
        const Sophus::SE3f& previousPose);
    bool AdvanceExternalPosePriorShadowTranslation(
        ORBmatcher& matcher, int searchThreshold, bool monocular);
    void ApplyExternalPosePriorShadowTranslation();
    void FinalizeExternalPosePriorShadowTranslationFrame();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;
    bool mbHasExternalRelativePosePrior{false};
    bool mbExternalPriorShadowOnly{false};
    bool mbExternalPriorOracleEnabled{false};
    bool mbExternalPriorOracleApply{false};
    bool mbExternalPriorUsedLastFrame{false};
    bool mbExternalPriorCandidateLastFrame{false};
    bool mbExternalPriorWouldUseLastFrame{false};
    bool mbExternalPriorConsensusPassLastFrame{false};
    bool mbExternalPriorDirectPassLastFrame{false};
    bool mbExternalPriorCommonSupportPassLastFrame{false};
    bool mbExternalPriorHypothesesAvailableLastFrame{false};
    int mnExternalPriorStaticMatchesLastFrame{-1};
    int mnExternalPriorDynamicMatchesLastFrame{-1};
    int mnExternalPriorStaticInliersLastFrame{-1};
    int mnExternalPriorDynamicInliersLastFrame{-1};
    int mnExternalPriorCommonSupportCountLastFrame{0};
    float mfExternalPriorStaticCommonScoreLastFrame{-1.0f};
    float mfExternalPriorDynamicCommonScoreLastFrame{-1.0f};
    Sophus::SE3f mExternalRelativePosePrior;
    Sophus::SE3f mExternalPriorOracleGroundTruthTcw;
    Motion3D::MatchedShamOracleResult
        mExternalPriorOracleResultLastFrame;
    Sophus::SE3f mExternalPriorStaticOptimizedPose;
    Sophus::SE3f mExternalPriorDynamicOptimizedPose;
    Sophus::SE3f mExternalPriorPreviousPose;
    double mExternalPriorPreviousTimestamp{0.0};
    int mnExternalPriorGateMinInlierGain{5};
    float mfExternalPriorGateMinInlierGainRatio{0.05f};
    int mnExternalPriorGateMaxStaticInliers{2147483647};
    float mfExternalPriorGateMinTranslationInnovation{0.0f};
    float mfExternalPriorGateMaxTranslationInnovation{1e6f};
    float mfExternalPriorGateMaxRotationInnovation{3.1415927f};
    float mfExternalPriorInformationScale{1.0f};
    bool mbExternalPriorUseDirectValidation{false};
    Motion3D::DirectRgbdPoseValidator::ScoreMode
        mExternalPriorDirectScoreMode{
            Motion3D::DirectRgbdPoseValidator::ScoreMode::Photometric};
    bool mbExternalPriorInjectLocalMap{true};
    bool mbExternalPriorInitializationOnly{false};
    bool mbExternalPriorRequireCommonSupportImprovement{false};
    bool mbExternalPriorBypassReliabilityGate{false};
    Motion3D::ReplayGatePolicy mExternalPriorGatePolicy{
        Motion3D::ReplayGatePolicy::LegacyConjunction};
    int mnExternalPriorShadowTranslationHorizon{0};
    float mfExternalPriorPosteriorMinScoreImprovement{0.0f};
    float mfExternalPriorShadowTranslationBlend{1.0f};
    float mfExternalPriorMaxStaticInformationLeverage{0.0f};
    Motion3D::TranslationLeverageMode
        mExternalPriorStaticInformationLeverageMode{
            Motion3D::TranslationLeverageMode::CapOnly};
    bool mbExternalPriorShadowTranslationActive{false};
    bool mbExternalPriorShadowTranslationPosteriorPending{false};
    bool mbExternalPriorShadowTranslationAppliedLastFrame{false};
    bool mbExternalPriorShadowTranslationFactorInjectedLastFrame{false};
    bool mbExternalPriorPosteriorValidLastFrame{false};
    bool mbExternalPriorPosteriorPassLastFrame{false};
    bool mbExternalPriorVelocityNeutralPending{false};
    bool mbExternalPriorVelocityNeutralizedLastFrame{false};
    int mnExternalPriorPosteriorStaticInliersLastFrame{-1};
    int mnExternalPriorPosteriorDynamicInliersLastFrame{-1};
    int mnExternalPriorPosteriorCommonSupportLastFrame{0};
    float mfExternalPriorPosteriorStaticScoreLastFrame{-1.0f};
    float mfExternalPriorPosteriorDynamicScoreLastFrame{-1.0f};
    int mnExternalPriorShadowTranslationRemaining{0};
    float mfExternalPriorShadowTranslationInnovationLastFrame{-1.0f};
    std::string mExternalPriorShadowTranslationStopReasonLastFrame{"none"};
    Frame mExternalPriorShadowLastFrame;
    Frame mExternalPriorShadowCurrentFrame;
    Sophus::SE3f mExternalPriorShadowVelocity;
    Sophus::SE3f mExternalPriorIncomingVelocity;
    Sophus::SE3f mExternalPriorShadowRelativePose;
    Eigen::Matrix<float, 6, 6>
        mExternalPriorShadowTranslationInformation =
            Eigen::Matrix<float, 6, 6>::Zero();
    KeyFrame* mpExternalPriorShadowReferenceKF{nullptr};
    Map* mpExternalPriorShadowMap{nullptr};
    Eigen::Matrix<float, 6, 6> mExternalRelativePoseInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    void ScoreExternalPosePriorCommonSupport(
        const vector<MapPoint*>& staticMatches,
        const vector<MapPoint*>& dynamicMatches,
        const Sophus::SE3f& staticPose,
        const Sophus::SE3f& dynamicPose);
    void ScoreExternalPosePriorDirectValidation(
        const Sophus::SE3f& staticPose,
        const Sophus::SE3f& dynamicPose,
        const Sophus::SE3f& previousPose);
    Sophus::SE3f RefreshPreviousFramePoseFromMap();
    void CaptureExternalPosePriorReplayStateIdentity(
        const vector<MapPoint*>& staticMatches,
        const vector<MapPoint*>& dynamicMatches,
        const Sophus::SE3f& previousPose);
    void BeginExternalPosePriorReplayPacket(
        const Sophus::SE3f& previousPose,
        const Sophus::SE3f& staticInitialPose,
        const Sophus::SE3f& dynamicInitialPose);
    void FinalizeExternalPosePriorReplayPacket();
    void CaptureExternalPosePriorReplayLocalMapSnapshot();
    void FinalizeExternalPosePriorReplayPosterior(
        const Sophus::SE3f& staticInitialPose,
        const Sophus::SE3f& dynamicPriorPose,
        const Eigen::Matrix<float, 6, 6>& dynamicInformation,
        const Frame& staticFrame, int staticInliers,
        const Frame& dynamicFrame, int dynamicInliers);
    void CaptureExternalPosePriorReplayExecutionState(Map* currentMap);
    bool ExternalPosePriorDirectValidationPasses() const;
    bool ExternalPosePriorCommonSupportPasses() const;
    bool ExternalPosePriorWouldUse(
        const Sophus::SE3f& staticPose,
        const Sophus::SE3f& dynamicPose,
        bool consensusPass,
        bool directPass,
        bool commonSupportPass) const;
    void EvaluateExternalPosePriorOracle(
        const Sophus::SE3f& staticPose,
        const Sophus::SE3f& dynamicPose);
    bool ExternalPosePriorSelectsDynamic() const;
    bool ExternalPosePriorInterventionEnabled() const;
    Motion3D::DirectRgbdPoseValidator mExternalPriorDirectValidator;
    Motion3D::DirectRgbdPoseValidator::Result
        mExternalPriorDirectValidationLastFrame;
    Motion3D::ReplayStateIdentity
        mExternalPriorReplayStateIdentityLastFrame;
    bool mbExternalPriorReplayCaptureEnabled{false};
    Motion3D::FrozenReplayPacket
        mExternalPriorReplayPacketLastFrame;
    Motion3D::ReplayExecutionState
        mExternalPriorReplayExecutionStateLastFrame;
    Motion3D::ReplayTrackingTrace
        mExternalPriorReplayTrackingTraceLastFrame;
    Motion3D::ReplayFreezeState mDeterministicReplayFreezeState;
    std::vector<Motion3D::ReplaySupportObservation>
        mExternalPriorCommonReplaySupportLastFrame;
    cv::Mat mExternalPriorDirectPreviousGray;
    cv::Mat mExternalPriorDirectPreviousDepth;
    cv::Mat mExternalPriorDirectPreviousStaticMask;
    cv::Mat mExternalPriorDirectCurrentDepth;
    double mExternalPriorDirectPreviousTimestamp{0.0};
    bool mbHasExternalPriorDirectPrevious{false};

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;

    ofstream f_inlier_ratio;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
