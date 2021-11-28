
// Reference
// https://github.com/dlunion/EasyOpenPose/blob/master/easyOpenPose.cpp

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct BlobData
{
    int count;
    float *list;
    int num;
    int channels;
    int height;
    int width;
    int capacity_count; //保留空间的元素个数长度，字节数请 * sizeof(float)
};

BlobData *createBlob_local(int num, int channels, int height, int width);

BlobData *createEmptyBlobData();

void releaseBlob_local(BlobData **blob);


// Round functions
// Signed
template <typename T>
inline char charRound(const T a)
{
    return char(a + 0.5f);
}

template <typename T>
inline signed char sCharRound(const T a)
{
    return (signed char)(a + 0.5f);
}

template <typename T>
inline int intRound(const T a)
{
    return int(a + 0.5f);
}

template <typename T>
inline long longRound(const T a)
{
    return long(a + 0.5f);
}

template <typename T>
inline long long longLongRound(const T a)
{
    return (long long)(a + 0.5f);
}

// Unsigned
template <typename T>
inline unsigned char uCharRound(const T a)
{
    return (unsigned char)(a + 0.5f);
}

template <typename T>
inline unsigned int uIntRound(const T a)
{
    return (unsigned int)(a + 0.5f);
}

template <typename T>
inline unsigned long ulongRound(const T a)
{
    return (unsigned long)(a + 0.5f);
}

template <typename T>
inline unsigned long long uLongLongRound(const T a)
{
    return (unsigned long long)(a + 0.5f);
}

// Max/min functions
template <typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template <typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

template <class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
    return fastMin(max, fastMax(min, value));
}

//根据得到的结果，连接身体区域
void connectBodyPartsCpu(std::vector<float> &poseKeypoints, const float *const heatMapPtr, const float *const peaksPtr,
                         const cv::Size &heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
                         const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor, std::vector<int> &keypointShape);

void renderPoseKeypointsCpu(cv::Mat &frame, const std::vector<float> &poseKeypoints, std::vector<int> keyshape,
                            const float renderThreshold, float scale, const bool blendOriginalFrame = true);

#define POSE_COCO_COLORS_RENDER_GPU \
    255.f, 0.f, 85.f,               \
        255.f, 0.f, 0.f,            \
        255.f, 85.f, 0.f,           \
        255.f, 170.f, 0.f,          \
        255.f, 255.f, 0.f,          \
        170.f, 255.f, 0.f,          \
        85.f, 255.f, 0.f,           \
        0.f, 255.f, 0.f,            \
        0.f, 255.f, 85.f,           \
        0.f, 255.f, 170.f,          \
        0.f, 255.f, 255.f,          \
        0.f, 170.f, 255.f,          \
        0.f, 85.f, 255.f,           \
        0.f, 0.f, 255.f,            \
        255.f, 0.f, 170.f,          \
        170.f, 0.f, 255.f,          \
        255.f, 0.f, 255.f,          \
        85.f, 0.f, 255.f

const std::vector<float> POSE_COCO_COLORS_RENDER{POSE_COCO_COLORS_RENDER_GPU};
const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17};
cv::Mat getImage(const cv::Mat &im, cv::Size baseSize = cv::Size(656, 368), float *scale = 0);
void nms(BlobData *bottom_blob, BlobData *top_blob, float threshold);

#define POSE_MAX_PEOPLE 96