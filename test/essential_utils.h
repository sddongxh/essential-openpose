#pragma once
#include <opencv2/opencv.hpp>
#include <openpose/core/array.hpp>

cv::Mat resizeFixedAspectRatio(const cv::Mat &cvMat, const double scaleFactor, const cv::Size &targetSize,
                               const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar &borderValue = cv::Scalar{0, 0, 0});

op::Array<float> cvMatToArray(const cv::Mat &image, const int normalize = 1);

void uCharCvMatToFloatPtr(float *floatPtrImage, const cv::Mat &cvImage, const int normalize);