#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat resizeFixedAspectRatio(const cv::Mat &cvMat, const double scaleFactor, const cv::Size &targetSize,
                               const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar &borderValue = cv::Scalar{0, 0, 0});


std::pair<cv::Mat, float> resizeFixedAspectRatio(const cv::Mat &image, const cv::Size &targetSize); 