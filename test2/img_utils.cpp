#include "img_utils.h"

cv::Mat resizeFixedAspectRatio(const cv::Mat &cvMat, const double scaleFactor, const cv::Size &targetSize,
                               const int borderMode, const cv::Scalar &borderValue)
{
    cv::Mat resizedCvMat;
    const cv::Size cvTargetSize{targetSize.width, targetSize.height};
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    M.at<double>(0, 0) = scaleFactor;
    M.at<double>(1, 1) = scaleFactor;
    if (scaleFactor != 1. || cvTargetSize != cvMat.size())
        cv::warpAffine(cvMat, resizedCvMat, M, cvTargetSize,
                       (scaleFactor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA), borderMode, borderValue);
    else
        cvMat.copyTo(resizedCvMat);
    return resizedCvMat;
}

std::pair<cv::Mat, float> resizeFixedAspectRatio(const cv::Mat &image, const cv::Size &targetSize)
{
    int w = targetSize.width, h = targetSize.height;
    float scale = h / (float)image.rows;
    int nh = h;
    int nw = int(image.cols * scale);
    if (nw > w)
    {
        nw = w;
        scale = w / (float)image.cols;
        nh = int(image.rows * scale);
    }
    cv::Rect rect(0, 0, nw, nh);
    cv::Mat bk = cv::Mat::zeros(h, w, CV_8UC3);
    cv::resize(image, bk(rect), cv::Size(nw, nh));
    return {bk, scale};
}
