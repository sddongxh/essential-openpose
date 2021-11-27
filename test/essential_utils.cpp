#include "essential_utils.h"

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

op::Array<float> cvMatToArray(const cv::Mat &image, const int normalize)
{
    assert(!image.empty() && image.channels() == 3);
    op::Array<float> a;
    a.reset({1, 3, image.rows, image.cols}); //NCHW
    uCharCvMatToFloatPtr(
        a.getPtr(), image, normalize);

    return a;
}

void uCharCvMatToFloatPtr(float *floatPtrImage, const cv::Mat &cvImage, const int normalize)
{
    try
    {
        // float* (deep net format): C x H x W
        // cv::Mat (OpenCV format): H x W x C
        const int width = cvImage.cols;
        const int height = cvImage.rows;
        const int channels = cvImage.channels();

        const auto *const originFramePtr = cvImage.data; // cv::Mat.data is always uchar
        for (auto c = 0; c < channels; c++)
        {
            const auto floatPtrImageOffsetC = c * height;
            for (auto y = 0; y < height; y++)
            {
                const auto floatPtrImageOffsetY = (floatPtrImageOffsetC + y) * width;
                const auto originFramePtrOffsetY = y * width;
                for (auto x = 0; x < width; x++)
                    floatPtrImage[floatPtrImageOffsetY + x] = float(
                        originFramePtr[(originFramePtrOffsetY + x) * channels + c]);
            }
        }
        if (normalize == 1)
        {

            // floatPtrImage wrapped as cv::Mat
            // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and
            // other optimized instruction sets).
            // In addition, the following if statement does not copy the pointer to a cv::Mat, just wrapps it.
            cv::Mat floatPtrImageCvWrapper(height * width * 3, 1, CV_32FC1, floatPtrImage); // CV_32FC3 warns about https://github.com/opencv/opencv/issues/16739
            floatPtrImageCvWrapper = floatPtrImageCvWrapper * (1 / 256.f) - 0.5f;
        }
        // ResNet
        else if (normalize == 2)
        {
            const int imageArea = width * height;
            const std::array<float, 3> means{102.9801, 115.9465, 122.7717};
            for (auto i = 0; i < 3; i++)
            {
                cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i * imageArea);
                floatPtrImageCvWrapper = floatPtrImageCvWrapper - means[i];
            }
        }
        // DenseNet
        else if (normalize == 2)
        {
            const auto scaleDenseNet = 0.017;
            const int imageArea = width * height;
            const std::array<float, 3> means{103.94f, 116.78f, 123.68f};
            for (auto i = 0; i < 3; i++)
            {
                cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i * imageArea);
                floatPtrImageCvWrapper = scaleDenseNet * (floatPtrImageCvWrapper - means[i]);
            }
        }
        // Unknown
        else if (normalize != 0)
            std::cerr << "Unknown normalization value (" << std::to_string(normalize) + ")." << __LINE__ << __FUNCTION__ << __FILE__ << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << __LINE__ << __FUNCTION__ << __FILE__ << std::endl;
    }
}