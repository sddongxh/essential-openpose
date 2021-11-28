#include "caffe_utils.h"

// void wrapInputLayer(const cv::Mat &image, caffe::Blob *input_layer)
// {
//     std::vector<cv::Mat> input_channels;
//     cv::split(image, input_channels);
// }

// void wrapInputLayer(std::vector<cv::Mat> *input_channels, caffe::Blob *input_layer)
// {
//     int width = input_layer->width();
//     int height = input_layer->height();
//     float *input_data = input_layer->mutable_cpu_data<float>();
//     for (int i = 0; i < input_layer->channels(); ++i)
//     {
//         cv::Mat channel(height, width, CV_32FC1, input_data);
//         input_channels->push_back(channel);
//         input_data += width * height;
//     }
// }

// Convert the input image to the input image format of the network.
cv::Mat preprocess(const cv::Mat &raw_image, const cv::Size & inputSize, int normalize)
{
    return cv::Mat{}; 
}