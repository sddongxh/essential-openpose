#pragma once
#include <vector>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include "img_utils.h"
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
// void wrapInputLayer(std::vector<cv::Mat> *input_channels, caffe::Blob *input_layer);
// void wrapInputLayer(const cv::Mat &image, caffe::Blob *input_layer);

/* Convert the input image to the input image format of the network. */
cv::Mat preprocess(const cv::Mat &raw_image, const cv::Size &inputSize, int normalize);