#include <opencv2/opencv.hpp>
#include <iostream>
#include <openpose/core/array.hpp>
#include <openpose/core/datum.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/core/matrix.hpp>
#include <spdlog/spdlog.h>
#include "essential_utils.h"
#include <chrono>

using namespace std;
using namespace op;

string caffeProto = "../models/pose/body_25/pose_deploy.prototxt";
string caffeTrainedModel = "../models/pose/body_25/pose_iter_584000.caffemodel";
string image_path = "COCO_val2014_000000000192.jpg";

// std::vector<Array<float>> createArray(
//     const Matrix &inputData, const std::vector<double> &scaleInputToNetInputs,
//     const std::vector<Point<int>> &netInputSizes)
// {
//     assert(!inputData.empty() && inputData.channels() == 3);
//     assert(scaleInputToNetInputs.size() == netInputSizes.size());

//     return {};
// }

op::Array<float> preprocess(const cv::Mat &image)
{
    int base_width = 656;
    int base_height = 368;
    cv::Size baseSize(base_width, base_height);
    cv::Mat resized = resizeFixedAspectRatio(image, 2, baseSize);
    op::Array<float> res;

    return res;
}

int main(int argc, char **argv)
{
    cout << "essential" << endl;

    auto net = op::NetCaffe(caffeProto, caffeTrainedModel, 0);
    const cv::Mat cvImageToProcess = cv::imread(image_path);
    const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
    cout << imageToProcess.channels() << " " << imageToProcess.cols() << " " << imageToProcess.rows() << " " << endl;

    // auto input = createArray(imageToProcess)

    net.initializationOnThread();

    // net.forwardPass(imageToProcess);

    cv::Size baseSize(656, 368);
    // cv::imshow("original", cvImageToProcess);
    // cv::waitKey(0);

    cv::Mat resized = resizeFixedAspectRatio(cvImageToProcess, 2, baseSize);
    auto input = cvMatToArray(resized, 1);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++)
    {
        net.forwardPass({input});
        if (i % 100 == 0) cout << i << endl; 
    }
    auto last = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "Elapsed(ms)=" << last.count() << std::endl;
    //cv::waitKey(0);
    return 0;
}