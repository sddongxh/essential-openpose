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

// string caffeProto = "../models/pose/body_25/pose_deploy.prototxt";
// string caffeTrainedModel = "../models/pose/body_25/pose_iter_584000.caffemodel";

string caffeProto = "../models/pose/coco/pose_deploy_linevec.prototxt";
string caffeTrainedModel = "../models/pose/coco/pose_iter_440000.caffemodel";
// string image_path = "COCO_val2014_000000000192.jpg";
string  image_path = "/home/xihua/Pictures/10294980_10152127690781045_570701756731952736_o.jpg";
// std::vector<Array<float>> createArray(
//     const Matrix &inputData, const std::vector<double> &scaleInputToNetInputs,
//     const std::vector<Point<int>> &netInputSizes)
// {
//     assert(!inputData.empty() && inputData.channels() == 3);
//     assert(scaleInputToNetInputs.size() == netInputSizes.size());

//     return {};
// }

op::Array<float> preprocess(const cv::Mat &image, const cv::Size &baseSize, int normalize = 1)
{
    cv::Mat resized = resizeFixedAspectRatio(image, 1, baseSize);
    auto res = cvMatToArray(resized, normalize);
    return res;
}

int main(int argc, char **argv)
{
    cout << "essential" << endl;

    auto net = op::NetCaffe(caffeProto, caffeTrainedModel, 0);
    const cv::Mat cvImageToProcess = cv::imread(image_path);
    const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
    cout << imageToProcess.channels() << " " << imageToProcess.cols() << " " << imageToProcess.rows() << " " << endl;
    auto raw_image = cvImageToProcess;
    net.initializationOnThread();
    int base_width = 656;
    int base_height = 368;
    cv::Size baseSize(base_width, base_height);

    float scale = 0;
    Mat resized = getImage(raw_image, baseSize, &scale);
    auto inputData = cvMatToArray(resized, 1);
    // auto inputData = preprocess(cvImageToProcess, baseSize);
    auto start = std::chrono::steady_clock::now();

    BlobData *nms_out = createBlob_local(1, 56, POSE_MAX_PEOPLE + 1, 3);
    BlobData *input = createBlob_local(1, 57, baseSize.height, baseSize.width);

    for (int i = 0; i < 1; i++)
    {
        net.forwardPass({inputData});
        if (i % 100 == 0)
            cout << i << endl;

        // auto caffeNetOutputBlob = net.getOutputBlobArray();
        auto net_output_blob = net.getOutputLayer();
        const float *net_output_data_begin = net_output_blob->cpu_data<float>();

        cout << "N = " << net_output_blob->num() << "x" << net_output_blob->channels() << "x" << net_output_blob->height() << "x" << net_output_blob->width() << endl;

        BlobData *net_output = createBlob_local(net_output_blob->num(), net_output_blob->channels(), net_output_blob->height(), net_output_blob->width());
        memcpy(net_output->list, net_output_data_begin, net_output_blob->count() * sizeof(float));

        //把heatmap给resize到约定大小
        for (int i = 0; i < net_output->channels; ++i)
        {
            cv::Mat um(baseSize.height, baseSize.width, CV_32F, input->list + baseSize.height * baseSize.width * i);

            //featuremap的resize插值方法很有关系
            cv::resize(cv::Mat(net_output->height, net_output->width, CV_32F, net_output->list + net_output->width * net_output->height * i), um, baseSize, 0, 0, cv::INTER_CUBIC);
        }

        //获取每个feature map的局部极大值
        nms(input, nms_out, 0.05);
        vector<float> keypoints;
        vector<int> shape;

        //得到局部极大值后，根据PAFs、points做部件连接
        connectBodyPartsCpu(keypoints, input->list, nms_out->list, baseSize, POSE_MAX_PEOPLE, 9, 0.05, 3, 0.4, 1, shape);
        renderPoseKeypointsCpu(raw_image, keypoints, shape, 0.05, scale);
        printf("finish. save result to 'test_openpose.jpg', people: %d\n", shape[0]);
        imwrite("test_openpose.jpg", raw_image);
        releaseBlob_local(&net_output);
        releaseBlob_local(&input);
        releaseBlob_local(&nms_out);
    }
    auto last = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "Elapsed(ms)=" << last.count() << std::endl;

    //cv::waitKey(0);
    return 0;
}