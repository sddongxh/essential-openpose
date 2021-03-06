#include <opencv2/opencv.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
// #include "essential_utils.h"
#include <chrono>
#include "netCaffe.h"
using namespace std;
#include "caffe_utils.h"
#include "essential_utils.h"
// string caffeProto = "../models/pose/body_25/pose_deploy.prototxt";
// string caffeTrainedModel = "../models/pose/body_25/pose_iter_584000.caffemodel";

string caffeProto = "../models/pose/coco/pose_deploy_linevec.prototxt";
string caffeTrainedModel = "../models/pose/coco/pose_iter_440000.caffemodel";
// string image_path = "COCO_val2014_000000000192.jpg";
string image_path = "/home/xihua/Pictures/10294980_10152127690781045_570701756731952736_o.jpg";

//     const Matrix &inputData, const std::vector<double> &scaleInputToNetInputs,
//     const std::vector<Point<int>> &netInputSizes)
// {
//     assert(!inputData.empty() && inputData.channels() == 3);
//     assert(scaleInputToNetInputs.size() == netInputSizes.size());

//     return {};
// }

// op::Array<float> preprocess(const cv::Mat &image, const cv::Size &baseSize, int normalize = 1)
// {
//     cv::Mat resized = resizeFixedAspectRatio(image, 1, baseSize);
//     auto res = cvMatToArray(resized, normalize);
//     return res;
// }

bool use_gpu = true;

int main(int argc, char **argv)
{

    cout << "essential" << endl;

    auto net = std::make_shared<NetCaffe>(caffeProto, caffeTrainedModel, use_gpu);
    int base_width = 656;
    int base_height = 368;
    cv::Size baseSize(base_width, base_height);
    net->initShape(3, base_height, base_width);
    cv::Mat raw_image = cv::imread(image_path);
    auto [resized_image, scale] = resizeFixedAspectRatio(raw_image, baseSize);

    // cv::imshow("raw", raw_image);
    // cv::waitKey(0);
    // cv::imshow("resized", resized_image);
    // cv::waitKey(0);

    cv::Mat normalized_image;
    resized_image.convertTo(normalized_image, CV_32F, 1.0 / 256, -0.5);
    auto t_start = std::chrono::steady_clock::now();
    // for (int i = 0; i < 10; i++)
    // {
    net->forward(normalized_image);
    // cout << i << endl;
    // }
    auto last = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start);
    std::cout << "Elapsed(ms)=" << last.count() << std::endl;

    auto net_output_blob = net->get_output_layer();
    int C = net_output_blob->channels();
    BlobData *net_output = createBlob_local(net_output_blob->num(), C, net_output_blob->height(), net_output_blob->width());
    const float *net_output_data_begin = net_output_blob->cpu_data<float>();
    memcpy(net_output->list, net_output_data_begin, net_output_blob->count() * sizeof(float));

    cout << "Output dims = " << net_output_blob->num() << "x" << C << "x" << net_output_blob->height() << "x" << net_output_blob->width() << endl;
    BlobData *nms_out = createBlob_local(1, C - 1, POSE_MAX_PEOPLE + 1, 3);
    BlobData *input = createBlob_local(1, C, baseSize.height, baseSize.width);

    //???heatmap???resize???????????????
    for (int i = 0; i < net_output->channels; ++i)
    {
        cv::Mat um(baseSize.height, baseSize.width, CV_32F, input->list + baseSize.height * baseSize.width * i);

        //featuremap???resize????????????????????????
        cv::resize(cv::Mat(net_output->height, net_output->width, CV_32F, net_output->list + net_output->width * net_output->height * i), um, baseSize, 0, 0, cv::INTER_CUBIC);
    }
    //????????????feature map??????????????????
    nms(input, nms_out, 0.05);
    vector<float> keypoints;
    vector<int> shape;
    //?????????????????????????????????PAFs???points???????????????
    connectBodyPartsCpu(keypoints, input->list, nms_out->list, baseSize, POSE_MAX_PEOPLE, 9, 0.05, 3, 0.4, 1, shape);
    renderPoseKeypointsCpu(raw_image, keypoints, shape, 0.05, );
    // renderPoseKeypointsCpu(raw_image, keypoints, shape, 0.05, 1 / scale);
    // printf("finish. save result to 'test_openpose.jpg', people: %d\n", shape[0]);
    // imwrite("test_openpose.jpg", raw_image);

    releaseBlob_local(&net_output);
    releaseBlob_local(&input);
    releaseBlob_local(&nms_out);

    return 0;
}
