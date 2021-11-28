#include "netCaffe.h"
using namespace std;

NetCaffe::NetCaffe(const std::string &caffeProto, const std::string &caffeTrainedModel, const bool useGPU,
                   const std::string &lastBlobName)
{
    use_gpu_ = useGPU;
    try
    {
        caffe::Caffe::set_mode(use_gpu_ ? caffe::Caffe::GPU : caffe::Caffe::CPU);
        net_.reset(new caffe::Net(caffeProto, caffe::TEST));
        net_->CopyTrainedLayersFrom(caffeTrainedModel);
        input_blob_ = net_->input_blobs()[0];
        output_blob_ = net_->blob_by_name("net_output").get();
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << __LINE__ << __FUNCTION__ << __FILE__ << std::endl;
    }
}

void NetCaffe::forward(const cv::Mat &input) const
{
    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels);
    cv::split(input, input_channels);
    net_->Forward();
}