#include "netCaffe.h"

void wrapInputLayer(std::vector<cv::Mat> *input_channels, caffe::Blob *input_layer)
{
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data<float>();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

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
        output_blob_ = net_->output_blobs()[0];
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << __LINE__ << __FUNCTION__ << __FILE__ << std::endl;
    }
}

void NetCaffe::forward(const cv::Mat &input) const
{
    std::vector<cv::Mat> input_channels; 
    cv::split(input, input_channels); 
    wrapInputLayer(&input_channels, input_blob_);
    net_->ForwardFrom(0);
}