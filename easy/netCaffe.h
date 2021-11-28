#pragma once

#include <caffe/caffe.hpp>
class NetCaffe
{
public:
    NetCaffe(const std::string &caffeProto, const std::string &caffeTrainedModel, const bool useGPU = true,
             const std::string &lastBlobName = "net_output");

    virtual ~NetCaffe() {}
    void initShape(int C, int H, int W)
    {
        input_blob_->Reshape(1, C, H, W);
        net_->Reshape();
    }
    void forward(const cv::Mat &input) const;
    caffe::Blob *getOutputLayer()
    {
        return output_blob_;
    }
    caffe::Blob *get_input_layer() { return input_blob_; };
    caffe::Blob *get_output_layer() { return net_->blob_by_name("net_output").get(); };

private:
    std::unique_ptr<caffe::Net> net_;
    bool use_gpu_ = true;
    caffe::Blob *input_blob_ = nullptr;
    caffe::Blob *output_blob_ = nullptr;
    void wrapInputLayer(std::vector<cv::Mat> *input_channels) const
    {
        auto input_layer = net_->input_blobs()[0];

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
};
