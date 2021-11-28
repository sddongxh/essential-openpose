#pragma once

#include <caffe/caffe.hpp>
class NetCaffe
{
public:
    NetCaffe(const std::string &caffeProto, const std::string &caffeTrainedModel, const bool useGPU = true,
             const std::string &lastBlobName = "net_output");

    virtual ~NetCaffe() {}
    void initShape(int C, int H, int W) {
        input_blob_->Reshape(1, C, H, W);
        net_->Reshape();  
    }
    void forward(const cv::Mat & input) const;
    caffe::Blob *getOutputLayer()
    {
        return output_blob_;
    }
    caffe::Blob * get_input_layer() {return input_blob_; }; 
    caffe::Blob * get_output_layer() {return output_blob_; }; 
private:
    std::unique_ptr<caffe::Net> net_;
    bool use_gpu_ = true;
    caffe::Blob *input_blob_ = nullptr;
    caffe::Blob *output_blob_ = nullptr;
};
