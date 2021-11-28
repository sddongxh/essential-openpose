#include <openpose/net/netCaffe.hpp>
#include <numeric> // std::accumulate
#include <atomic>
#include <caffe/net.hpp>
#include <spdlog/spdlog.h>
#include <openpose/core/common.hpp>
// #include <openpose/utilities/fileSystem.hpp>
// #include <openpose/utilities/standard.hpp>
#include <filesystem>
namespace fs = std::filesystem;

template <typename T>
bool vectorsAreEqual(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
{
    try
    {
        if (vectorA.size() != vectorB.size())
            return false;
        else
        {
            for (auto i = 0u; i < vectorA.size(); i++)
                if (vectorA[i] != vectorB[i])
                    return false;
            return true;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << __LINE__ << __FUNCTION__ << __FILE__ << std::endl;
        return false;
    }
}

namespace op
{
    std::mutex sMutexNetCaffe;
    std::atomic<bool> sGoogleLoggingInitialized{false};

    struct NetCaffe::ImplNetCaffe
    {
        // Init with constructor
        const int mGpuId;
        const std::string mCaffeProto;
        const std::string mCaffeTrainedModel;
        const std::string mLastBlobName;
        std::vector<int> mNetInputSize4D;

        std::unique_ptr<caffe::Net> upCaffeNet;
        boost::shared_ptr<caffe::TBlob<float>> spOutputBlob;

        ImplNetCaffe(const std::string &caffeProto, const std::string &caffeTrainedModel, const int gpuId,
                     const bool enableGoogleLogging, const std::string &lastBlobName) : mGpuId{gpuId},
                                                                                        mCaffeProto{caffeProto},
                                                                                        mCaffeTrainedModel{caffeTrainedModel},
                                                                                        mLastBlobName{lastBlobName}
        {

            try
            {
                const std::string message{".\nPossible causes:\n"
                                          "\t1. Not downloading the OpenPose trained models.\n"
                                          "\t2. Not running OpenPose from the root directory (i.e., where the `model` folder is located, but do not move the `model` folder!). E.g.,\n"
                                          "\t\tRight example for the Windows portable binary: `cd {OpenPose_root_path}; bin/openpose.exe`\n"
                                          "\t\tWrong example for the Windows portable binary: `cd {OpenPose_root_path}/bin; openpose.exe`\n"
                                          "\t3. Using paths with spaces."};
                if (!fs::exists(mCaffeProto))
                    spdlog::error("Prototxt file not found: {}{}{}{}{}", mCaffeProto, message, __LINE__, __FUNCTION__, __FILE__);
                if (!fs::exists(mCaffeTrainedModel))
                    spdlog::error("Caffe trained model file not found: {}{}{}" + mCaffeTrainedModel + message,
                                  __LINE__, __FUNCTION__, __FILE__);
            }

            catch (const std::exception &e)
            {
                spdlog::error("{}{}{}{}", e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    };
    inline void reshapeNetCaffe(caffe::Net *caffeNet, const std::vector<int> &dimensions)
    {
        try
        {
            caffeNet->blobs()[0]->Reshape(dimensions);
            caffeNet->Reshape();
        }
        catch (const std::exception &e)
        {
            spdlog::error("{}{}{}{}", e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NetCaffe::NetCaffe(const std::string &caffeProto, const std::string &caffeTrainedModel, const int gpuId,
                       const bool enableGoogleLogging, const std::string &lastBlobName)
        : upImpl{
              new ImplNetCaffe{caffeProto, caffeTrainedModel, gpuId, enableGoogleLogging,
                               lastBlobName}}
    {
    }
    NetCaffe::~NetCaffe()
    {
    }

    void NetCaffe::initializationOnThread()
    {
        try
        {
            caffe::Caffe::set_mode(caffe::Caffe::GPU);
            caffe::Caffe::SetDevice(upImpl->mGpuId);
            upImpl->upCaffeNet.reset(new caffe::Net{upImpl->mCaffeProto, caffe::TEST});
            upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
            // Set spOutputBlob
            upImpl->spOutputBlob = boost::static_pointer_cast<caffe::TBlob<float>>(
                upImpl->upCaffeNet->blob_by_name(upImpl->mLastBlobName));
            // Sanity check
            if (upImpl->spOutputBlob == nullptr)
                error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: " + upImpl->mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception &e)
        {
            spdlog::error("{}{}{}{}", e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
    void NetCaffe::forwardPass(const Array<float> &inputData) const
    {
        try
        {
            if (inputData.empty())
                spdlog::error("The Array inputData cannot be empty. {}{}{}", __LINE__, __FUNCTION__, __FILE__);
            if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
                spdlog::error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width]. {}{}{}",
                              __LINE__, __FUNCTION__, __FILE__);
            // Reshape Caffe net if required
            if (!vectorsAreEqual(upImpl->mNetInputSize4D, inputData.getSize()))
            {
                upImpl->mNetInputSize4D = inputData.getSize();
                reshapeNetCaffe(upImpl->upCaffeNet.get(), inputData.getSize());
            }

            // Copy frame data to GPU memory
            auto *gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data<float>();
            cudaMemcpy(gpuImagePtr, inputData.getConstPtr(), inputData.getVolume() * sizeof(float),
                       cudaMemcpyHostToDevice);
            // Perform deep network forward pass
            upImpl->upCaffeNet->ForwardFrom(0);
        }
        catch (const std::exception &e)
        {
            spdlog::error("{}{}{}{}", e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::shared_ptr<ArrayCpuGpu<float>> NetCaffe::getOutputBlobArray() const
    {
        try
        {
            return std::make_shared<ArrayCpuGpu<float>>(upImpl->spOutputBlob.get());
        }
        catch (const std::exception &e)
        {
            spdlog::error("{}{}{}{}", e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
    caffe::Blob *NetCaffe::getOutputLayer() const
    {
        return upImpl->spOutputBlob.get();
    }
}