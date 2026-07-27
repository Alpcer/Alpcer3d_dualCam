#include "IGEVTensorRTEngine.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

// CUDA错误检查宏
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t e = (err);                          \
        if (e != cudaSuccess) {                         \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(e) \
                      << " at line " << __LINE__ << std::endl; \
            assert(false);                              \
        }                                               \
    } while(0)

// TensorRT 日志类
class TrtLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} g_trt_logger;

// ===================== 构造与析构 =====================
IGEVTensorRTEngine::IGEVTensorRTEngine(const std::string& engine_path, int max_batch, cv::Size image_size)
    : max_batch_(max_batch), image_size_(image_size)
{
    // 1. 加载引擎
    if (!loadEngine(engine_path))
        throw std::runtime_error("Failed to load TensorRT engine");

    // 2. 核心修复：先创建原生CUDA流，再包装为OpenCV流
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    cv_stream_ = cv::cuda::StreamAccessor::wrapStream(cuda_stream_);

    // 3. 分配单帧显存
    const size_t input_sz = 3 * image_size_.height * image_size_.width * sizeof(float);
    const size_t output_sz = image_size_.height * image_size_.width * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input_left_,   input_sz));
    CUDA_CHECK(cudaMalloc(&d_input_right_,  input_sz));
    CUDA_CHECK(cudaMalloc(&d_output_depth_, output_sz));

    // 4. 分配批量显存
    CUDA_CHECK(cudaMalloc(&d_input_left_batch_,   max_batch_ * input_sz));
    CUDA_CHECK(cudaMalloc(&d_input_right_batch_,  max_batch_ * input_sz));
    CUDA_CHECK(cudaMalloc(&d_output_depth_batch_, max_batch_ * output_sz));

    // 5. 预分配CPU输出内存
    cpu_depth_.create(image_size_.height, image_size_.width, CV_32FC1);

    std::cout << "Engine init success. Max batch = " << max_batch_ << std::endl;
}

IGEVTensorRTEngine::~IGEVTensorRTEngine()
{
    // 逆序释放显存
    CUDA_CHECK(cudaFree(d_output_depth_batch_));
    CUDA_CHECK(cudaFree(d_input_right_batch_));
    CUDA_CHECK(cudaFree(d_input_left_batch_));

    CUDA_CHECK(cudaFree(d_output_depth_));
    CUDA_CHECK(cudaFree(d_input_right_));
    CUDA_CHECK(cudaFree(d_input_left_));

    // 销毁 CUDA 流
    if (cuda_stream_) cudaStreamDestroy(cuda_stream_);

    // ========== 核心修正：TensorRT 10.x 用 delete 替代 destroy() ==========
    if (context_) delete context_;
    if (engine_)  delete engine_;
    if (runtime_) delete runtime_;

    std::cout << "Engine resources released." << std::endl;
}


// ===================== 引擎加载 =====================
bool IGEVTensorRTEngine::loadEngine(const std::string& engine_path)
{
    std::ifstream fin(engine_path, std::ios::binary | std::ios::ate);
    if (!fin.is_open()) {
        std::cerr << "Cannot open engine: " << engine_path << std::endl;
        return false;
    }
    const size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<char> engine_buf(file_size);
    fin.read(engine_buf.data(), file_size);
    fin.close();

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) return false;

    engine_ = runtime_->deserializeCudaEngine(engine_buf.data(), file_size);
    if (!engine_) return false;

    context_ = engine_->createExecutionContext();
    if (!context_) return false;

    return true;
}

// ===================== GPU预处理 =====================
void IGEVTensorRTEngine::preprocess_gpu(const cv::cuda::GpuMat& src, float* dst)
{
    // 1. 缩放到目标尺寸（OpenCV接口传cv_stream_）
//     cv::cuda::GpuMat resized;
//     cv::cuda::resize(src, resized, cv::Size(image_size_.width, image_size_.height), 0, 0, cv::INTER_LINEAR, cv_stream_);

    // 2. BGR转RGB
    cv::cuda::GpuMat rgb_mat;
    cv::cuda::cvtColor(src, rgb_mat, cv::COLOR_BGR2RGB, 0, cv_stream_);

    // 3. uint8转float32，归一化到[0,1]
    cv::cuda::GpuMat float_mat;
    rgb_mat.convertTo(float_mat, CV_32FC3, 1.0f / 255.0f, 0.0f, cv_stream_);

    // 4. HWC → CHW 排布
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(float_mat, channels, cv_stream_);

    const size_t ch_sz = image_size_.height * image_size_.width * sizeof(float);
    // 原生CUDA拷贝传cuda_stream_
    CUDA_CHECK(cudaMemcpyAsync(dst,                      channels[0].data, ch_sz, cudaMemcpyDeviceToDevice, cuda_stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst + image_size_.height * image_size_.width,      channels[1].data, ch_sz, cudaMemcpyDeviceToDevice, cuda_stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst + 2 * image_size_.height * image_size_.width,  channels[2].data, ch_sz, cudaMemcpyDeviceToDevice, cuda_stream_));
}

// ===================== 单帧推理 =====================
cv::Mat IGEVTensorRTEngine::infer(const cv::Mat& cpu_left, const cv::Mat& cpu_right)
{
    // A. 异步上传原图（OpenCV接口传cv_stream_）
    d_raw_left_.upload(cpu_left, cv_stream_);
    d_raw_right_.upload(cpu_right, cv_stream_);

    // B. GPU预处理
    preprocess_gpu(d_raw_left_,  d_input_left_);
    preprocess_gpu(d_raw_right_, d_input_right_);

    // C. 绑定张量地址+设置形状
    context_->setTensorAddress("left_image",  d_input_left_);
    context_->setTensorAddress("right_image", d_input_right_);
    context_->setTensorAddress("disparity",  d_output_depth_);
    context_->setInputShape("left_image",  nvinfer1::Dims4{1, 3, image_size_.height, image_size_.width});
    context_->setInputShape("right_image", nvinfer1::Dims4{1, 3, image_size_.height, image_size_.width});

    // D. TensorRT异步推理（原生CUDA流）
    context_->enqueueV3(cuda_stream_);

    // E. 异步D2H回传结果
    CUDA_CHECK(cudaMemcpyAsync(cpu_depth_.data, d_output_depth_,
                               image_size_.height * image_size_.width * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_stream_));

    // F. 全程仅同步一次
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    return cpu_depth_.clone();
}

// ===================== 原生Batch批量推理 =====================
std::vector<cv::Mat> IGEVTensorRTEngine::inferBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& batch_input)
{
    const int B = static_cast<int>(batch_input.size());
    if (B <= 0 || B > max_batch_) {
        std::cerr << "Batch size " << B << " out of range [1, " << max_batch_ << "]" << std::endl;
        return {};
    }

    const size_t input_step  = 3 * image_size_.height * image_size_.width;
    const size_t output_step = image_size_.height * image_size_.width;

    // 1. 批量上传+预处理，写入偏移显存
    for (int i = 0; i < B; ++i)
    {
        float* left_dst  = d_input_left_batch_  + i * input_step;
        float* right_dst = d_input_right_batch_ + i * input_step;

        d_raw_left_.upload(batch_input[i].first,  cv_stream_);
        d_raw_right_.upload(batch_input[i].second, cv_stream_);

        preprocess_gpu(d_raw_left_,  left_dst);
        preprocess_gpu(d_raw_right_, right_dst);
    }

    // 2. 绑定批量显存
    context_->setTensorAddress("left_image",  d_input_left_batch_);
    context_->setTensorAddress("right_image", d_input_right_batch_);
    context_->setTensorAddress("disparity",  d_output_depth_batch_);

    // 3. 设置动态Batch形状
    context_->setInputShape("left_image",  nvinfer1::Dims4{B, 3, image_size_.height, image_size_.width});
    context_->setInputShape("right_image", nvinfer1::Dims4{B, 3, image_size_.height, image_size_.width});

    // 4. 单次enqueue完成整批推理（RTX5090核心加速点）
    context_->enqueueV3(cuda_stream_);

    // 5. 批量异步D2H回传
    std::vector<cv::Mat> outputs(B);
    for (int i = 0; i < B; ++i)
    {
        outputs[i].create(image_size_.height, image_size_.width, CV_32FC1);
        const float* src = d_output_depth_batch_ + i * output_step;
        CUDA_CHECK(cudaMemcpyAsync(outputs[i].data, src,
                                   output_step * sizeof(float),
                                   cudaMemcpyDeviceToHost, cuda_stream_));
    }

    // 6. 统一同步一次
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    return outputs;
}

cv::cuda::GpuMat IGEVTensorRTEngine::inferGpu(const cv::cuda::GpuMat& gpu_left, const cv::cuda::GpuMat& gpu_right)
{
    // 1. 直接在 GPU 上进行预处理 (BGR2RGB, float32 转换, HWC -> CHW)[cite: 4]
    preprocess_gpu(gpu_left,  d_input_left_);
    preprocess_gpu(gpu_right, d_input_right_);

    // 2. 绑定 GPU 显存地址并设置维度[cite: 4]
    context_->setTensorAddress("left_image",  d_input_left_);
    context_->setTensorAddress("right_image", d_input_right_);
    context_->setTensorAddress("disparity",  d_output_depth_);
    context_->setInputShape("left_image",  nvinfer1::Dims4{1, 3, image_size_.height, image_size_.width});
    context_->setInputShape("right_image", nvinfer1::Dims4{1, 3, image_size_.height, image_size_.width});

    // 3. 触发异步 TensorRT 推理[cite: 4]
    context_->enqueueV3(cuda_stream_);

    // 4. 同步等待推理完成[cite: 4]
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    // 5. 将输出显存指针 d_output_depth_ 直接包装为 GpuMat 零拷贝返回[cite: 4]
    return cv::cuda::GpuMat(image_size_.height, image_size_.width, CV_32FC1, d_output_depth_);
}
