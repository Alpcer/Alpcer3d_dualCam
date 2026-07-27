#pragma once
#include <string>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <NvInfer.h>

// 模型输入尺寸，与导出ONNX时保持一致
constexpr int IGEV_H = 1024;
constexpr int IGEV_W = 1280;

class IGEVTensorRTEngine
{
public:
    /**
     * @param engine_path TensorRT引擎文件路径
     * @param max_batch 最大支持批量数，必须与引擎构建时max batch一致
     */
    IGEVTensorRTEngine(const std::string& engine_path, int max_batch = 4, cv::Size image_size = cv::Size(IGEV_W, IGEV_H));
    ~IGEVTensorRTEngine();

    // 禁止拷贝赋值
    IGEVTensorRTEngine(const IGEVTensorRTEngine&) = delete;
    IGEVTensorRTEngine& operator=(const IGEVTensorRTEngine&) = delete;

    /// 单帧双目推理
    cv::Mat infer(const cv::Mat& cpu_left, const cv::Mat& cpu_right);

    /// 原生Batch批量推理（真正利用RTX5090算力）
    std::vector<cv::Mat> inferBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& batch_input);

    /// 全 GPU 推理接口：输入和输出均为显存图像，无 H2D/D2H 拷贝
    cv::cuda::GpuMat inferGpu(const cv::cuda::GpuMat& gpu_left, const cv::cuda::GpuMat& gpu_right);

private:
    /// GPU端预处理：BGR uint8 → RGB CHW float32 + 归一化
    void preprocess_gpu(const cv::cuda::GpuMat& src, float* dst);
    /// 加载并反序列化引擎
    bool loadEngine(const std::string& engine_path);

    // TensorRT 核心对象
    nvinfer1::IRuntime* runtime_       = nullptr;
    nvinfer1::ICudaEngine* engine_     = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // 双流设计：底层共用同一条物理流
    cudaStream_t cuda_stream_ = nullptr;   // 原生CUDA流：给TensorRT、原生CUDA API使用
    cv::cuda::Stream cv_stream_;           // OpenCV包装流：给OpenCV CUDA算子使用

    // 单帧GPU缓存
    cv::cuda::GpuMat d_raw_left_;
    cv::cuda::GpuMat d_raw_right_;
    float* d_input_left_   = nullptr;
    float* d_input_right_  = nullptr;
    float* d_output_depth_ = nullptr;

    // 批量GPU显存
    float* d_input_left_batch_   = nullptr;
    float* d_input_right_batch_  = nullptr;
    float* d_output_depth_batch_ = nullptr;
    int max_batch_ = 1;

    // 预分配CPU输出缓存，避免每次推理重复malloc
    cv::Mat cpu_depth_;

    //图片规格
    cv::Size image_size_;
};
