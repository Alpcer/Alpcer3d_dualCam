//
// Created by Zachary on 2026/4/22.
// 完美适配 amd64 + NVIDIA GPU (CUDA 12.8)
// 替代原 Rockchip MPP + RGA 逻辑，采用 nvJPEG + NPP (NVIDIA Performance Primitives) 硬件加速
// 全局显存/内存单例预分配，零分配开销，零报错
//

#ifndef LIVOX_COLOR_CUDA_NVJPEG_DECODER_H
#define LIVOX_COLOR_CUDA_NVJPEG_DECODER_H

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

// nvJPEG 错误检查宏
#define CHECK_NVJPEG(call) \
    do { \
        nvjpegStatus_t status = call; \
        if (status != NVJPEG_STATUS_SUCCESS) { \
            std::cerr << "nvJPEG Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

class CudaNvjpegDecoder {
public:
    CudaNvjpegDecoder(int target_width, int target_height)
            : width_(target_width), height_(target_height) {

        // 1. 创建 CUDA 流，实现异步并行处理
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream." << std::endl;
            return;
        }

        // 2. 初始化 nvJPEG 句柄和状态
        if (nvjpegCreateSimple(&nvjpeg_handle_) != NVJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to create nvJPEG handle." << std::endl;
            return;
        }
        if (nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_) != NVJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to create nvJPEG state." << std::endl;
            return;
        }

        // 3. 预分配显存 (Device Memory) 和 锁页内存 (Pinned Host Memory)
        allocate_buffers();

        initialized_ = true;
        std::cout << "NVIDIA nvJPEG Decoder initialized (CUDA 12.8). Hardware acceleration ready." << std::endl;
    }

    ~CudaNvjpegDecoder() {
        std::cout << "CudaNvjpegDecoder destroying..." << std::endl;

        if (stream_) {
            cudaStreamSynchronize(stream_);
        }

        // 销毁 nvJPEG 资源
        if (nvjpeg_state_) {
            nvjpegJpegStateDestroy(nvjpeg_state_);
            nvjpeg_state_ = nullptr;
        }
        if (nvjpeg_handle_) {
            nvjpegDestroy(nvjpeg_handle_);
            nvjpeg_handle_ = nullptr;
        }

        // 释放显存 (GPU)
        if (d_decoded_bgr_) {
            cudaFree(d_decoded_bgr_);
            d_decoded_bgr_ = nullptr;
        }
        if (d_target_bgr_) {
            cudaFree(d_target_bgr_);
            d_target_bgr_ = nullptr;
        }

        // 释放 CPU 锁页内存 (Host Pinned)
        if (h_bgr_ptr_) {
            cudaFreeHost(h_bgr_ptr_);
            h_bgr_ptr_ = nullptr;
        }

        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        std::cout << "CudaNvjpegDecoder destroyed safely." << std::endl;
    }

    /**
     * @brief 基础硬件解码接口
     */
    bool decode(const uint8_t* jpeg_data, size_t size, cv::Mat& out_mat) {
        if (!initialized_ || !jpeg_data || size == 0) return false;

        // --- 1. 获取 JPEG 图像的原始分辨率 ---
        int widths[NVJPEG_MAX_COMPONENT];
        int heights[NVJPEG_MAX_COMPONENT];
        int channels;
        nvjpegChromaSubsampling_t subsampling;

        CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, jpeg_data, size, &channels, &subsampling, widths, heights));
        int actual_width = widths[0];
        int actual_height = heights[0];

        // --- 2. 配置解码目标显存地址 ---
        nvjpegImage_t img_desc;
        img_desc.channel[0] = d_decoded_bgr_;
        img_desc.pitch[0] = (unsigned int)(max_alloc_width_ * 3);

        // --- 3. GPU 硬件解码 ---
        // 【已修复】修改宏名字大小写为 NVJPEG_OUTPUT_BGRI
        CHECK_NVJPEG(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, jpeg_data, size,
                                  NVJPEG_OUTPUT_BGRI, &img_desc, stream_));

        // --- 4. 图像缩放处理 ---
        uint8_t* final_gpu_buffer = d_decoded_bgr_;
        if (actual_width != width_ || actual_height != height_) {
            if (!process_with_npp_resize(d_decoded_bgr_, actual_width, actual_height, max_alloc_width_ * 3,
                                         d_target_bgr_, width_, height_, width_ * 3)) {
                return false;
            }
            final_gpu_buffer = d_target_bgr_;
        }



        // --- 5. 异步将结果从 GPU 显存拷贝回 CPU 锁页内存 ---
        CHECK_CUDA(cudaMemcpyAsync(h_bgr_ptr_, final_gpu_buffer, width_ * height_ * 3,
                                   cudaMemcpyDeviceToHost, stream_));

        // --- 6. 同步等待 GPU 任务全部完成 ---
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        // 包装为 OpenCV Mat (零内存拷贝，直接绑定 h_bgr_ptr_)
        out_mat = cv::Mat(height_, width_, CV_8UC3, h_bgr_ptr_);
        return true;
    }

    /**
     * @brief 解码并裁剪的重载版本
     */
    bool decode(const uint8_t* jpeg_data, size_t size, cv::Rect crop_rect, cv::Mat& out_mat, bool scale_to_target = true) {
        if (!initialized_ || !jpeg_data || size == 0) return false;

        int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], channels;
        nvjpegChromaSubsampling_t subsampling;
        CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, jpeg_data, size, &channels, &subsampling, widths, heights));
        int actual_width = widths[0];
        int actual_height = heights[0];

        if (crop_rect.x < 0 || crop_rect.y < 0 ||
            crop_rect.x + crop_rect.width > actual_width ||
            crop_rect.y + crop_rect.height > actual_height ||
            crop_rect.width <= 0 || crop_rect.height <= 0) {
            std::cerr << "Invalid crop rect! Frame size: " << actual_width << "x" << actual_height
                      << ", Crop rect: [" << crop_rect.x << ", " << crop_rect.y
                      << ", " << crop_rect.width << "x" << crop_rect.height << "]" << std::endl;
            return false;
        }

        // 1. GPU 解码到显存
        nvjpegImage_t img_desc;
        img_desc.channel[0] = d_decoded_bgr_;
        img_desc.pitch[0] = (unsigned int)(max_alloc_width_ * 3);

        // 【已修复】修改宏名字大小写为 NVJPEG_OUTPUT_BGRI
        CHECK_NVJPEG(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, jpeg_data, size,
                                  NVJPEG_OUTPUT_BGRI, &img_desc, stream_));

        // 2. 使用 NPP 进行裁剪与缩放
        int out_w = scale_to_target ? width_ : crop_rect.width;
        int out_h = scale_to_target ? height_ : crop_rect.height;

        if (!process_with_npp_crop_scale(d_decoded_bgr_, actual_width, actual_height, max_alloc_width_ * 3,
                                         d_target_bgr_, out_w, out_h, out_w * 3, crop_rect)) {
            return false;
        }

        // 3. 拷贝回主机
        CHECK_CUDA(cudaMemcpyAsync(h_bgr_ptr_, d_target_bgr_, out_w * out_h * 3,
                                   cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        out_mat = cv::Mat(out_h, out_w, CV_8UC3, h_bgr_ptr_);
        return true;
    }

    /**
 * @brief 全 GPU 解码：将 nvJPEG/NPP 的显存结果直接包装为 GpuMat (零显存拷贝)[cite: 2]
 */
    bool decodeToGpu(const uint8_t* jpeg_data, size_t size, cv::cuda::GpuMat& out_gpu_mat) {
        if (!initialized_ || !jpeg_data || size == 0) return false;

        int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], channels;
        nvjpegChromaSubsampling_t subsampling;
        CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, jpeg_data, size, &channels, &subsampling, widths, heights));
        int actual_width = widths[0];
        int actual_height = heights[0];

        printf("actual_width:%d actual_height:%d\n", actual_width, actual_height);

        nvjpegImage_t img_desc;
        img_desc.channel[0] = d_decoded_bgr_;
        img_desc.pitch[0] = (unsigned int)max_alloc_pitch_;

        // 确保该流上的上一个解码任务已经处理完毕，释放显存工作区
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        CHECK_NVJPEG(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, jpeg_data, size,
                                  NVJPEG_OUTPUT_BGRI, &img_desc, stream_));

        uint8_t* final_gpu_buffer = d_decoded_bgr_;
        size_t step = img_desc.pitch[0];

        if (actual_width != width_ || actual_height != height_) {
            step = ((width_ * 3 + 511) / 512) * 512;
            if (!process_with_npp_resize(d_decoded_bgr_, actual_width, actual_height, img_desc.pitch[0],
                                         d_target_bgr_, width_, height_, step)) {
                return false;
            }
            final_gpu_buffer = d_target_bgr_;
        }

        // 关键：等待流内 GPU 任务完成，确保后续在外部使用时显存已准备就绪
        CHECK_CUDA(cudaStreamSynchronize(stream_));

        // 使用用户显存指针直接构造 GpuMat，不发生任何内存拷贝
        out_gpu_mat = cv::cuda::GpuMat(height_, width_, CV_8UC3, final_gpu_buffer, step);
        return true;
    }

private:
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    nvjpegJpegState_t nvjpeg_state_ = nullptr;
    cudaStream_t stream_ = nullptr;

    int width_, height_;
    int max_alloc_width_ = 3840;
    int max_alloc_height_ = 1080;
    size_t max_alloc_pitch_;
    bool initialized_ = false;

    uint8_t* d_decoded_bgr_ = nullptr;
    uint8_t* d_target_bgr_ = nullptr;
    void* h_bgr_ptr_ = nullptr;

    void allocate_buffers() {
        // 确保宽度至少是 3840，并按需对齐（这里以 256 对齐为例，保证显存访问高效）
        max_alloc_width_ = std::max(width_, max_alloc_width_);
        max_alloc_height_ = std::max(height_, max_alloc_height_);

        // 调试：在 allocate_buffers() 之前调用
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "Before Alloc: Free Mem " << free_mem / 1024 / 1024 << " MB" << std::endl;

        // 计算对齐后的 Pitch
        size_t pitch = ((max_alloc_width_ * 3 + 511) / 512) * 512;
        size_t max_bgr_size = pitch * max_alloc_height_;
        size_t target_bgr_size = width_ * height_ * 3;

        // 重新分配
        cudaFree(d_decoded_bgr_);
        cudaFree(d_target_bgr_);
        cudaFreeHost(h_bgr_ptr_);

        cudaMalloc(&d_decoded_bgr_, max_bgr_size);
        cudaMalloc(&d_target_bgr_, target_bgr_size);
        cudaMallocHost(&h_bgr_ptr_, target_bgr_size);

        // 更新成员变量，供 decode 使用
        max_alloc_pitch_ = pitch;
    }

    /**
     * @brief 【已修复】构造 NPP 上下文并绑定我们定义的 CUDA Stream
     */
    inline NppStreamContext get_npp_context() {
        NppStreamContext npp_ctx;
        nppGetStreamContext(&npp_ctx); // 初始化当前设备的默认配置
        npp_ctx.hStream = stream_;     // 绑定当前解码器特有的 CUDA 流
        return npp_ctx;
    }

    bool process_with_npp_resize(const uint8_t* d_src, int src_w, int src_h, int src_step,
                                 uint8_t* d_dst, int dst_w, int dst_h, int dst_step) {
        NppiSize src_size = {src_w, src_h};
        NppiRect src_roi = {0, 0, src_w, src_h};
        NppiRect dst_roi = {0, 0, dst_w, dst_h};

        // 【已修复】使用 get_npp_context() 获取正确的结构体传参
        NppStatus status = nppiResize_8u_C3R_Ctx(
                d_src, src_step, src_size, src_roi,
                d_dst, dst_step, dst_size(dst_w, dst_h), dst_roi,
                NPPI_INTER_LINEAR, get_npp_context());

        if (status != NPP_SUCCESS) {
            std::cerr << "NPP resize failed with error code: " << status << std::endl;
            return false;
        }
        return true;
    }

    bool process_with_npp_crop_scale(const uint8_t* d_src, int src_w, int src_h, int src_step,
                                     uint8_t* d_dst, int dst_w, int dst_h, int dst_step,
                                     cv::Rect crop_rect) {
        NppiSize src_size = {src_w, src_h};
        NppiRect src_roi = {crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height};
        NppiRect dst_roi = {0, 0, dst_w, dst_h};

        // 【已修复】使用 get_npp_context() 获取正确的结构体传参
        NppStatus status = nppiResize_8u_C3R_Ctx(
                d_src, src_step, src_size, src_roi,
                d_dst, dst_step, dst_size(dst_w, dst_h), dst_roi,
                NPPI_INTER_LINEAR, get_npp_context());

        if (status != NPP_SUCCESS) {
            std::cerr << "NPP crop & resize failed with error code: " << status << std::endl;
            return false;
        }
        return true;
    }

    inline NppiSize dst_size(int w, int h) {
        NppiSize s = {w, h};
        return s;
    }
};

#endif //LIVOX_COLOR_CUDA_NVJPEG_DECODER_H