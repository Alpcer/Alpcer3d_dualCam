//
// Created by Zachary on 2026/7/20.
//
// =====================================================================
// 自定义 CUDA Kernel：在显存中直接将经纬度展开视差图转换为深度图
// =====================================================================
#include <opencv2/opencv.hpp>          // 💡 新增：解决 GpuMat 完整类型定义
#include <opencv2/core/cuda.hpp>       // 💡 新增：解决 GpuMat 到 PtrStepSz 的隐式转换
#include <opencv2/core/cuda/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "fisheye_stereo_pipeline_gpu.h"

__global__ void dispToDepthKernel(
        cv::cuda::PtrStepSz<float> disp_map,
        cv::cuda::PtrStepSz<float> depth_map,
        float cx_diff, float baseline,
        double alpha_min, double alpha_range)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= disp_map.cols || v >= disp_map.rows) return;

    float raw_disp = disp_map(v, u);

    // 1. 无效值与非正视差剔除
    if (isnan(raw_disp) || isinf(raw_disp) || raw_disp <= 0.0f) {
        depth_map(v, u) = 0.0f;
        return;
    }

    // 2. 扣除左右相机光心差异
    float true_disp = raw_disp - cx_diff;
    if (true_disp <= 0.0f) {
        depth_map(v, u) = 0.0f;
        return;
    }

    // 3. 按照经纬度展开几何计算左相机射线角度
    double alpha = alpha_min + (static_cast<double>(u) / (disp_map.cols - 1)) * alpha_range;
    double angle_x_left = acos(sin(alpha));

    // 4. 计算右相机射线角度
    double alpha_right = alpha_min + ((static_cast<double>(u) - true_disp) / (disp_map.cols - 1)) * alpha_range;
    double angle_x_right = CV_PI - acos(sin(alpha_right));

    // 5. 正弦定理三角剖分计算深度 r (除以1000将毫米转为米，如需保持毫米可去掉 /1000.0)
    double denom = sin(CV_PI - angle_x_left - angle_x_right);
    if (fabs(denom) < 1e-6) { // 防止出现平行光线导致除零或极大异常值
        depth_map(v, u) = 0.0f;
        return;
    }

    double r = baseline * sin(angle_x_right) / denom / 1000.0;
    depth_map(v, u) = static_cast<float>(r);
}

// 头文件中声明的包装函数的具体实现
void launchDispToDepthKernel(
        cv::cuda::GpuMat& disp_map,
        cv::cuda::GpuMat& depth_map,
        float cx_diff, float baseline,
        double alpha_min, double alpha_range)
{
    dim3 block(16, 16);
    dim3 grid((disp_map.cols + block.x - 1) / block.x,
              (disp_map.rows + block.y - 1) / block.y);

    // 隐式转换为 PtrStepSz 并启动 Kernel
    dispToDepthKernel<<<grid, block>>>(disp_map, depth_map, cx_diff, baseline, alpha_min, alpha_range);

    // 同步或错误检查
    cudaDeviceSynchronize();
}