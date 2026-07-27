#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>  // GPU Remap 必须引用
#include <opencv2/core/cuda/common.hpp> // PtrStepSz 必须引用
#include "cuda_decoder.h"
#include "IGEVTensorRTEngine.h"

// 声明一个纯 C++ 签名的包装函数，这样头文件不包含任何 CUDA 特殊语法
void launchDispToDepthKernel(
        cv::cuda::GpuMat& disp_map,
        cv::cuda::GpuMat& depth_map,
        float cx_diff, float baseline,
        double alpha_min, double alpha_range);

class FisheyeStereoPipelineGpu {
public:
    FisheyeStereoPipelineGpu(
            const std::string& trt_engine_path,
            int jpeg_w, int jpeg_h,
            const cv::Matx33d& K_left, const cv::Vec4d& D_left,
            const cv::Matx33d& K_right, const cv::Vec4d& D_right,
            double fov_h, double fov_v,
            float baseline) // <-- 新增 baseline 参数 [source: 1]
            : raw_width_(jpeg_w), raw_height_(jpeg_h),
              single_eye_size_(jpeg_w / 2, jpeg_h),
              igev_size_(512, 512),
              baseline_(baseline),
              fov_h_(fov_h), fov_v_(fov_v)
    {
        // 自动计算左右相机光心 X 的差异 (比硬编码 cols/2 更精确)
        float cx_left  = static_cast<float>(K_left(0, 2));
        float cx_right = static_cast<float>(K_right(0, 2));
//        cx_diff_ = cx_left - cx_right;

        decoder_ = std::make_unique<CudaNvjpegDecoder>(raw_width_, raw_height_);
        {
//            int panorama_s;
//            double f_fish = K_left(0, 0); // 鱼眼原始焦距
//            double k1 = D_left[0], k2 = D_left[1], k3 = D_left[2], k4 = D_left[3];
//            double tmp_theta = (180.0 / 2.0) * CV_PI / 180.0;
//            double tmp_theta2 = tmp_theta * tmp_theta;
//            double tmp_theta_d = tmp_theta * (1.0 + k1 * tmp_theta2 + k2 * tmp_theta2 * tmp_theta2 + k3 * tmp_theta2 * tmp_theta2 * tmp_theta2 + k4 * tmp_theta2 * tmp_theta2 * tmp_theta2 * tmp_theta2);
//            double tmp_r_fov = f_fish * tmp_theta_d;
//            panorama_s = tmp_r_fov * 2;
//            panorama_s -= panorama_s % 32;
//            igev_size_ = cv::Size(panorama_s, panorama_s);
            std::cout << "igev_size_:" << igev_size_ << std::endl;
        }
        igev_engine_ = std::make_unique<IGEVTensorRTEngine>(trt_engine_path, 1, igev_size_);

        // 1. 在 CPU 生成映射表
        cv::Mat mapx_l, mapy_l, mapx_r, mapy_r, inv_mapx_l, inv_mapy_l;
        buildFisheyeUnwrapMap(K_left, D_left, igev_size_, fov_h, fov_v, mapx_l, mapy_l);
        buildFisheyeUnwrapMap(K_right, D_right, igev_size_, fov_h, fov_v, mapx_r, mapy_r);
        buildFisheyeWrapMap(K_left, D_left, single_eye_size_, igev_size_, fov_h, fov_v, inv_mapx_l, inv_mapy_l);

        // 2. 核心：初始化时一次性将 Remap 映射表上传至 GPU 显存，供后续频繁使用
        d_mapx_left_.upload(mapx_l);
        d_mapy_left_.upload(mapy_l);
        d_mapx_right_.upload(mapx_r);
        d_mapy_right_.upload(mapy_r);
        d_inv_mapx_left_.upload(inv_mapx_l);
        d_inv_mapy_left_.upload(inv_mapy_l);
    }

    /**
     * @brief 全流程 GPU 内执行，最后仅执行一次内存回传
     */
    bool processGpu(const uint8_t* jpeg_data, size_t size,
                    cv::Mat& out_left_fisheye,
                    cv::Mat& out_right_fisheye,
                    cv::Mat& out_result_fisheye)
    {
        // Step 1: GPU 硬件解码，结果直接留在显存中 (零拷贝)
        cv::cuda::GpuMat d_decoded_mat;
        if (!decoder_->decodeToGpu(jpeg_data, size, d_decoded_mat)) {
            std::cerr << "GPU 解码失败！" << std::endl;
            return false;
        }
        std::cout << "`" << std::endl;
        // Step 2: GPU 内居中切开 (OpenCV GpuMat 针对 cv::Rect 切片是零显存拷贝，仅创建 Header)
        cv::cuda::GpuMat d_left_fisheye  = d_decoded_mat(cv::Rect(0, 0, single_eye_size_.width, single_eye_size_.height));
        cv::cuda::GpuMat d_right_fisheye = d_decoded_mat(cv::Rect(single_eye_size_.width, 0, single_eye_size_.width, single_eye_size_.height));
        if (d_left_fisheye.empty()) { std::cerr << "Error: Left crop is empty!" << std::endl; return false; }
        std::cout << "2" << std::endl;
        // Step 3: GPU 内调用 cv::cuda::remap 得到左右眼经纬度展开图
        cv::cuda::GpuMat d_left_unwrapped, d_right_unwrapped;
        cv::cuda::remap(d_left_fisheye, d_left_unwrapped, d_mapx_left_, d_mapy_left_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::cuda::remap(d_right_fisheye, d_right_unwrapped, d_mapx_right_, d_mapy_right_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        if (d_left_unwrapped.empty()) { std::cerr << "Error: Unwrap remap output is empty!" << std::endl; return false; }
        std::cout << "3" << std::endl;
        // Step 4: 送入 TensorRT 引擎在 GPU 内直接推理
        cv::cuda::GpuMat d_result_unwrapped = igev_engine_->inferGpu(d_left_unwrapped, d_right_unwrapped);
        if (d_result_unwrapped.empty()) { std::cerr << "Error: IGEV Inference output empty!" << std::endl; return false; }
        std::cout << "4" << std::endl;
        // =========================================================================
        // Step 4.5 (新增): GPU 内视差图转深度图 (全并行，无需 D2H 回传) [source: 1]
        // =========================================================================
        cv::cuda::GpuMat d_result_unwrapped_depth(d_result_unwrapped.size(), CV_32FC1);
        launchDispToDepthKernel(
                d_result_unwrapped,
                d_result_unwrapped_depth,
                cx_diff_, baseline_,
                -fov_h_ / 2.0, fov_h_
        );
        std::cout << "5" << std::endl;
        // Step 5: GPU 内逆向 Remap 将结果（已替换为深度图 d_result_unwrapped_depth）还原为最终的鱼眼几何图 [source: 1]
        cv::cuda::GpuMat d_result_fisheye;
        cv::cuda::remap(d_result_unwrapped_depth, d_result_fisheye, d_inv_mapx_left_, d_inv_mapy_left_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        if (d_result_fisheye.empty()) { std::cerr << "Error: Wrap remap output is empty!" << std::endl; return false; }
        std::cout << "6" << std::endl;
        //todo 添加TURBO Colormap将最终鱼眼深度图可视化

        // Step 6: 最终仅在此处进行一次性的 D2H (Device to Host) 数据取回！
        d_left_fisheye.download(out_left_fisheye);
        d_right_fisheye.download(out_right_fisheye);
        d_result_fisheye.download(out_result_fisheye);
        if (out_left_fisheye.empty()) { std::cerr << "Error:Output out_left_fisheye is empty!" << std::endl; return false; }
        if (out_right_fisheye.empty()) { std::cerr << "Error:Output out_right_fisheye is empty!" << std::endl; return false; }
        if (out_result_fisheye.empty()) { std::cerr << "Error:Output out_result_fisheye is empty!" << std::endl; return false; }
        std::cout << "7" << std::endl;
        if (print) {
            print = false;
            cv::Mat left_unwrapped, right_unwrapped, result_unwrapped, result_unwrapped_depth;
            d_left_unwrapped.download(left_unwrapped);
            d_right_unwrapped.download(right_unwrapped);
            d_result_unwrapped.download(result_unwrapped);
            d_result_unwrapped_depth.download(result_unwrapped_depth); // 调试新增：下载中间展开深度图 [source: 1]

            int tu = 324, tv = 324;
            float raw_disp = result_unwrapped.at<float>(tv, tu);
            std::cout << "raw_disp:" << raw_disp << ", depth:" << (result_unwrapped_depth.at<float>(tv, tu)) << std::endl;
            {
                double alpha_min = -CV_PI / 2.0;
                double alpha_range = CV_PI;
                double beta_min = -CV_PI / 2.0;
                double beta_range = CV_PI;

                double beta = beta_min + (static_cast<double>(tv) / (result_unwrapped.rows - 1)) * beta_range;
                double cos_beta = std::cos(beta);
                double sin_beta = std::sin(beta);

                float true_disp = raw_disp - cx_diff_;
                double alpha = alpha_min + (static_cast<double>(tu) / (result_unwrapped.cols - 1)) * alpha_range;
                double cos_alpha = std::cos(alpha);
                double sin_alpha = std::sin(alpha);

                double angle_x_left = std::acos(sin_alpha);
                double alpha_right = alpha_min + ((static_cast<double>(tu)-true_disp) / (result_unwrapped.cols - 1)) * alpha_range;
                double angle_x_right = CV_PI - std::acos(std::sin(alpha_right));
                double r = baseline_*std::sin(angle_x_right)/std::sin(CV_PI - angle_x_left - angle_x_right)/1000.0;
                std::cout << "cx_diff_:" << cx_diff_ << ", baseline_:" << baseline_ << ", r:" << r << std::endl;
            }


            cv::Mat disp_result_unwrapped, disp_result_unwrapped_depth;
            cv::normalize(result_unwrapped, disp_result_unwrapped, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::imwrite("/factory_tools/tmp/test/disp_result_unwrapped.png", disp_result_unwrapped);

            cv::imwrite("/factory_tools/tmp/test/left_unwrapped.png", left_unwrapped);
            cv::imwrite("/factory_tools/tmp/test/right_unwrapped.png", right_unwrapped);
            cv::imwrite("/factory_tools/tmp/test/result_unwrapped.png", result_unwrapped);
            cv::imwrite("/factory_tools/tmp/test/result_unwrapped.tiff", result_unwrapped);
            cv::imwrite("/factory_tools/tmp/test/result_unwrapped_depth.tiff", result_unwrapped_depth);
            cv::imwrite("/factory_tools/tmp/test/out_left_fisheye.png", out_left_fisheye);
            cv::imwrite("/factory_tools/tmp/test/out_result_fisheye.tiff", out_result_fisheye);
            cv::imwrite("/factory_tools/tmp/test/out_right_fisheye.png", out_right_fisheye);
        }

        return true;
    }

private:
    bool print = true;
    int raw_width_, raw_height_;
    cv::Size single_eye_size_;
    cv::Size igev_size_;

    float baseline_;     // 新增：双目基线 [source: 1]
    float cx_diff_ = 0.0f;      // 新增：左右相机光心差 [source: 1]
    double fov_h_, fov_v_; // 新增：保存视场角 [source: 1]

    std::unique_ptr<CudaNvjpegDecoder> decoder_;
    std::unique_ptr<IGEVTensorRTEngine> igev_engine_;

    // GPU 端的映射表缓存
    cv::cuda::GpuMat d_mapx_left_, d_mapy_left_;
    cv::cuda::GpuMat d_mapx_right_, d_mapy_right_;
    cv::cuda::GpuMat d_inv_mapx_left_, d_inv_mapy_left_;

    /**
 * @brief 构建鱼眼图像横向经纬度展开（Equirectangular）的 Remap 映射表
 * * @param K 鱼眼相机内参矩阵 (cv::Matx33d)
 * @param D 鱼眼相机畸变系数 (cv::Vec4d -> k1, k2, k3, k4)
 * @param outSize 目标展开图的尺寸 (cv::Size)
 * @param fovHorizontal 水平展开总视角 (弧度, 例如 M_PI 表示 180度)
 * @param fovVertical 垂直展开总视角 (弧度, 例如 M_PI/2 表示 90度)
 * @param mapx 输出的 X 映射表 (CV_32FC1)
 * @param mapy 输出的 Y 映射表 (CV_32FC1)
 */
    void buildFisheyeUnwrapMap(
            const cv::Matx33d& K,
            const cv::Vec4d& D,
            const cv::Size& outSize,
            double fovHorizontal,
            double fovVertical,
            cv::Mat& mapx,
            cv::Mat& mapy)
    {
        mapx.create(outSize, CV_32FC1);
        mapy.create(outSize, CV_32FC1);

        // 计算展开角度范围
        double alpha_min = -fovHorizontal / 2.0;
        double alpha_range = fovHorizontal;
        double beta_min = -fovVertical / 2.0;
        double beta_range = fovVertical;

        std::vector<cv::Point3d> ray3D_list;
        ray3D_list.reserve(outSize.width * outSize.height);

        for (int v = 0; v < outSize.height; ++v) {
            double beta = beta_min + (static_cast<double>(v) / (outSize.height - 1)) * beta_range;
            double cos_beta = std::cos(beta);
            double sin_beta = std::sin(beta);

            for (int u = 0; u < outSize.width; ++u) {
                double alpha = alpha_min + (static_cast<double>(u) / (outSize.width - 1)) * alpha_range;
                double cos_alpha = std::cos(alpha);
                double sin_alpha = std::sin(alpha);

                double X = sin_alpha;
                double Y = cos_alpha * sin_beta;
                double Z = cos_alpha * cos_beta;

                ray3D_list.push_back(cv::Point3d(X, Y, Z));
            }
        }

        std::vector<cv::Point2d> projected_points;
        cv::fisheye::projectPoints(ray3D_list, projected_points, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0), K, D);

        // 填充至 remap 映射表
        int idx = 0;
        for (int v = 0; v < outSize.height; ++v) {
            float* ptr_x = mapx.ptr<float>(v);
            float* ptr_y = mapy.ptr<float>(v);
            for (int u = 0; u < outSize.width; ++u) {
                ptr_x[u] = static_cast<float>(projected_points[idx].x);
                ptr_y[u] = static_cast<float>(projected_points[idx].y);
                idx++;
            }
        }
    }

    /**
 * @brief 构建将经纬度展开图 (Equirectangular) 逆向还原为鱼眼图的 Remap 映射表
 * @param K 左眼鱼眼相机内参矩阵 (cv::Matx33d)
 * @param D 左眼鱼眼相机畸变系数 (cv::Vec4d)
 * @param fisheyeSize 目标鱼眼图的尺寸 (cv::Size，通常为切开后的单眼分辨率)
 * @param equiSize IGEV模型输出的展开图尺寸 (cv::Size，如 1280x1024)
 * @param fovHorizontal 水平展开总视角 (弧度)
 * @param fovVertical 垂直展开总视角 (弧度)
 * @param inv_mapx 输出的逆向 X 映射表 (CV_32FC1)
 * @param inv_mapy 输出的逆向 Y 映射表 (CV_32FC1)
 */
    void buildFisheyeWrapMap(
            const cv::Matx33d& K,
            const cv::Vec4d& D,
            const cv::Size& fisheyeSize,
            const cv::Size& equiSize,
            double fovHorizontal,
            double fovVertical,
            cv::Mat& inv_mapx,
            cv::Mat& inv_mapy)
    {
        inv_mapx.create(fisheyeSize, CV_32FC1);
        inv_mapy.create(fisheyeSize, CV_32FC1);

        // 1. 生成鱼眼图像素网格坐标
        std::vector<cv::Point2d> fisheye_points;
        fisheye_points.reserve(fisheyeSize.width * fisheyeSize.height);
        for (int v = 0; v < fisheyeSize.height; ++v) {
            for (int u = 0; u < fisheyeSize.width; ++u) {
                fisheye_points.push_back(cv::Point2d(u, v));
            }
        }

        // 2. 去畸变，获取归一化相机坐标 (x, y)，此时相当于 z = 1
        std::vector<cv::Point2d> undistorted_points;
        cv::fisheye::undistortPoints(fisheye_points, undistorted_points, K, D);

        double alpha_min = -fovHorizontal / 2.0;
        double beta_min  = -fovVertical / 2.0;

        int idx = 0;
        for (int v = 0; v < fisheyeSize.height; ++v) {
            float* ptr_x = inv_mapx.ptr<float>(v);
            float* ptr_y = inv_mapy.ptr<float>(v);
            for (int u = 0; u < fisheyeSize.width; ++u) {
                double x = undistorted_points[idx].x;
                double y = undistorted_points[idx].y;
                idx++;

                // 3. 根据归一化坐标反推经纬度角度 alpha 与 beta
                double beta  = std::atan2(y, 1.0);
                double alpha = std::atan(x * std::cos(beta));

                // 4. 将角度映射回 Equirectangular 展开图的像素坐标 (u_equi, v_equi)
                double u_equi = (alpha - alpha_min) / fovHorizontal * (equiSize.width - 1);
                double v_equi = (beta - beta_min) / fovVertical * (equiSize.height - 1);

                ptr_x[u] = static_cast<float>(u_equi);
                ptr_y[u] = static_cast<float>(v_equi);
            }
        }
    }
};