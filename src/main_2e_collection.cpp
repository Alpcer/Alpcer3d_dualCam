#include <iostream>
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <experimental/filesystem>
#include <stdio.h>
#include <ctime>
#include <cstring>
#include "common_utils.h"
#include <iomanip>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/ocl.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <omp.h>

// 引入 ROS2 及 PointCloud2 相关头文件
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "camera_2eye.h"
#include <CL/cl.h>
#include <deque>
#include <numeric>
#include <cmath>
#include <atomic>

//#define IMU_UPSIDEDOWN
namespace fs = std::experimental::filesystem::v1;
using namespace cv;
using namespace std;

cv::Mat frame_image;
volatile size_t framesScanned = 0;
int image_actual_width;
int image_actual_height;

//计算参数
cv::Mat g_k1, g_k2, g_r1, g_r2, g_extrinsic_left, g_extrinsic_right, g_map_x1, g_map_y1, g_map_x2, g_map_y2;
cv::Mat g_fisheye_stereo_map_x1, g_fisheye_stereo_map_y1, g_fisheye_stereo_map_x2, g_fisheye_stereo_map_y2;
cv::Mat g_panorama_map_x1, g_panorama_map_y1, g_panorama_map_x2, g_panorama_map_y2;
double g_baseline;
cv::Matx33d g_fisheye_k[2];
cv::Vec4d g_fisheye_d[2];     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/

// --- 全局队列与线程同步变量 ---
std::mutex g_cloud_mutex;
std::condition_variable g_cloud_cv;
std::queue<cv::Mat> g_image_queue;

CAMERA2EYE::CameraSingle* pCameraSingle = nullptr;

std::string time_format(uint64_t ns) {
    auto t = (time_t)(ns/1000000000);
    std::tm *tm = std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// --- 线程 1：相机持续采集 (15Hz) ---
void CameraCaptureThread(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_publisher, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_publisher, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher, rclcpp::Node::SharedPtr node) {
    std::string frame_id = node->get_parameter("cam_frame_id").as_string();
    cv::Matx33d tmp_undistort_k;
    cv::Size tmp_undistort_img_size;
    cv::UMat umat;
    long t_s = common_utils::currentTimeMilliseconds();
//    bool print = true;
    while (rclcpp::ok()) {
        // 调用你补充好的相机抓拍缓存函数
        if (pCameraSingle == nullptr) {
            break;
        }
        cv::Mat left, right, depth;
        pCameraSingle->shootStereo(left, right, depth);
//        if (print) {
//            print = false;
//            cv::imwrite("/factory_tools/tmp/test/left.jpg", left);
//            cv::imwrite("/factory_tools/tmp/test/right.jpg", right);
//            cv::imwrite("/factory_tools/tmp/test/depth.jpg", depth);
//        }

        auto msg_left = std::make_unique<sensor_msgs::msg::Image>();
        msg_left->header.stamp = node->now();
        msg_left->header.frame_id = frame_id;
        msg_left->height = left.rows;
        msg_left->width = left.cols;
        msg_left->step = static_cast<sensor_msgs::msg::Image::_step_type>(left.step);
        msg_left->is_bigendian = 0; // x86 和 ARM 架构绝大多数都是小端序 (Little Endian)
        msg_left->encoding = "bgr8";
        msg_left->data.assign(left.data, left.data + left.step * left.rows);
        left_publisher->publish(std::move(msg_left));

        auto msg_right = std::make_unique<sensor_msgs::msg::Image>();
        msg_right->header.stamp = node->now();
        msg_right->header.frame_id = frame_id;
        msg_right->height = right.rows;
        msg_right->width = right.cols;
        msg_right->step = static_cast<sensor_msgs::msg::Image::_step_type>(right.step);
        msg_right->is_bigendian = 0; // x86 和 ARM 架构绝大多数都是小端序 (Little Endian)
        msg_right->encoding = "bgr8";
        msg_right->data.assign(right.data, right.data + right.step * right.rows);
        right_publisher->publish(std::move(msg_right));

        auto msg_depth = std::make_unique<sensor_msgs::msg::Image>();
        msg_depth->header.stamp = node->now();
        msg_depth->header.frame_id = frame_id;
        msg_depth->height = depth.rows;
        msg_depth->width = depth.cols;
        msg_depth->step = static_cast<sensor_msgs::msg::Image::_step_type>(depth.step);
        msg_depth->is_bigendian = 0; // x86 和 ARM 架构绝大多数都是小端序 (Little Endian)
        msg_depth->encoding = "32FC1";
        msg_depth->data.assign(depth.data, depth.data + depth.step * depth.rows);
        depth_publisher->publish(std::move(msg_depth));

        long t_e = common_utils::currentTimeMilliseconds();
        printf("image published, freq: %f Hz\n", 1000.0f/(t_e-t_s));
        t_s = t_e;
    }
}

// --- 线程 2：点云处理与赋色 (速率取决于此线程性能) ---
void ColorizationWorkerThread(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher, rclcpp::Node::SharedPtr node) {
    cv::Mat bgr_image;

    while (rclcpp::ok()) {

        // 1. 阻塞等待，直到拿到最新的点云帧
        {
            std::unique_lock<std::mutex> lock(g_cloud_mutex);
            g_cloud_cv.wait(lock, [] { return (!g_image_queue.empty()) || !rclcpp::ok(); });
            if (!rclcpp::ok()) break;
            bgr_image = std::move(g_image_queue.front());
            g_image_queue.pop();
        }

        long t_c = common_utils::currentTimeMilliseconds();


        printf("data published, processing cost: %f s\n", (common_utils::currentTimeMilliseconds()-t_c)/1000.0f);
    }
}

bool parseResolution(const std::string& resolution, int& width, int& height) {
    std::stringstream ss(resolution);
    char separator;

    ss >> width >> separator >> height;

    // 检查解析是否成功，并且分隔符是 'x'
    if (!ss.fail() && separator == 'x') {
        return true;
    }

    return false;
}

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

void initCalculation() {
    cv::Mat R, T;
    cv::Matx33d k[2];    /*****    摄像机内参数矩阵    ****/
    cv::Vec4d d[2];     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    {
        FileStorage fs("/camera/camera_stereo.xml", FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Open file storage failed!");
        }
        fs["R"] >> R;
        fs["T"] >> T;
        fs["K0"] >> k[0];
        fs["D0"] >> d[0];
        fs["K1"] >> k[1];
        fs["D1"] >> d[1];
        fs.release();
    }

    int width = image_actual_width, height = image_actual_height;
    cv::Size image_size = cv::Size(width/2, height);

// ==================== 新增：基于目标 FOV 与内切正方形的精确控制 ====================

    // 1. 定义你期望保留的目标 FOV（单位：度，例如 100.0 度）
    double target_fov_deg = 120.0;
    double theta = (target_fov_deg / 2.0) * CV_PI / 180.0; // 转换为弧度半视角

    // 2. 借助鱼眼畸变模型，计算该 FOV 在原始鱼眼图像上的物理投影半径（以左目为例）
    double f_fish = k[0](0, 0); // 鱼眼原始焦距
    double k1 = d[0][0], k2 = d[0][1], k3 = d[0][2], k4 = d[0][3];
    double theta2 = theta * theta;
    double theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2 + k3 * theta2 * theta2 * theta2 + k4 * theta2 * theta2 * theta2 * theta2);
    double r_fov = f_fish * theta_d;

    // 3. 计算圆的内切正方形边长 S，并向下取偶数以优化内存对齐
    int S = static_cast<int>(std::sqrt(2.0) * r_fov);
    S = S - S%32;
    cv::Size rectified_image_size(S, S);

    int panorama_s;
    {
        double tmp_theta = (180.0 / 2.0) * CV_PI / 180.0;
        double tmp_theta2 = tmp_theta * tmp_theta;
        double tmp_theta_d = tmp_theta * (1.0 + k1 * tmp_theta2 + k2 * tmp_theta2 * tmp_theta2 + k3 * tmp_theta2 * tmp_theta2 * tmp_theta2 + k4 * tmp_theta2 * tmp_theta2 * tmp_theta2 * tmp_theta2);
        double tmp_r_fov = f_fish * tmp_theta_d;
        panorama_s = tmp_r_fov*2;
        panorama_s = panorama_s - panorama_s%32;
    }
    cv::Size panorama_size(panorama_s, panorama_s);

    std::cout << ">>> 目标 FOV: " << target_fov_deg << " 度" << std::endl;
    std::cout << ">>> 对应的鱼眼圆半径: " << r_fov << " 像素" << std::endl;
    std::cout << ">>> 动态计算出的内切正方形大小 (重映射目标尺寸): " << S << "x" << S << std::endl;

    // 4. 调用 OpenCV 计算基础校正矩阵（这里将 rectified_image_size 传给 newImageSize 参数）
    cv::Mat R1, R2, P1, P2, Q;
    std::cout << "image_size:" << image_size << std::endl;
    std::cout << "rectified_image_size:" << rectified_image_size << std::endl;
    std::cout << "k0:" << k[0] << std::endl;
    std::cout << "k1:" << k[1] << std::endl;
    std::cout << "d0:" << d[0] << std::endl;
    std::cout << "d1:" << d[1] << std::endl;
    std::cout << "R:" << R << std::endl;
    std::cout << "T:" << T << std::endl;
    cv::fisheye::stereoRectify(k[0], d[0], k[1], d[1], image_size, R, T, R1, R2, P1, P2, Q,
                               cv::CALIB_ZERO_DISPARITY, rectified_image_size, 0.0, 1.0);
    std::cout << "R1 :" << R1 << std::endl;
    std::cout << "R2 :" << R2 << std::endl;
    std::cout << "P1 :" << P1 << std::endl;
    std::cout << "P2 :" << P2 << std::endl;
    std::cout << "Q  :" << Q  << std::endl;

    // 5. 依据针孔模型，精确计算并强制覆盖焦距 f_rect，从而绝对控制输出 FOV
    double f_rect = S / (2.0 * std::tan(theta));
    double f_old = P1.at<double>(0, 0); // 备份旧焦距用于等比例缩放基线

    // 重新将主点精准定位到新方形图像的正中心
    double cx_new = S / 2.0;
    double cy_new = S / 2.0;

    // 强制覆写 P1 的内参
    P1.at<double>(0, 0) = P1.at<double>(1, 1) = f_rect;
    P1.at<double>(0, 2) = cx_new;
    P1.at<double>(1, 2) = cy_new;

    // 强制覆写 P2 的内参，保持 fx, fy, cx, cy 与 P1 绝对一致以保证极线水平对齐
    P2.at<double>(0, 0) = P2.at<double>(1, 1) = f_rect;
    P2.at<double>(0, 2) = cx_new;
    P2.at<double>(1, 2) = cy_new;
    // 基线平移项 (Tx * f) 必须根据新旧焦距比值进行等比例缩放
    P2.at<double>(0, 3) = P2.at<double>(0, 3) * (f_rect / f_old);

    // 6. 必须同步重构 Q 矩阵，否则后文 reprojectImageTo3D 还原出的三维点云会产生严重的拉伸和畸变
    Q.at<double>(0, 0) = 1.0;  Q.at<double>(0, 1) = 0.0;  Q.at<double>(0, 2) = 0.0;  Q.at<double>(0, 3) = -cx_new;
    Q.at<double>(1, 0) = 0.0;  Q.at<double>(1, 1) = 1.0;  Q.at<double>(1, 2) = 0.0;  Q.at<double>(1, 3) = -cy_new;
    Q.at<double>(2, 0) = 0.0;  Q.at<double>(2, 1) = 0.0;  Q.at<double>(2, 2) = 0.0;  Q.at<double>(2, 3) = f_rect;
    // Q(3,2) 是 -1/B（B为双目物理基线），不受焦距改变影响，维持 stereoRectify 的默认输出即可
    // Q(3,3) 在 CALIB_ZERO_DISPARITY 且左右主点严格对齐时恒为 0.0
    Q.at<double>(3, 3) = 0.0;

    std::cout << "P1 (Modified):" << P1 << std::endl;
    std::cout << "P2 (Modified):" << P2 << std::endl;
    std::cout << "Q  (Modified):" << Q  << std::endl;

    // ===================================================================================

    // 2. 生成映射表 (此时传入动态计算的 rectified_image_size)
    cv::Mat map1x, map1y, map2x, map2y;
    cv::fisheye::initUndistortRectifyMap(k[0], d[0], R1, P1, rectified_image_size, CV_32FC1, map1x, map1y);
    cv::fisheye::initUndistortRectifyMap(k[1], d[1], R2, P2, rectified_image_size, CV_32FC1, map2x, map2y);

    g_k1 = P1.colRange(0, 3).clone();
    g_k2 = P2.colRange(0, 3).clone();
    g_r1 = R1.clone();
    g_r2 = R2.clone();
    g_map_x1 = map1x.clone();
    g_map_y1 = map1y.clone();
    g_map_x2 = map2x.clone();
    g_map_y2 = map2y.clone();

    g_baseline = cv::norm(T);
    g_fisheye_k[0] = k[0];
    g_fisheye_k[1] = k[1];
    g_fisheye_d[0] = d[0];
    g_fisheye_d[1] = d[1];

    //横向展开
    buildFisheyeUnwrapMap(k[0], d[0], panorama_size, CV_PI, CV_PI, g_panorama_map_x1, g_panorama_map_y1);
    buildFisheyeUnwrapMap(k[1], d[1], panorama_size, CV_PI, CV_PI, g_panorama_map_x2, g_panorama_map_y2);
}

int main(int argc, const char *argv[]) {

    // 初始化 ROS 2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("depth_camera_node");
    // --- 声明并获取参数 ---
    node->declare_parameter<std::string>("cam_frame_id", "camera");
    node->declare_parameter<std::string>("resolution", "3840x1080");
    node->declare_parameter<std::string>("qos", "SystemDefaults");

    std::string resolution = node->get_parameter("resolution").as_string();
    if (!parseResolution(resolution, image_actual_width, image_actual_height)) {
        throw std::runtime_error("parsing image resolution fail!");
    }
    std::cout << "image_actual_width:" << image_actual_width << ", image_actual_height:" << image_actual_height << std::endl;


    std::string camera_intrinsic_filepath_le, camera_intrinsic_filepath_re;
    camera_intrinsic_filepath_le = "/camera/camera_params_le.xml";
    camera_intrinsic_filepath_re = "/camera/camera_params_re.xml";

    std::cout << "camera_intrinsic_filepath_le:" << camera_intrinsic_filepath_le << std::endl;
    std::cout << "camera_intrinsic_filepath_re:" << camera_intrinsic_filepath_re << std::endl;

    // 创建发布者
    rclcpp::QoS qos = rclcpp::SystemDefaultsQoS();
    std::string qos_param = node->get_parameter("qos").as_string();
    if (qos_param == "SystemDefaults") {
        qos = rclcpp::SystemDefaultsQoS();
    } else if (qos_param == "SensorData") {
        qos = rclcpp::SensorDataQoS();
    } else if (qos_param == "Services") {
        qos = rclcpp::ServicesQoS();
    } else {
        std::cout << "unknown QoS param, set to SystemDefaults." << std::endl;
        qos = rclcpp::SystemDefaultsQoS();
    }
    auto publisher_left = node->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);
    auto publisher_right = node->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);
    auto publisher_parallax = node->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", qos);

    pCameraSingle = new CAMERA2EYE::CameraSingle("/dev/video0", image_actual_width, image_actual_height, 2560, 720, camera_intrinsic_filepath_le, camera_intrinsic_filepath_re);

    initCalculation();

    // --- 启动双线程 ---
    // 线程1：相机以 ~15Hz 疯狂向缓存写入 JPEG
    std::thread t_camera(CameraCaptureThread, publisher_left, publisher_right, publisher_parallax, node);
    // 线程2：点云处理线程消费队列，并发布 ROS2 消息
//    std::thread t_worker(ColorizationWorkerThread, publisher_image, node);

    printf("初始化完毕.\n");

    // ROS 2 异步 Spin
    rclcpp::spin(node);
    rclcpp::shutdown();

    // 2. 关键：唤醒可能正在等待数据的线程
    {
        std::lock_guard<std::mutex> lock(g_cloud_mutex);
        g_cloud_cv.notify_all();
    }

    common_utils::sleepMilliseconds(1000);
//    if (t_worker.joinable()) t_worker.join();
    if (t_camera.joinable()) t_camera.join();

    if (pCameraSingle) {
        delete pCameraSingle;
        pCameraSingle = nullptr;
    }
    printf("depth camera end!\n");
    return 0;
}
