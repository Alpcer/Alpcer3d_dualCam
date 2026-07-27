cmake_minimum_required(VERSION 3.10)
project(depth_camera LANGUAGES CXX CUDA) # 💡【修改1】显式声明支持 CXX 和 CUDA 语言

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 💡【修改2】设置 CUDA 的编译标准（通常与 CXX 一致）以及指定生成的 GPU 架构（根据你的显卡调整，如 Orin/Rockchip/NVIDIA）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# 💡 新增：解决 CMP0104 警告
cmake_policy(SET CMP0104 NEW)
# "native" 表示自动检测当前机器上的显卡架构并针对性编译（推荐本地开发使用）
#set(CMAKE_CUDA_ARCHITECTURES "native")
# 如果你需要在其他机器运行或使用的 CMake 版本低于 3.24，可替换为具体显卡算力，例如：
set(CMAKE_CUDA_ARCHITECTURES 90)

add_compile_options(-lstdc++fs)

# ===================== 依赖1：CUDA 运行时 =====================
cmake_policy(SET CMP0074 NEW)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# ===================== 依赖2：OpenCV（需带 CUDA 模块） =====================
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc cudaarithm cudaimgproc calib3d highgui cudawarping)
message("OpenCV linked libs: ${OpenCV_LIBS}")
if ("opencv_cudaimgproc" IN_LIST OpenCV_LIBS)
    message(STATUS "✅ OpenCV 包含 cudaimgproc 模块")
else()
    message(WARNING "❌ OpenCV 缺少 cudaimgproc，无法使用 cv::cuda::extractChannel")
endif()

# 必须先找到 ament_cmake
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Eigen3 REQUIRED)

include_directories(include
        /usr/include/x86_64-linux-gnu
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS})

link_directories(/usr/lib/x86_64-linux-gnu)

# 💡【修改3】将你的 .cu 文件加入到编译生成的可执行文件中
add_executable(depth_camera_node
        src/main_2e_collection.cpp
        src/IGEVTensorRTEngine.cpp
        src/fisheye_stereo_pipeline_gpu.cu # <--- 确保填入正确的相对路径
        )

# ROS2 依赖项绑定
ament_target_dependencies(depth_camera_node rclcpp sensor_msgs Threads OpenCV OpenMP)

# 头文件搜索路径
target_include_directories(depth_camera_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# 库文件搜索路径
target_link_directories(depth_camera_node PRIVATE ${OpenCV_INClUDE_DIRS})

target_link_libraries(depth_camera_node
        stdc++fs
        nvinfer
        nvonnxparser
        ${CUDA_LIBRARIES}
        # NVIDIA CUDA 与硬件加速库 (提供显存、解码与图像缩放实现)
        CUDA::cudart
        CUDA::nvjpeg
        CUDA::nppig
        CUDA::nppc
        CUDA::nppicc
        CUDA::nppidei
        )

# --- 增加 RPATH 设置 ---
#set(CMAKE_INSTALL_RPATH "${TRTRTX_LIB_DIR}")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ==================== 设置安装路径 ====================
install(TARGETS
        depth_camera_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}
        )

ament_package()