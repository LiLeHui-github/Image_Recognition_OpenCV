cmake_minimum_required(VERSION 3.12)
project(SimpleDogClassifier)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# 查找OpenCV
set(OpenCV_DIR "F:/source/Project/Cpp/Image_Recognition_OpenCV/lib/opencv/install")
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 创建SimpleDogClassifier库
add_library(SimpleDogClassifier 
    SimpleDogClassifier.cpp
)

# 链接OpenCV库到分类器库
target_link_libraries(SimpleDogClassifier 
    ${OpenCV_LIBS}
)

# 训练程序
add_executable(train 
    train.cpp
)
target_link_libraries(train 
    SimpleDogClassifier
    ${OpenCV_LIBS}
)

# 预测程序
add_executable(predict 
    predict.cpp
)
target_link_libraries(predict 
    SimpleDogClassifier
    ${OpenCV_LIBS}
)

# 设置输出目录
set_target_properties(train predict PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 输出信息
message(STATUS "===== 优化版犬种分类器 =====")
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "优化特性:")
message(STATUS "  ✓ 智能ROI提取")
message(STATUS "  ✓ 数据增强")
message(STATUS "  ✓ 多尺度HOG特征")
message(STATUS "  ✓ SVM自动参数调优")
message(STATUS "===============================")

# 构建后显示使用说明
add_custom_target(usage ALL
    COMMAND ${CMAKE_COMMAND} -E echo "===== 构建完成 ====="
    COMMAND ${CMAKE_COMMAND} -E echo "可执行文件位置:"
    COMMAND ${CMAKE_COMMAND} -E echo "  训练: ${CMAKE_BINARY_DIR}/bin/train.exe"
    COMMAND ${CMAKE_COMMAND} -E echo "  预测: ${CMAKE_BINARY_DIR}/bin/predict.exe"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "使用方法:"
    COMMAND ${CMAKE_COMMAND} -E echo "  优化版训练: ./train"
    COMMAND ${CMAKE_COMMAND} -E echo "  基础版训练: ./train --basic"
    COMMAND ${CMAKE_COMMAND} -E echo "  预测图片: ./predict test.jpg"
    COMMAND ${CMAKE_COMMAND} -E echo "  性能测试: ./predict --test"
    DEPENDS train predict
) 