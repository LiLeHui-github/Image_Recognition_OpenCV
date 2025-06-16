cmake_minimum_required(VERSION 3.12)
project(SimpleDogClassifier)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP /utf-8")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fexec-charset=UTF-8 -finput-charset=UTF-8")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# 查找OpenCV
set(OpenCV_DIR "../lib/install")
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