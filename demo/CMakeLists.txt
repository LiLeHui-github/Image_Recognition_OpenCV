cmake_minimum_required(VERSION 3.18)

project(Demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /utf-8")

# 导入OpenCV
set(OpenCV_DIR "F:/work/project/Cpp/OpenCV/Image_Recognition_OpenCV/lib/install")
find_package(OpenCV REQUIRED)

# 模型训练
add_executable(train train.cpp)
target_include_directories(train PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(train ${OpenCV_LIBS})

# 模型预测
add_executable(predict predict.cpp)
target_include_directories(predict PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(predict ${OpenCV_LIBS})