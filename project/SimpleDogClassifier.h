#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>
#include <map>
#include <set>

// 边界框结构
struct BBox
{
    int x, y, width, height;
    std::string part; // "head" 或 "body"
};

// 标注信息
struct DogAnnotation
{
    std::string breed;       // 品种名
    std::string image_file;  // 图片文件名
    std::vector<BBox> boxes; // 边界框列表
};

class SimpleDogClassifier
{
private:
    cv::HOGDescriptor hog_small_; // 小尺度HOG (64x64)
    cv::HOGDescriptor hog_large_; // 大尺度HOG (128x128)
    cv::Ptr<cv::ml::SVM> svm_;
    std::vector<std::string> breed_names_;
    bool use_data_augmentation_; // 是否使用数据增强
    bool use_auto_tuning_;       // 是否使用SVM自动调参

public:
    SimpleDogClassifier(bool enable_optimizations = true);

    // 核心功能
    bool train(const std::string &images_dir, const std::string &annotations_dir);
    std::string predict(const std::string &image_path);

    // 模型管理
    void saveModel(const std::string &model_path);
    bool loadModel(const std::string &model_path);

    // 性能评估
    void evaluatePerformance(const std::string &images_dir, const std::string &annotations_dir);

private:
    // XML解析
    DogAnnotation parseXML(const std::string &xml_path);

    // 特征提取方法
    cv::Mat extractFeatures(const cv::Mat &image, const DogAnnotation &annotation);
    cv::Mat extractOptimizedFeatures(const cv::Mat &image, const DogAnnotation &annotation);

    // 智能ROI提取
    cv::Mat extractSmartROI(const cv::Mat &image, const DogAnnotation &annotation);

    // 数据增强
    std::vector<cv::Mat> augmentImage(const cv::Mat &image);

    // 辅助函数
    cv::Mat rotateImage(const cv::Mat &src, float angle);
    cv::Rect expandROI(const cv::Rect &roi, const cv::Size &img_size, float ratio);
};