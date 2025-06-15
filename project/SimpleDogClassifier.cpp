#include "SimpleDogClassifier.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;

SimpleDogClassifier::SimpleDogClassifier(bool enable_optimizations)
    : use_data_augmentation_(enable_optimizations), use_auto_tuning_(enable_optimizations)
{
    // 初始化小尺度HOG (适合头部特征)
    hog_small_ = cv::HOGDescriptor(
        cv::Size(64, 64),         // 窗口大小
        cv::Size(16, 16),         // 块大小
        cv::Size(8, 8),           // 块步长
        cv::Size(8, 8),           // 单元大小
        9,                        // 方向bins
        1,                        // derivAperture
        -1,                       // winSigma (自动)
        cv::HOGDescriptor::L2Hys, // 归一化
        0.2,                      // L2Hys阈值
        true                      // gamma校正
    );

    // 初始化大尺度HOG (适合全身特征)
    hog_large_ = cv::HOGDescriptor(
        cv::Size(128, 128), // 更大窗口
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9,
        1,
        -1,
        cv::HOGDescriptor::L2Hys,
        0.2,
        true);

    // 初始化SVM
    svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::RBF);

    // 如果不启用自动调优，使用默认参数
    if (!use_auto_tuning_)
    {
        svm_->setC(1.0);
        svm_->setGamma(0.5);
    }
}

bool SimpleDogClassifier::train(const std::string &images_dir, const std::string &annotations_dir)
{
    std::cout << "开始训练..." << std::endl;

    // 1. 收集所有标注文件
    std::vector<DogAnnotation> annotations;
    std::set<std::string> breed_set;

    std::cout << "读取标注文件..." << std::endl;

    try
    {
        for (const auto &entry : fs::recursive_directory_iterator(annotations_dir))
        {
            if (entry.path().extension() == ".xml")
            {
                DogAnnotation annotation = parseXML(entry.path().string());
                if (!annotation.breed.empty())
                {
                    annotations.push_back(annotation);
                    breed_set.insert(annotation.breed);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    // 2. 建立品种索引
    breed_names_.clear();
    for (const auto &breed : breed_set)
    {
        breed_names_.push_back(breed);
    }
    std::sort(breed_names_.begin(), breed_names_.end());

    std::cout << "发现 " << breed_names_.size() << " 个品种，"
              << annotations.size() << " 个样本" << std::endl;

    // 3. 数据增强 + 特征提取
    std::cout << "提取特征..." << std::endl;
    if (use_data_augmentation_)
    {
        std::cout << "启用数据增强..." << std::endl;
    }

    std::vector<cv::Mat> all_features;
    std::vector<int> all_labels;

    int processed = 0;
    for (const auto &annotation : annotations)
    {
        // 构建图片路径
        std::string image_path = images_dir + "/" + annotation.breed + "/" + annotation.image_file;

        cv::Mat image = cv::imread(image_path);
        if (image.empty())
        {
            std::cout << "无法读取: " << image_path << std::endl;
            continue;
        }

        // 智能ROI提取
        cv::Mat roi = extractSmartROI(image, annotation);

        // 获取品种标签
        auto it = std::find(breed_names_.begin(), breed_names_.end(), annotation.breed);
        int label = std::distance(breed_names_.begin(), it);

        // 数据增强
        std::vector<cv::Mat> augmented_images;
        if (use_data_augmentation_)
        {
            augmented_images = augmentImage(roi);
        }
        else
        {
            augmented_images.push_back(roi);
        }

        // 为每个增强图像提取特征
        for (const auto &aug_img : augmented_images)
        {
            cv::Mat features = extractOptimizedFeatures(aug_img, annotation);
            if (!features.empty())
            {
                all_features.push_back(features);
                all_labels.push_back(label);
            }
        }

        processed++;
        if (processed % 100 == 0)
        {
            std::cout << "处理进度: " << processed << "/" << annotations.size()
                      << " (增强到 " << all_features.size() << " 样本)" << std::endl;
        }
    }

    std::cout << "成功处理 " << processed << " 个样本，共生成 " << all_features.size() << " 个训练样本" << std::endl;

    if (all_features.empty())
    {
        std::cout << "没有有效的训练数据" << std::endl;
        return false;
    }

    // 4. 准备训练数据
    cv::Mat training_data;
    cv::vconcat(all_features, training_data);
    cv::Mat labels(all_labels, true);

    training_data.convertTo(training_data, CV_32FC1);
    labels.convertTo(labels, CV_32SC1);

    // 5. 训练SVM (支持自动调参)
    std::cout << "开始SVM训练..." << std::endl;
    std::cout << "训练数据: " << training_data.rows << " 样本, " << training_data.cols << " 特征" << std::endl;

    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(
        training_data, cv::ml::ROW_SAMPLE, labels);

    bool success;
    if (use_auto_tuning_)
    {
        std::cout << "启用SVM自动参数调优..." << std::endl;
        success = svm_->trainAuto(train_data);

        if (success)
        {
            std::cout << "最优参数: C=" << svm_->getC() << ", Gamma=" << svm_->getGamma() << std::endl;
        }
    }
    else
    {
        success = svm_->train(train_data);
    }

    if (success)
    {
        std::cout << "训练完成！" << std::endl;
    }
    else
    {
        std::cout << "训练失败" << std::endl;
    }

    return success;
}

std::string SimpleDogClassifier::predict(const std::string &image_path)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cout << "无法读取图片: " << image_path << std::endl;
        return "unknown";
    }

    // 没有标注信息时，使用优化的特征提取
    DogAnnotation empty_annotation;
    cv::Mat features = extractOptimizedFeatures(image, empty_annotation);
    features.convertTo(features, CV_32FC1);

    float result = svm_->predict(features);
    int breed_index = static_cast<int>(result);

    if (breed_index >= 0 && breed_index < breed_names_.size())
    {
        return breed_names_[breed_index];
    }

    return "unknown";
}

void SimpleDogClassifier::saveModel(const std::string &model_path)
{
    // 保存SVM模型
    svm_->save(model_path);

    // 保存品种名列表
    std::ofstream file(model_path + ".breeds");
    for (const auto &breed : breed_names_)
    {
        file << breed << std::endl;
    }
    file.close();

    std::cout << "模型已保存到: " << model_path << std::endl;
}

bool SimpleDogClassifier::loadModel(const std::string &model_path)
{
    try
    {
        // 加载SVM模型
        svm_ = cv::ml::StatModel::load<cv::ml::SVM>(model_path);

        // 加载品种名列表
        std::ifstream file(model_path + ".breeds");
        std::string breed;
        breed_names_.clear();
        while (std::getline(file, breed))
        {
            breed_names_.push_back(breed);
        }
        file.close();

        std::cout << "模型已加载，包含 " << breed_names_.size() << " 个品种" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cout << "模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

DogAnnotation SimpleDogClassifier::parseXML(const std::string &xml_path)
{
    DogAnnotation annotation;

    std::ifstream file(xml_path);
    if (!file.is_open())
    {
        return annotation;
    }

    // 从路径获取品种名和文件名
    fs::path path(xml_path);
    annotation.breed = path.parent_path().filename().string();
    annotation.image_file = path.stem().string();

    // 简化的XML解析
    std::string line;
    BBox current_box;
    bool in_object = false, in_bndbox = false;
    std::string object_name;

    while (std::getline(file, line))
    {
        // 移除空格
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        if (line.find("<object>") != std::string::npos)
        {
            in_object = true;
            current_box = BBox{};
        }
        else if (line.find("</object>") != std::string::npos)
        {
            if (in_object && !object_name.empty())
            {
                current_box.part = object_name;
                annotation.boxes.push_back(current_box);
            }
            in_object = false;
            object_name.clear();
        }
        else if (in_object && line.find("<name>") != std::string::npos)
        {
            size_t start = line.find(">") + 1;
            size_t end = line.find("</name>");
            if (end != std::string::npos)
            {
                object_name = line.substr(start, end - start);
            }
        }
        else if (line.find("<bndbox>") != std::string::npos)
        {
            in_bndbox = true;
        }
        else if (line.find("</bndbox>") != std::string::npos)
        {
            in_bndbox = false;
        }
        else if (in_bndbox)
        {
            if (line.find("<xmin>") != std::string::npos)
            {
                size_t start = line.find(">") + 1;
                size_t end = line.find("</xmin>");
                current_box.x = std::stoi(line.substr(start, end - start));
            }
            else if (line.find("<ymin>") != std::string::npos)
            {
                size_t start = line.find(">") + 1;
                size_t end = line.find("</ymin>");
                current_box.y = std::stoi(line.substr(start, end - start));
            }
            else if (line.find("<xmax>") != std::string::npos)
            {
                size_t start = line.find(">") + 1;
                size_t end = line.find("</xmax>");
                int xmax = std::stoi(line.substr(start, end - start));
                current_box.width = xmax - current_box.x;
            }
            else if (line.find("<ymax>") != std::string::npos)
            {
                size_t start = line.find(">") + 1;
                size_t end = line.find("</ymax>");
                int ymax = std::stoi(line.substr(start, end - start));
                current_box.height = ymax - current_box.y;
            }
        }
    }

    return annotation;
}

cv::Mat SimpleDogClassifier::extractFeatures(const cv::Mat &image, const DogAnnotation &annotation)
{
    // 优先使用头部区域，如果没有就用身体区域
    BBox target_box;
    bool found = false;

    // 先找头部
    for (const auto &box : annotation.boxes)
    {
        if (box.part == "head")
        {
            target_box = box;
            found = true;
            break;
        }
    }

    // 没有头部就找身体
    if (!found)
    {
        for (const auto &box : annotation.boxes)
        {
            if (box.part == "body")
            {
                target_box = box;
                found = true;
                break;
            }
        }
    }

    cv::Mat roi;
    if (found)
    {
        // 使用标注区域
        cv::Rect rect(target_box.x, target_box.y, target_box.width, target_box.height);
        rect &= cv::Rect(0, 0, image.cols, image.rows); // 确保在图像范围内
        if (rect.width > 0 && rect.height > 0)
        {
            roi = image(rect);
        }
    }

    // 如果没有有效的标注区域，使用全图
    if (roi.empty())
    {
        roi = image;
    }

    // 调整到HOG需要的尺寸
    cv::Mat resized;
    cv::resize(roi, resized, hog_small_.winSize);

    // 计算HOG特征
    std::vector<float> descriptors;
    hog_small_.compute(resized, descriptors);

    return cv::Mat(descriptors, true).reshape(1, 1);
}

// 新的优化特征提取方法
cv::Mat SimpleDogClassifier::extractOptimizedFeatures(const cv::Mat &image, const DogAnnotation &annotation)
{
    std::vector<float> combined_features;

    // 1. 小尺度特征 (适合头部)
    cv::Mat small_img;
    cv::resize(image, small_img, hog_small_.winSize);
    std::vector<float> small_features;
    hog_small_.compute(small_img, small_features);
    combined_features.insert(combined_features.end(), small_features.begin(), small_features.end());

    // 2. 大尺度特征 (适合全身)
    cv::Mat large_img;
    cv::resize(image, large_img, hog_large_.winSize);
    std::vector<float> large_features;
    hog_large_.compute(large_img, large_features);
    combined_features.insert(combined_features.end(), large_features.begin(), large_features.end());

    return cv::Mat(combined_features, true).reshape(1, 1);
}

// 智能ROI提取
cv::Mat SimpleDogClassifier::extractSmartROI(const cv::Mat &image, const DogAnnotation &annotation)
{
    cv::Rect optimal_roi;

    // 1. 优先使用头部区域
    for (const auto &box : annotation.boxes)
    {
        if (box.part == "head")
        {
            optimal_roi = cv::Rect(box.x, box.y, box.width, box.height);
            break;
        }
    }

    // 2. 没有头部就用身体
    if (optimal_roi.area() == 0)
    {
        for (const auto &box : annotation.boxes)
        {
            if (box.part == "body")
            {
                optimal_roi = cv::Rect(box.x, box.y, box.width, box.height);
                break;
            }
        }
    }

    // 3. 扩展ROI包含更多上下文
    if (optimal_roi.area() > 0)
    {
        optimal_roi = expandROI(optimal_roi, image.size(), 0.15);
        optimal_roi &= cv::Rect(0, 0, image.cols, image.rows);
        return image(optimal_roi);
    }

    // 4. 没有标注就用全图
    return image;
}

// 数据增强
std::vector<cv::Mat> SimpleDogClassifier::augmentImage(const cv::Mat &image)
{
    std::vector<cv::Mat> augmented;
    augmented.push_back(image.clone());

    // 1. 水平翻转
    cv::Mat flipped;
    cv::flip(image, flipped, 1);
    augmented.push_back(flipped);

    // 2. 轻微旋转
    augmented.push_back(rotateImage(image, -8.0f));
    augmented.push_back(rotateImage(image, 8.0f));

    // 3. 亮度调整
    cv::Mat brightened, darkened;
    image.convertTo(brightened, -1, 1.2, 0);
    image.convertTo(darkened, -1, 0.8, 0);
    augmented.push_back(brightened);
    augmented.push_back(darkened);

    return augmented;
}

// 图像旋转辅助函数
cv::Mat SimpleDogClassifier::rotateImage(const cv::Mat &src, float angle)
{
    cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(src, rotated, rot_mat, src.size());
    return rotated;
}

// ROI扩展辅助函数
cv::Rect SimpleDogClassifier::expandROI(const cv::Rect &roi, const cv::Size &img_size, float ratio)
{
    int expand_w = static_cast<int>(roi.width * ratio);
    int expand_h = static_cast<int>(roi.height * ratio);

    return cv::Rect(
        std::max(0, roi.x - expand_w / 2),
        std::max(0, roi.y - expand_h / 2),
        std::min(img_size.width - (roi.x - expand_w / 2), roi.width + expand_w),
        std::min(img_size.height - (roi.y - expand_h / 2), roi.height + expand_h));
}

// 性能评估
void SimpleDogClassifier::evaluatePerformance(const std::string &images_dir, const std::string &annotations_dir)
{
    std::cout << "=== 开始性能评估 ===" << std::endl;

    // 重新加载测试数据
    std::vector<DogAnnotation> test_annotations;
    try
    {
        for (const auto &entry : fs::recursive_directory_iterator(annotations_dir))
        {
            if (entry.path().extension() == ".xml")
            {
                DogAnnotation annotation = parseXML(entry.path().string());
                if (!annotation.breed.empty())
                {
                    test_annotations.push_back(annotation);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "加载测试数据失败: " << e.what() << std::endl;
        return;
    }

    int correct = 0;
    int total = 0;
    std::map<std::string, int> breed_correct;
    std::map<std::string, int> breed_total;

    for (const auto &annotation : test_annotations)
    {
        std::string image_path = images_dir + "/" + annotation.breed + "/" + annotation.image_file + ".jpg";
        std::string predicted = predict(image_path);

        breed_total[annotation.breed]++;
        total++;

        if (predicted == annotation.breed)
        {
            correct++;
            breed_correct[annotation.breed]++;
        }

        if (total % 100 == 0)
        {
            std::cout << "评估进度: " << total << "/" << test_annotations.size() << std::endl;
        }
    }

    std::cout << "\n=== 评估结果 ===" << std::endl;
    std::cout << "总体准确率: " << (double)correct / total * 100 << "%" << std::endl;
    std::cout << "正确: " << correct << " / " << total << std::endl;

    std::cout << "\n前10个品种准确率:" << std::endl;
    int count = 0;
    for (const auto &breed : breed_names_)
    {
        if (breed_total[breed] > 0 && count < 10)
        {
            double acc = (double)breed_correct[breed] / breed_total[breed] * 100;
            std::cout << breed << ": " << acc << "% ("
                      << breed_correct[breed] << "/" << breed_total[breed] << ")" << std::endl;
            count++;
        }
    }
}