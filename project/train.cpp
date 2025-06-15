// 案例1: 优化版训练模型
// 用法: ./train [--basic]

#include "SimpleDogClassifier.h"
#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
    std::cout << "狗品种分类器 - 优化训练模式" << std::endl;
    std::cout << "================================" << std::endl;

    // 检查是否使用基础模式
    bool use_optimizations = true;
    if (argc > 1 && std::string(argv[1]) == "--basic")
    {
        use_optimizations = false;
        std::cout << "运行模式: 基础版 (无优化)" << std::endl;
    }
    else
    {
        std::cout << "运行模式: 优化版 (推荐)" << std::endl;
        std::cout << "优化特性:" << std::endl;
        std::cout << "  ✓ 智能ROI提取 (优先头部区域)" << std::endl;
        std::cout << "  ✓ 数据增强 (翻转、旋转、亮度)" << std::endl;
        std::cout << "  ✓ 多尺度HOG特征 (64x64 + 128x128)" << std::endl;
        std::cout << "  ✓ SVM自动参数调优" << std::endl;
        std::cout << "  ✓ 预期识别率提升 15-25%" << std::endl;
    }
    std::cout << std::endl;

    // 1. 创建分类器 (传入优化参数)
    SimpleDogClassifier classifier(use_optimizations);

    // 2. 设置数据路径
    std::string images_dir = "F:/source/Project/Cpp/Image_Recognition_OpenCV/resource/low-resolution";       // 图片文件夹
    std::string annotations_dir = "F:/source/Project/Cpp/Image_Recognition_OpenCV/resource/low-annotations"; // 标注文件夹

    std::cout << "图片目录: " << images_dir << std::endl;
    std::cout << "标注目录: " << annotations_dir << std::endl;
    std::cout << std::endl;

    // 3. 开始训练 (记录时间)
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "开始训练..." << std::endl;

    bool success = classifier.train(images_dir, annotations_dir);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);

    if (success)
    {
        // 4. 保存训练好的模型
        std::string model_path = use_optimizations ? "dog_model_optimized.xml" : "dog_model_basic.xml";
        classifier.saveModel(model_path);

        std::cout << std::endl;
        std::cout << "训练成功完成！" << std::endl;
        std::cout << "训练时间: " << duration.count() << " 分钟" << std::endl;
        std::cout << "模型已保存为: " << model_path << std::endl;
        std::cout << "品种列表: " << model_path << ".breeds" << std::endl;

        if (use_optimizations)
        {
            std::cout << std::endl;
            std::cout << "优化版特性已启用，预期识别率提升 15-25%" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "后续步骤:" << std::endl;
        std::cout << "1. 使用 predict.exe 预测新图片" << std::endl;
        std::cout << "2. 运行 predict.exe --test 评估模型性能" << std::endl;
        std::cout << std::endl;
        std::cout << "示例: ./predict test_dog.jpg" << std::endl;
    }
    else
    {
        std::cout << "训练失败，请检查数据路径和文件格式" << std::endl;
        std::cout << "确保以下目录存在且包含数据:" << std::endl;
        std::cout << "- 图片目录: " << images_dir << std::endl;
        std::cout << "- 标注目录: " << annotations_dir << std::endl;
        return 1;
    }

    return 0;
}