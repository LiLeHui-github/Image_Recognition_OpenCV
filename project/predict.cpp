// 案例2: 优化版预测图片
// 用法: ./predict <图片路径> [--basic] [--test]

#include "SimpleDogClassifier.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "狗品种分类器 - 预测模式" << std::endl;
    std::cout << "================================" << std::endl;

    // 解析命令行参数
    bool use_basic = false;
    bool run_test = false;
    std::string image_path;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--basic")
        {
            use_basic = true;
        }
        else if (arg == "--test")
        {
            run_test = true;
        }
        else if (image_path.empty() && arg[0] != '-')
        {
            image_path = arg;
        }
    }

    // 检查参数
    if (!run_test && image_path.empty())
    {
        std::cout << "用法: " << argv[0] << " <图片路径> [选项]" << std::endl;
        std::cout << "选项:" << std::endl;
        std::cout << "  --basic    使用基础模型" << std::endl;
        std::cout << "  --test     运行性能测试" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " test_dog.jpg" << std::endl;
        std::cout << "  " << argv[0] << " test_dog.jpg --basic" << std::endl;
        std::cout << "  " << argv[0] << " --test" << std::endl;
        return 1;
    }

    // 确定模型路径
    std::string model_path = use_basic ? "dog_model_basic.xml" : "dog_model_optimized.xml";

    std::cout << "使用模型: " << (use_basic ? "基础版" : "优化版") << std::endl;

    if (!run_test)
    {
        std::cout << "预测图片: " << image_path << std::endl;
    }
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << std::endl;

    // 1. 创建分类器 (根据模型类型确定优化参数)
    SimpleDogClassifier classifier(!use_basic);

    // 2. 加载训练好的模型
    if (!classifier.loadModel(model_path))
    {
        std::cout << "无法加载模型，请先运行训练！" << std::endl;
        std::cout << "基础版: ./train --basic" << std::endl;
        std::cout << "优化版: ./train" << std::endl;
        return 1;
    }

    if (run_test)
    {
        // 运行性能测试
        std::cout << "开始性能评估..." << std::endl;
        std::string images_dir = "F:/source/Project/Cpp/Image_Recognition_OpenCV/resource/low-resolution";
        std::string annotations_dir = "F:/source/Project/Cpp/Image_Recognition_OpenCV/resource/low-annotations";

        classifier.evaluatePerformance(images_dir, annotations_dir);
    }
    else
    {
        // 3. 预测单张图片
        std::cout << "正在分析图片..." << std::endl;
        std::string predicted_breed = classifier.predict(image_path);

        // 4. 显示结果
        std::cout << std::endl;
        std::cout << "预测结果: " << predicted_breed << std::endl;

        if (predicted_breed == "unknown")
        {
            std::cout << "无法识别此图片" << std::endl;
            std::cout << "可能原因:" << std::endl;
            std::cout << "- 图片中没有狗" << std::endl;
            std::cout << "- 图片质量过低" << std::endl;
            std::cout << "- 品种不在训练数据中" << std::endl;
        }
        else
        {
            std::cout << "这是一只: " << predicted_breed << std::endl;
            if (!use_basic)
            {
                std::cout << "使用优化版模型预测" << std::endl;
            }
        }
    }

    return 0;
}