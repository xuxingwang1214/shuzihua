#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>  // LibTorch TorchScript API

// 手部检测结果
struct HandResult {
    std::vector<cv::Point2f> points;  // 21个关键点
    float score;
    bool isRight;
};

class MotionRecognizer
{
public:
    MotionRecognizer();
    ~MotionRecognizer();

    void initialize();
    void recognize(const cv::Mat& frame);

    // 获取识别结果 (对应 C# QueryResult)
    void getResult(std::map<std::string, float>* output);
    // 获取骨架 JSON (对应 C# exactbones / extractBone)
    std::string extractBone();
    // 获取手部关键点 JSON
    std::string extractHands();
    cv::Mat generatePreview();

private:
    // LibTorch 模型
    torch::jit::script::Module yoloModel_;
    torch::jit::script::Module tcnModel_;
    torch::jit::script::Module handModel_;
    bool modelsLoaded_ = false;
    
    // 设备 (CPU/CUDA)
    torch::Device device_ = torch::kCPU;

    static constexpr int NUM_JOINTS = 17;
    static constexpr int WINDOW_SIZE = 32; // 匹配 Python dummy_input 的 32
    static constexpr int TCN_CHANNELS = 12; // 6个点 * (x,y) = 12
    static constexpr int HAND_INPUT_SIZE = 224;

    std::vector<std::vector<float>> frameBuffer_;
    std::vector<float> lastOutput_;

    cv::Mat previewFrame_;
    std::vector<cv::Point2f> joints_;
    std::vector<HandResult> hands_;

    void runYOLO(const cv::Mat& frame);
    void runTCN();
    void runHandDetection(const cv::Mat& frame);
    HandResult detectHand(const cv::Mat& frame, const cv::Rect& roi);
    
    // 辅助函数: OpenCV Mat 转 Tensor
    torch::Tensor matToTensor(const cv::Mat& image);
};