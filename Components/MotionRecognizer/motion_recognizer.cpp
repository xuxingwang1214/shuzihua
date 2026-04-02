#include "pch.h"
#include "motion_recognizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <iostream>

MotionRecognizer::MotionRecognizer() 
    : device_(torch::kCPU) {}

MotionRecognizer::~MotionRecognizer() = default;

void MotionRecognizer::initialize()
{
    try {
        // 检查 CUDA 可用性
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        } else {
            device_ = torch::kCPU;
            std::cout << "Using CPU device" << std::endl;
        }

        // 加载 TorchScript 模型
        std::cout << "Loading YOLO model..." << std::endl;
        yoloModel_ = torch::jit::load("./yolov8n-pose_traced.pt");
        yoloModel_.to(device_);
        yoloModel_.eval();
        
        std::cout << "Loading TCN model..." << std::endl;
        tcnModel_ = torch::jit::load("./tcn_action_traced.pt");
        tcnModel_.to(device_);
        tcnModel_.eval();
        
        std::cout << "Loading Hand model..." << std::endl;
        handModel_ = torch::jit::load("./hand_landmark_traced.pt");
        handModel_.to(device_);
        handModel_.eval();
        
        modelsLoaded_ = true;
        std::cout << "Models loaded successfully!" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading models: " << e.what() << std::endl;
        modelsLoaded_ = false;
    }

    frameBuffer_.clear();
    lastOutput_.assign(6, 0.0f);
}

torch::Tensor MotionRecognizer::matToTensor(const cv::Mat& image)
{
    // 将 OpenCV Mat 转换为 PyTorch Tensor
    // 输入: BGR HWC 格式
    // 输出: RGB NCHW 格式, 归一化到 [0, 1]
    
    cv::Mat rgb, resized;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    
    // 创建 tensor [H, W, C]
    auto tensor = torch::from_blob(
        resized.data, 
        {resized.rows, resized.cols, 3}, 
        torch::kFloat32
    ).clone();  // clone 确保数据拥有权
    
    // 转换为 [C, H, W]
    tensor = tensor.permute({2, 0, 1});
    
    // 添加 batch 维度 [1, C, H, W]
    tensor = tensor.unsqueeze(0);
    
    return tensor.to(device_);
}

void MotionRecognizer::recognize(const cv::Mat& frame)
{
    if (frame.empty() || !modelsLoaded_) return;
    previewFrame_ = frame.clone();

    runYOLO(frame);
    runHandDetection(frame);

    // 如果没检测到关键点
    int validCount = 0;
    for (const auto& pt : joints_) {
        if (pt.x > 0.1f || pt.y > 0.1f) validCount++;
    }

    // 至少需要看到关键关节点才送入 TCN
    if (validCount < 6) {
        if (!frameBuffer_.empty()) frameBuffer_.erase(frameBuffer_.begin());
        return;
    }

    // 提取胳膊 6 个点: 5,6,7,8,9,10 (左肩、右肩、左肘、右肘、左腕、右腕)
    std::vector<float> oneFrame(12); // TCN_CHANNELS = 12
    int armIdx[] = { 5, 6, 7, 8, 9, 10 };
    
    // 计算肩膀中点用于中心化（与训练环境一致）
    float shoulder_mid_x = (joints_[5].x + joints_[6].x) / 2.0f;
    float shoulder_mid_y = (joints_[5].y + joints_[6].y) / 2.0f;
    
    for (int i = 0; i < 6; ++i) {
        int idx = armIdx[i];
        // joints_中存储的已经是归一化到[0,1]的坐标，直接用于中心化
        // 不再除以frame.cols/rows（避免双重归一化）
        float rel_x = joints_[idx].x - shoulder_mid_x;
        float rel_y = joints_[idx].y - shoulder_mid_y;
        oneFrame[i * 2 + 0] = rel_x;
        oneFrame[i * 2 + 1] = rel_y;
    }

    frameBuffer_.push_back(oneFrame);
    if (frameBuffer_.size() > 32) frameBuffer_.erase(frameBuffer_.begin()); // WINDOW_SIZE = 32
    if (frameBuffer_.size() == 32) runTCN();
}

void MotionRecognizer::runHandDetection(const cv::Mat& frame)
{
    hands_.clear();
    
    // 基于手腕和肘部位置检测手部
    // YOLO pose 关键点索引: 9=左手腕, 10=右手腕, 7=左肘, 8=右肘
    int handConfig[][2] = { {9, 7}, {10, 8} };  // {手腕, 肘部}
    
    for (int h = 0; h < 2; ++h) {
        int wristIdx = handConfig[h][0];
        int elbowIdx = handConfig[h][1];
        
        if (wristIdx >= (int)joints_.size() || joints_[wristIdx].x <= 0 || joints_[wristIdx].y <= 0)
            continue;
        
        cv::Point2f wrist = joints_[wristIdx];
        
        // 计算 ROI 大小
        int roiSize = 180;
        if (elbowIdx < (int)joints_.size() && joints_[elbowIdx].x > 0) {
            cv::Point2f elbow = joints_[elbowIdx];
            double armLength = cv::norm(wrist - elbow);
            roiSize = std::max(120, std::min(300, (int)(armLength * 1.2)));
        }
        
        // 计算偏移
        int offsetX = 0, offsetY = 0;
        if (elbowIdx < (int)joints_.size() && joints_[elbowIdx].x > 0) {
            offsetX = (int)((wrist.x - joints_[elbowIdx].x) * 0.3);
            offsetY = (int)((wrist.y - joints_[elbowIdx].y) * 0.3);
        }
        
        // 计算 ROI
        cv::Rect roi(
            (int)wrist.x - roiSize / 2 + offsetX,
            (int)wrist.y - roiSize / 2 + offsetY,
            roiSize, roiSize
        );
        
        // 裁剪到图像边界
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (roi.width > 50 && roi.height > 50) {
            HandResult result = detectHand(frame, roi);
            result.isRight = (wristIdx == 10);
            if (result.score > 0.005f && result.points.size() == 21) {
                hands_.push_back(result);
            }
        }
    }
}

HandResult MotionRecognizer::detectHand(const cv::Mat& frame, const cv::Rect& roi)
{
    HandResult result;
    result.score = 0;
    result.isRight = false;
    
    try {
        // 裁剪并预处理
        cv::Mat crop = frame(roi);
        cv::Mat resized, rgb;
        cv::resize(crop, resized, cv::Size(HAND_INPUT_SIZE, HAND_INPUT_SIZE));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
        
        // 转为 Tensor [1, 3, 224, 224]
        auto tensor = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kFloat32).clone();
        tensor = tensor.permute({2, 0, 1}).unsqueeze(0).to(device_);
        
        // 推理
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = handModel_.forward(inputs);
        
        // 处理输出 (list of 3 tensors)
        auto outputList = output.toList();
        auto xyz = outputList.get(0).toTensor().to(torch::kCPU);  // [1, 63]
        auto scoreT = outputList.get(1).toTensor().to(torch::kCPU);  // [1, 1]
        auto lrT = outputList.get(2).toTensor().to(torch::kCPU);  // [1, 1]
        
        result.score = scoreT[0][0].item<float>();
        result.isRight = lrT[0][0].item<float>() > 0.5f;
        
        // 提取关键点并映射到原图坐标
        auto xyzAccessor = xyz.accessor<float, 2>();
        for (int i = 0; i < 21; ++i) {
            float px_in_crop = xyzAccessor[0][i * 3 + 0];  // x in 224x224
            float py_in_crop = xyzAccessor[0][i * 3 + 1];  // y in 224x224
            
            // 映射回原图坐标
            float px = roi.x + (px_in_crop / HAND_INPUT_SIZE) * roi.width;
            float py = roi.y + (py_in_crop / HAND_INPUT_SIZE) * roi.height;
            result.points.push_back(cv::Point2f(px, py));
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Hand detection error: " << e.what() << std::endl;
    }
    
    return result;
}

void MotionRecognizer::runYOLO(const cv::Mat& frame)
{
    joints_.clear();
    joints_.resize(17, cv::Point2f(0.0f, 0.0f)); // 初始化 17 个关键点为 (0,0)
    
    try {
        // 准备输入 Tensor
        torch::Tensor inputTensor = matToTensor(frame);
        
        // 推理
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        auto output = yoloModel_.forward(inputs);
        
        // 处理输出 - YOLOv8-Pose 标准输出格式: [1, 56, 8400]
        // 56 = 4(bbox: cx,cy,w,h) + 1(conf) + 51(17 keypoints * 3)
        torch::Tensor outputTensor;
        
        if (output.isTensor()) {
            outputTensor = output.toTensor().to(torch::kCPU);
        } else if (output.isTuple()) {
            auto tuple = output.toTuple();
            outputTensor = tuple->elements()[0].toTensor().to(torch::kCPU);
        } else if (output.isList()) {
            auto list = output.toList();
            outputTensor = list.get(0).toTensor().to(torch::kCPU);
        }
        
        // 转置为 [1, 8400, 56] 便于处理
        outputTensor = outputTensor.transpose(1, 2).contiguous();
        auto accessor = outputTensor.accessor<float, 3>();
        
        float sx = (float)frame.cols / 640.0f;
        float sy = (float)frame.rows / 640.0f;
        
        // 找到置信度最高的检测框
        int bestIdx = -1;
        float bestConf = 0.25f; // 最低置信度阈值
        
        int numDetections = accessor.size(1); // 8400
        for (int i = 0; i < numDetections; ++i) {
            float conf = accessor[0][i][4]; // 置信度在索引4
            if (conf > bestConf) {
                bestConf = conf;
                bestIdx = i;
            }
        }
        
        // 如果找到有效检测，提取关键点
        if (bestIdx >= 0) {
            // 关键点数据从索引5开始，每个关键点3个值(x, y, conf)
            for (int k = 0; k < 17; ++k) {
                int baseIdx = 5 + k * 3;
                float kx = accessor[0][bestIdx][baseIdx + 0];     // x (基于 640 尺度)
                float ky = accessor[0][bestIdx][baseIdx + 1];     // y (基于 640 尺度)
                float kconf = accessor[0][bestIdx][baseIdx + 2];  // 关键点置信度
                
                // 归一化到[0,1]范围，与训练环境一致
                if (kconf > 0.25f) {  // 统一置信度阈值为0.25
                    joints_[k] = cv::Point2f(kx / 640.0f, ky / 640.0f);
                }
            }
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "YOLO inference error: " << e.what() << std::endl;
    }
}

void MotionRecognizer::runTCN() {
    try {
        // 将 [32, 12] 转置为 [1, 12, 32]
        std::vector<float> inputT(12 * 32);
        for (int c = 0; c < 12; ++c) {
            for (int t = 0; t < 32; ++t) {
                inputT[c * 32 + t] = frameBuffer_[t][c];
            }
        }

        // 创建输入 Tensor [1, 12, 32]
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor tensor = torch::from_blob(
            inputT.data(), 
            {1, 12, 32}, 
            options
        ).clone().to(device_);

        // 推理
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = tcnModel_.forward(inputs).toTensor().to(torch::kCPU);
        float* out = output.data_ptr<float>();

        // 标准 Softmax 处理
        float max_v = *std::max_element(out, out + 6);
        float sum = 0.0f;
        for (int i = 0; i < 6; ++i) {
            lastOutput_[i] = std::exp(out[i] - max_v);
            sum += lastOutput_[i];
        }
        for (int i = 0; i < 6; ++i) lastOutput_[i] /= sum;
        
        // 置信度检查：低于0.6强制判定为Background（与本地环境一致）
        float max_confidence = *std::max_element(lastOutput_.begin(), lastOutput_.end());
        if (max_confidence < 0.6f) {
            lastOutput_.assign(6, 0.0f);
            lastOutput_[0] = 1.0f;  // 100% Background
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "TCN inference error: " << e.what() << std::endl;
    }
}

void MotionRecognizer::getResult(std::map<std::string, float>* output) {
    static const char* names[6] = { "Background", "Stir-Frying", "Seasoning", "Cover Lid", "Uncover Lid", "Add Water" };
    output->clear();
    for (int i = 0; i < 6; ++i) (*output)[names[i]] = lastOutput_[i];
}

std::string MotionRecognizer::extractBone() {
    std::string s = "{\"joints\":[";
    
    // joints_ 中存储的是归一化到[0,1]的坐标
    // 需要转换为原始帧的像素坐标用于可视化
    int frameWidth = previewFrame_.empty() ? 640 : previewFrame_.cols;
    int frameHeight = previewFrame_.empty() ? 640 : previewFrame_.rows;
    
    for (size_t i = 0; i < joints_.size(); ++i) {
        // 将归一化坐标[0,1]转换为像素坐标
        // joints_存储的是基于640归一化的坐标，先转回640尺度，再缩放到实际帧尺寸
        int px = (int)(joints_[i].x * 640.0f * frameWidth / 640.0f);
        int py = (int)(joints_[i].y * 640.0f * frameHeight / 640.0f);
        
        s += "{\"x\":" + std::to_string(px) + ",\"y\":" + std::to_string(py) + "}";
        if (i + 1 < joints_.size()) s += ",";
    }
    s += "]}";
    return s;
}

std::string MotionRecognizer::extractHands() {
    // 返回格式: {"hands":[{"points":[{"x":1,"y":2},...], "score":0.95, "isRight":true}, ...]}
    std::string s = "{\"hands\":[";
    for (size_t h = 0; h < hands_.size(); ++h) {
        const auto& hand = hands_[h];
        s += "{\"points\":[";
        for (size_t i = 0; i < hand.points.size(); ++i) {
            s += "{\"x\":" + std::to_string((int)hand.points[i].x) + 
                 ",\"y\":" + std::to_string((int)hand.points[i].y) + "}";
            if (i + 1 < hand.points.size()) s += ",";
        }
        s += "],\"score\":" + std::to_string(hand.score) + 
             ",\"isRight\":" + (hand.isRight ? "true" : "false") + "}";
        if (h + 1 < hands_.size()) s += ",";
    }
    s += "]}";
    return s;
}

cv::Mat MotionRecognizer::generatePreview() { return previewFrame_; }