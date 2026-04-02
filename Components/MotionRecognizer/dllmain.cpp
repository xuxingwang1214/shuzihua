#include "pch.h"
#include "motion_recognizer.h"
#include <memory>
#include <objbase.h> // 使用 CoTaskMemAlloc

static std::shared_ptr<MotionRecognizer> recognizer = nullptr;

// 保证内存能被 C# 正确回收
wchar_t* stringToWcharPtr(const std::string& str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    wchar_t* buffer = (wchar_t*)CoTaskMemAlloc(len * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, buffer, len);
    return buffer;
}

extern "C" {
    __declspec(dllexport) void initialize() {
        if (!recognizer) recognizer = std::make_shared<MotionRecognizer>();
        recognizer->initialize();
    }

    __declspec(dllexport) void onSceneUpdate(int width, int height, unsigned char* data) {
        if (!recognizer || !data) return;
        cv::Mat img(height, width, CV_8UC3, data);
        recognizer->recognize(img);
    }

    __declspec(dllexport) wchar_t* queryResult() {
        if (!recognizer) return nullptr;
        std::map<std::string, float> output;
        recognizer->getResult(&output);
        std::string res = "";
        for (auto const& [name, score] : output) res += name + ":" + std::to_string(score) + "\n";
        return stringToWcharPtr(res);
    }

    __declspec(dllexport) wchar_t* extractBone() {
        if (!recognizer) return nullptr;
        return stringToWcharPtr(recognizer->extractBone());
    }

    __declspec(dllexport) wchar_t* extractHands() {
        if (!recognizer) return nullptr;
        return stringToWcharPtr(recognizer->extractHands());
    }
}