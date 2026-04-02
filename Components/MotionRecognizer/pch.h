#pragma once
#ifndef PCH_H
#define PCH_H

#include "framework.h"
#include <map>
#include <string>
#include <opencv2/opencv.hpp>

// LibTorch headers (必须在其他头文件之前)
#include <torch/script.h>
#include <torch/torch.h>

std::wstring gbkToWstring(const std::string& str);
wchar_t* mapToOutputString(std::map<std::string, float> input);
wchar_t* wstringToOutputString(std::wstring resultW);

extern "C" _declspec(dllexport) void initialize();
extern "C" _declspec(dllexport) void onSceneUpdate(int width, int height, unsigned char* data);
extern "C" _declspec(dllexport) wchar_t* queryResult();
extern "C" _declspec(dllexport) wchar_t* extractBone();
extern "C" _declspec(dllexport) void generatePreview(int& width, int& height, unsigned char* data);

#endif // PCH_H
