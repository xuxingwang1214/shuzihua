#include "pch.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <objbase.h>
#include <Windows.h>

using namespace std;

wstring gbkToWstring(const string& str)
{
    if (str.empty()) return wstring();
    int size_needed = ::MultiByteToWideChar(936, 0, str.c_str(), -1, nullptr, 0);
    if (size_needed == 0) return wstring();
    wstring wstr(size_needed - 1, L'\0');
    ::MultiByteToWideChar(936, 0, str.c_str(), -1, &wstr[0], size_needed);
    return wstr;
}

wchar_t* mapToOutputString(map<string, float> input)
{
    wstring resultW;
    resultW.reserve(input.size() * 32);
    for (const auto& kv : input)
    {
        wstring keyW = gbkToWstring(kv.first);
        wostringstream oss;
        oss << keyW << L":" << fixed << setprecision(2) << kv.second << L"\n";
        resultW += oss.str();
    }

    if (resultW.empty())
    {
        const wchar_t emptyStr[] = L"";
        wchar_t* buffer = (wchar_t*)CoTaskMemAlloc(sizeof(emptyStr));
        if (buffer) wcscpy_s(buffer, 1, emptyStr);
        return buffer;
    }

    size_t size = (resultW.length() + 1) * sizeof(wchar_t);
    wchar_t* buffer = (wchar_t*)CoTaskMemAlloc(size);
    if (buffer) wcscpy_s(buffer, resultW.length() + 1, resultW.c_str());
    return buffer;
}

wchar_t* wstringToOutputString(wstring resultW)
{
    if (resultW.empty())
    {
        const wchar_t emptyStr[] = L"";
        wchar_t* buffer = (wchar_t*)CoTaskMemAlloc(sizeof(emptyStr));
        if (buffer) wcscpy_s(buffer, 1, emptyStr);
        return buffer;
    }

    size_t size = (resultW.length() + 1) * sizeof(wchar_t);
    wchar_t* buffer = (wchar_t*)CoTaskMemAlloc(size);
    if (buffer) wcscpy_s(buffer, resultW.length() + 1, resultW.c_str());
    return buffer;
}
