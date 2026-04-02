@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   TCN 模型转换 ^& DLL 编译部署工具
echo ============================================================
echo.

:: 设置路径 (批处理文件所在目录为 MotionRecognizeBrief)
set "ROOT_DIR=%~dp0"
set "MODELS_DIR=%ROOT_DIR%Demos\MotionRecognizeTester\models"
set "DLL_PROJECT_DIR=%ROOT_DIR%Components\MotionRecognizer"
set "OUTPUT_DIR=%ROOT_DIR%Demos\MotionRecognizeTester\bin\x64\Release"

:: 查找 Python (优先使用项目虚拟环境)
set "PYTHON="
if exist "%ROOT_DIR%..\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT_DIR%..\.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT_DIR%.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON=python"
    )
)

if "%PYTHON%"=="" (
    echo [错误] 未找到 Python，请安装 Python 并确保在 PATH 中
    pause
    exit /b 1
)
echo 使用 Python: %PYTHON%

:: 检查 tcn_action.pth 是否存在
if not exist "%MODELS_DIR%\tcn_action.pth" (
    echo [错误] 未找到模型文件
    pause
    exit /b 1
)

echo.
echo [1/4] 转换 PyTorch 模型为 ONNX...
echo.

call "%PYTHON%" "%ROOT_DIR%convert_tcn.py" "%MODELS_DIR%"
if errorlevel 1 (
    echo [错误] ONNX 导出失败!
    pause
    exit /b 1
)

echo.
echo [2/4] 查找 Visual Studio MSBuild...

:: 查找 MSBuild
set "MSBUILD="

:: 常见安装路径
for %%p in (
    "D:\VS2022\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
) do (
    if exist %%~p (
        set "MSBUILD=%%~p"
        goto :found_msbuild
    )
)

:: 使用 vswhere 查找
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
        if exist "%%i\MSBuild\Current\Bin\MSBuild.exe" (
            set "MSBUILD=%%i\MSBuild\Current\Bin\MSBuild.exe"
            goto :found_msbuild
        )
    )
)

:found_msbuild
if "%MSBUILD%"=="" (
    echo [错误] 未找到 MSBuild，请安装 Visual Studio 2022/2019
    pause
    exit /b 1
)
echo 找到 MSBuild: %MSBUILD%

echo.
echo [3/4] 编译 MotionRecognizer.dll (Release x64)...
echo.

call "%MSBUILD%" "%DLL_PROJECT_DIR%\MotionRecognizer.sln" /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal /nologo
if errorlevel 1 (
    echo.
    echo [错误] DLL 编译失败!
    pause
    exit /b 1
)

echo.
echo [4/4] 部署文件到输出目录...

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: 复制 DLL
if exist "%DLL_PROJECT_DIR%\x64\Release\MotionRecognizer.dll" (
    echo   复制 MotionRecognizer.dll
    copy /Y "%DLL_PROJECT_DIR%\x64\Release\MotionRecognizer.dll" "%OUTPUT_DIR%\" >nul
) else (
    echo [警告] 未找到编译后的 DLL
)

:: 复制 ONNX 模型
if exist "%MODELS_DIR%\tcn_action.onnx" (
    echo   复制 tcn_action.onnx
    copy /Y "%MODELS_DIR%\tcn_action.onnx" "%OUTPUT_DIR%\" >nul
)

:: 复制依赖 DLL
set "OPENCV_DLL=%DLL_PROJECT_DIR%\..\.env\opencv\build\x64\vc16\bin\opencv_world490.dll"
if exist "%OPENCV_DLL%" (
    echo   复制 opencv_world490.dll
    copy /Y "%OPENCV_DLL%" "%OUTPUT_DIR%\" >nul
)

set "ONNX_DLL=%DLL_PROJECT_DIR%\3rdParty\onnxruntime\runtimes\win-x64\native\onnxruntime.dll"
if exist "%ONNX_DLL%" (
    echo   复制 onnxruntime.dll
    copy /Y "%ONNX_DLL%" "%OUTPUT_DIR%\" >nul
)

echo.
echo ============================================================
echo   构建和部署完成!
echo ============================================================
echo.
echo 输出目录: %OUTPUT_DIR%
echo.
echo 已部署的关键文件:
echo   - MotionRecognizer.dll
echo   - tcn_action.onnx
echo.

pause
