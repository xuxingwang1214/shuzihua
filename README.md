# MotionRecognizeBrief

Motion recognition project with Python scripts, a C++ runtime component, and a C# demo app.

## Project Structure

- `Components/MotionRecognizer`: C++ DLL project (Visual Studio)
- `Demos/MotionRecognizeTester`: C# test app
- Root Python scripts: model conversion, verification, and YOLO checks

## Build Notes

### C++ Component

1. Open `Components/MotionRecognizer/MotionRecognizer.sln` in Visual Studio.
2. Confirm third-party dependencies are available:
	- ONNX Runtime under `Components/MotionRecognizer/3rdParty/onnxruntime`
	- Libtorch is intentionally ignored from git and should be prepared locally.
3. Build `x64 Release` (or `Debug` as needed).

### C# Demo

1. Open `Demos/MotionRecognizeTester/MotionRecognizeTester.sln`.
2. Restore NuGet packages.
3. Build and run.

## Models and Git LFS

Model files are tracked by Git LFS:

- `*.pt`
- `*.onnx`
- `*.pth`

Check LFS files with:

```bash
git lfs ls-files
```

## What Is Ignored

The repository ignores build outputs and local environment artifacts, including:

- `.venv`, `.vs`, `bin`, `obj`, `x64`
- local libtorch directory
- binary runtime files such as `*.dll`, `*.exe`, `*.pdb`

## Quick Verification

Useful commands after clone:

```bash
git lfs install
git lfs pull
```

Then run Python checks from project root as needed (examples):

```bash
python verify_yolo.py
python verify_tcn_precision.py
```
