"""完整验证 TCN ONNX 模型输出精度"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import os

# TCN 模型定义 (与原始 pth 模型一致)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.res = nn.Conv1d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.res(x))

class SimpleTCN(nn.Module):
    def __init__(self, input_channels=12, num_classes=6):
        super().__init__()
        self.layer1 = TCNBlock(input_channels, 64)
        self.layer2 = TCNBlock(64, 128)
        self.layer3 = TCNBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# 路径
pth_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\models\tcn_action.pth"
onnx_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\tcn_action.onnx"

print("=" * 60)
print("TCN 模型精度验证")
print("=" * 60)

# 加载 PyTorch 模型
print("\n加载 PyTorch 模型...")
model = SimpleTCN(input_channels=12, num_classes=6)
state_dict = torch.load(pth_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# 加载 ONNX 模型
print(f"加载 ONNX 模型: {onnx_path}")
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# 测试多组随机输入
print("\n测试 10 组随机输入:")
print("-" * 60)

max_diff_overall = 0
for i in range(10):
    # 生成随机输入
    np.random.seed(i * 42)
    test_input = np.random.randn(1, 12, 32).astype(np.float32)
    
    # PyTorch 推理
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input)).numpy()
    
    # ONNX 推理
    onnx_output = ort_session.run(None, {'input_skels': test_input})[0]
    
    # 计算差异
    diff = np.abs(torch_output - onnx_output)
    max_diff = diff.max()
    max_diff_overall = max(max_diff_overall, max_diff)
    
    # 应用 softmax 检查概率输出
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    torch_probs = softmax(torch_output[0])
    onnx_probs = softmax(onnx_output[0])
    prob_diff = np.abs(torch_probs - onnx_probs)
    
    print(f"测试 {i+1}: max_logit_diff={max_diff:.6f}, max_prob_diff={prob_diff.max():.6f}")

print("-" * 60)
print(f"\n最大 logit 差异: {max_diff_overall:.6f}")

if max_diff_overall < 0.001:
    print("✓ 模型精度验证通过！ONNX 与 PyTorch 输出高度一致")
else:
    print("✗ 警告: ONNX 与 PyTorch 输出存在较大差异")

# 详细查看一个输出
print("\n\n详细输出对比 (测试 1):")
print("-" * 60)
np.random.seed(42)
test_input = np.random.randn(1, 12, 32).astype(np.float32)

with torch.no_grad():
    torch_output = model(torch.from_numpy(test_input)).numpy()
onnx_output = ort_session.run(None, {'input_skels': test_input})[0]

labels = ["Background", "Stir-Frying", "Seasoning", "Cover Lid", "Uncover Lid", "Add Water"]
print(f"{'类别':15s} | {'PyTorch':>10s} | {'ONNX':>10s} | {'差值':>10s}")
print("-" * 60)
for i, label in enumerate(labels):
    print(f"{label:15s} | {torch_output[0,i]:10.4f} | {onnx_output[0,i]:10.4f} | {abs(torch_output[0,i]-onnx_output[0,i]):10.6f}")

# Softmax 后的概率
print("\nSoftmax 后的概率:")
torch_probs = np.exp(torch_output[0] - np.max(torch_output[0]))
torch_probs = torch_probs / torch_probs.sum()
onnx_probs = np.exp(onnx_output[0] - np.max(onnx_output[0]))
onnx_probs = onnx_probs / onnx_probs.sum()

print(f"{'类别':15s} | {'PyTorch':>10s} | {'ONNX':>10s}")
print("-" * 60)
for i, label in enumerate(labels):
    print(f"{label:15s} | {torch_probs[i]*100:9.2f}% | {onnx_probs[i]*100:9.2f}%")
