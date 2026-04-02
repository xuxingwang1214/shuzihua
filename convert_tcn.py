"""
TCN 模型 PyTorch -> ONNX 转换脚本
用法: python convert_tcn.py <models_dir>
"""
import torch
import torch.nn as nn
import os
import sys

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

def main():
    if len(sys.argv) < 2:
        models_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(models_dir, 'Demos', 'MotionRecognizeTester', 'models')
    else:
        models_dir = sys.argv[1]
    
    pth_path = os.path.join(models_dir, 'tcn_action.pth')
    onnx_path = os.path.join(models_dir, 'tcn_action.onnx')
    
    if not os.path.exists(pth_path):
        print(f'[错误] 未找到模型文件: {pth_path}')
        sys.exit(1)
    
    print(f'加载模型: {pth_path}')
    model = SimpleTCN(input_channels=12, num_classes=6)
    model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=False))
    model.eval()
    
    print(f'导出 ONNX: {onnx_path}')
    dummy_input = torch.randn(1, 12, 32)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_skels'],
        output_names=['output_actions'],
        dynamo=False
    )
    
    size = os.path.getsize(onnx_path)
    print(f'模型大小: {size} bytes ({size/1024:.1f} KB)')
    print('ONNX 导出成功!')

if __name__ == '__main__':
    main()
