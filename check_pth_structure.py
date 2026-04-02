"""检查原始 pth 模型的结构"""

import torch

pth_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\models\tcn_action.pth"

state_dict = torch.load(pth_path, map_location='cpu')

print("原始 pth 模型的权重键名:")
print("=" * 60)
for key in sorted(state_dict.keys()):
    shape = state_dict[key].shape
    print(f"{key}: {shape}")

print("\n\n权重键名列表:")
print(list(state_dict.keys()))
