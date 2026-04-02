"""深入分析 YOLOv8-Pose 端到端模型的输出格式"""

import onnxruntime as ort
import numpy as np
import cv2

model_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\yolov8n-pose_end2end.onnx"
video_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\sample.mp4"

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# 从视频提取帧
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

orig_h, orig_w = frame.shape[:2]
resized = cv2.resize(frame, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
input_data = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})
out = outputs[0]

print("=== 深入分析输出格式 ===")
print(f"输出形状: {out.shape}")  # [1, 300, 57]

# 获取最高置信度的检测
confs = out[0, :, 4]
best_idx = np.argmax(confs)
det = out[0, best_idx]

print(f"\n最佳检测 (idx={best_idx}, conf={det[4]:.4f}):")
print(f"\n所有 57 个值:")
for i in range(57):
    print(f"  [{i:2d}] = {det[i]:10.4f}", end="")
    if i < 4:
        print(f"  <- bbox")
    elif i == 4:
        print(f"  <- confidence")
    elif i >= 5:
        kpt_idx = (i - 5) // 3
        kpt_offset = (i - 5) % 3
        labels = ['x', 'y', 'conf']
        kpt_names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
                     "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                     "l_wrist", "r_wrist", "l_hip", "r_hip",
                     "l_knee", "r_knee", "l_ankle", "r_ankle"]
        print(f"  <- kpt[{kpt_idx}].{labels[kpt_offset]} ({kpt_names[kpt_idx]})")
    else:
        print()

# 让我尝试另一种解释：可能是 [x, y, conf] 顺序但数据错位了
print("\n\n=== 检查数据是否错位 ===")
print("假设1: 标准格式 [x, y, conf]")
print("假设2: 数据可能从索引6开始而不是5")

# 尝试不同的起始偏移
for start in [5, 6, 7]:
    print(f"\n--- 从索引 {start} 开始读取关键点 ---")
    valid_kpts = 0
    for k in range(min(5, (57 - start) // 3)):  # 只打印前5个关键点
        kx = det[start + k*3 + 0]
        ky = det[start + k*3 + 1]
        kc = det[start + k*3 + 2]
        print(f"  kpt[{k}]: x={kx:8.2f}, y={ky:8.2f}, c={kc:8.2f}")
        if 0 <= kx <= 640 and 0 <= ky <= 640:
            valid_kpts += 1
    print(f"  有效关键点 (x,y in 0-640): {valid_kpts}")

# 尝试另一种可能：17个关键点可能是按 (x, y) 对排列，后面跟置信度
print("\n\n=== 假设3: 关键点按 [17个x, 17个y, 17个conf] 排列 ===")
# 如果是这样，57-5=52 不能整除51，所以不太可能

# 检查原始数据模式
print("\n=== 检查数值范围模式 ===")
vals = det[5:].reshape(-1)  # 52个值
print(f"索引 5-56 的值 ({len(vals)} 个):")
print(f"  范围: [{vals.min():.2f}, {vals.max():.2f}]")
print(f"  0-1 范围内: {np.sum((vals >= 0) & (vals <= 1))}")
print(f"  0-640 范围内: {np.sum((vals >= 0) & (vals <= 640))}")
print(f"  > 640: {np.sum(vals > 640)}")

# 按位置分组检查
print("\n按 mod 3 分组:")
for r in range(3):
    group = det[5+r::3]
    print(f"  位置 {r} (每3个取1个): 范围 [{group.min():.2f}, {group.max():.2f}]")
