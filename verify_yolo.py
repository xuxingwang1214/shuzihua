"""验证 YOLO 模型输出"""

import onnxruntime as ort
import numpy as np
import cv2

model_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\yolov8n-pose_end2end.onnx"
video_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\sample.mp4"

print("=" * 60)
print("YOLO 姿态检测模型验证")
print("=" * 60)

# 加载模型
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("\n模型信息:")
for inp in session.get_inputs():
    print(f"  输入: {inp.name} - {inp.shape}")
for out in session.get_outputs():
    print(f"  输出: {out.name} - {out.shape}")

# 从视频提取帧
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取视频帧!")
    exit(1)

orig_h, orig_w = frame.shape[:2]
print(f"\n原始帧尺寸: {orig_w}x{orig_h}")

# 预处理
resized = cv2.resize(frame, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
input_data = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

# 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})
out = outputs[0]

print(f"\n输出形状: {out.shape}")

# 分析输出
if out.shape == (1, 17, 3):
    print("\n✓ 输出格式正确: [1, 17, 3] (17个关键点，每个点 x, y, conf)")
    
    sx = orig_w / 640.0
    sy = orig_h / 640.0
    
    kpt_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    print("\n关键点检测结果:")
    print("-" * 60)
    valid_count = 0
    for k in range(17):
        x, y, conf = out[0, k]
        # 映射回原图坐标
        orig_x = x * sx
        orig_y = y * sy
        
        status = "✓" if conf > 0.35 else "✗"
        if conf > 0.35:
            valid_count += 1
        
        print(f"{status} {k:2d}. {kpt_names[k]:15s}: x={orig_x:6.1f}, y={orig_y:6.1f}, conf={conf:.4f}")
    
    print("-" * 60)
    print(f"有效关键点数量 (conf > 0.35): {valid_count}/17")
    
    # 绘制结果
    draw_frame = frame.copy()
    for k in range(17):
        x, y, conf = out[0, k]
        if conf > 0.35:
            px, py = int(x * sx), int(y * sy)
            cv2.circle(draw_frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(draw_frame, str(k), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite("yolo_result.jpg", draw_frame)
    print("\n结果已保存到 yolo_result.jpg")
else:
    print(f"\n✗ 意外的输出格式: {out.shape}")
