"""从视频提取一帧并测试 YOLOv8-Pose 模型"""

import onnxruntime as ort
import numpy as np
import cv2

# 模型路径
model_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\yolov8n-pose_end2end.onnx"
video_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\sample.mp4"

# 加载模型
print(f"加载模型: {model_path}")
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("\n模型信息:")
for inp in session.get_inputs():
    print(f"  输入 {inp.name}: {inp.shape}")
for out in session.get_outputs():
    print(f"  输出 {out.name}: {out.shape}")

# 从视频提取帧
print(f"\n加载视频: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频!")
    exit(1)

# 跳到第100帧（应该有人物）
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取帧!")
    exit(1)

print(f"帧尺寸: {frame.shape}")

# 保存原始帧用于查看
cv2.imwrite("test_frame.jpg", frame)
print("保存测试帧到 test_frame.jpg")

# 预处理
orig_h, orig_w = frame.shape[:2]
resized = cv2.resize(frame, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# 归一化到 0-1
normalized = rgb.astype(np.float32) / 255.0
# HWC -> CHW -> NCHW
input_data = normalized.transpose(2, 0, 1)[np.newaxis, ...]

print(f"\n输入数据形状: {input_data.shape}")
print(f"输入数据范围: [{input_data.min():.3f}, {input_data.max():.3f}]")

# 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})

out = outputs[0]
print(f"\n输出形状: {out.shape}")

# 分析输出
if len(out.shape) == 3 and out.shape[2] == 57:
    batch, num_det, values = out.shape
    
    # 检查置信度分布
    confs = out[0, :, 4]
    print(f"\n置信度统计:")
    print(f"  最小: {confs.min():.4f}")
    print(f"  最大: {confs.max():.4f}")
    print(f"  均值: {confs.mean():.4f}")
    print(f"  > 0.5: {np.sum(confs > 0.5)}")
    print(f"  > 0.25: {np.sum(confs > 0.25)}")
    print(f"  > 0.1: {np.sum(confs > 0.1)}")
    
    # 找最高置信度的检测
    best_idx = np.argmax(confs)
    best_conf = confs[best_idx]
    print(f"\n最高置信度检测: idx={best_idx}, conf={best_conf:.4f}")
    
    if best_conf > 0.01:
        det = out[0, best_idx]
        
        # bbox
        x1, y1, x2, y2 = det[0:4]
        print(f"\nBounding Box (640x640尺度):")
        print(f"  x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
        
        # 检查是否需要坐标转换
        print(f"\n检查坐标范围:")
        print(f"  x 范围: [{det[5:56:3].min():.1f}, {det[5:56:3].max():.1f}]")
        print(f"  y 范围: [{det[6:56:3].min():.1f}, {det[6:56:3].max():.1f}]")
        
        # 关键点
        print(f"\n关键点信息 (17个):")
        kpt_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 绘制结果
        draw_frame = frame.copy()
        sx = orig_w / 640.0
        sy = orig_h / 640.0
        
        keypoints = []
        for k in range(17):
            kx = det[5 + k*3 + 0]
            ky = det[5 + k*3 + 1]
            kc = det[5 + k*3 + 2]
            keypoints.append((kx * sx, ky * sy, kc))
            
            print(f"  {k}: {kpt_names[k]:15s} - x={kx:6.1f}, y={ky:6.1f}, conf={kc:.4f}")
            
            if kc > 0.3:
                cv2.circle(draw_frame, (int(kx * sx), int(ky * sy)), 5, (0, 255, 0), -1)
        
        # 绘制 bbox
        cv2.rectangle(draw_frame, 
                      (int(x1 * sx), int(y1 * sy)), 
                      (int(x2 * sx), int(y2 * sy)), 
                      (0, 255, 255), 2)
        
        cv2.imwrite("test_result.jpg", draw_frame)
        print("\n保存结果到 test_result.jpg")
else:
    print(f"意外的输出形状: {out.shape}")
    print(f"原始数据 (前100个值): {out.ravel()[:100]}")
