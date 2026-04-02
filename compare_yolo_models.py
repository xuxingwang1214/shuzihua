"""比较新旧 YOLOv8-Pose 模型的输出格式"""

import onnxruntime as ort
import numpy as np
import cv2

# 模型路径
new_model = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\yolov8n-pose_end2end.onnx"
backup_model = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\models\yolov8n-pose_end2end_backup.onnx"
video_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\sample.mp4"

import os
print(f"新模型存在: {os.path.exists(new_model)}, 大小: {os.path.getsize(new_model)/1024/1024:.2f} MB")
print(f"备份模型存在: {os.path.exists(backup_model)}, 大小: {os.path.getsize(backup_model)/1024/1024:.2f} MB" if os.path.exists(backup_model) else "备份模型不存在")

# 从视频提取帧
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

orig_h, orig_w = frame.shape[:2]
resized = cv2.resize(frame, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
input_data = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

def analyze_model(model_path, name):
    print(f"\n{'='*60}")
    print(f"分析 {name}: {model_path}")
    print('='*60)
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    print("\n输入:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    
    print("\n输出:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    for i, out in enumerate(outputs):
        print(f"\n输出 {i} 详情:")
        print(f"  形状: {out.shape}")
        print(f"  范围: [{out.min():.4f}, {out.max():.4f}]")
        
        if len(out.shape) == 3:
            batch, dim1, dim2 = out.shape
            print(f"  解释: batch={batch}, {dim1} 个检测/关键点, 每个 {dim2} 个值")
            
            # 检查是否是 [1, 300, 57] 端到端格式
            if dim2 == 57:
                confs = out[0, :, 4]
                best_idx = np.argmax(confs)
                det = out[0, best_idx]
                print(f"\n  最佳检测 (idx={best_idx}):")
                print(f"    bbox: x1={det[0]:.1f}, y1={det[1]:.1f}, x2={det[2]:.1f}, y2={det[3]:.1f}")
                print(f"    conf: {det[4]:.4f}")
                print(f"    前3个关键点:")
                for k in range(3):
                    print(f"      kpt[{k}]: {det[5+k*3]:.2f}, {det[5+k*3+1]:.2f}, {det[5+k*3+2]:.2f}")
            
            # 检查是否是 [1, 17, 3] 格式
            elif dim1 == 17 and dim2 == 3:
                print(f"\n  关键点 (17, 3) 格式:")
                for k in range(5):
                    print(f"    kpt[{k}]: x={out[0,k,0]:.2f}, y={out[0,k,1]:.2f}, conf={out[0,k,2]:.4f}")
            
            # 其他格式
            else:
                print(f"\n  前5个检测/行的数据:")
                for j in range(min(5, dim1)):
                    row = out[0, j, :min(10, dim2)]
                    print(f"    [{j}]: {row}")

# 分析两个模型
analyze_model(new_model, "新模型")
analyze_model(backup_model, "备份模型")

# 也检查一下 Components 目录下的模型
comp_model = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Components\MotionRecognizer\models\yolov8n-pose_end2end.onnx"
if os.path.exists(comp_model):
    analyze_model(comp_model, "Components目录模型")
