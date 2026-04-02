"""检查 YOLOv8-Pose 模型的实际输出格式"""

import onnxruntime as ort
import numpy as np
import cv2
import os

# 模型路径
model_path = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release\yolov8n-pose_end2end.onnx"

# 加载模型
print(f"加载模型: {model_path}")
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# 打印输入输出信息
print("\n=== 模型信息 ===")
print("输入:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")

print("\n输出:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

# 创建测试输入
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"\n=== 测试推理 ===")

# 创建一个有内容的测试图像（比全黑好）
dummy_image = np.random.randn(1, 3, 640, 640).astype(np.float32) * 0.1 + 0.5

outputs = session.run(None, {input_name: dummy_image})

for i, out in enumerate(outputs):
    print(f"\n输出 {i}:")
    print(f"  形状: {out.shape}")
    print(f"  类型: {out.dtype}")
    print(f"  最小值: {out.min():.4f}")
    print(f"  最大值: {out.max():.4f}")
    print(f"  均值: {out.mean():.4f}")
    
    if len(out.shape) == 3:
        # 假设 [batch, num_detections, values]
        batch, num_det, values = out.shape
        print(f"\n  每个检测有 {values} 个值")
        
        # 查看前几个检测的置信度
        if values >= 5:
            print(f"\n  前10个检测的置信度 (索引4):")
            for j in range(min(10, num_det)):
                conf = out[0, j, 4]
                print(f"    检测 {j}: conf={conf:.4f}")
            
            # 找最高置信度
            confs = out[0, :, 4]
            max_idx = np.argmax(confs)
            print(f"\n  最高置信度检测: idx={max_idx}, conf={confs[max_idx]:.4f}")
            
            if confs[max_idx] > 0.1:
                det = out[0, max_idx]
                print(f"\n  该检测的全部 {values} 个值:")
                for k in range(values):
                    print(f"    [{k}] = {det[k]:.4f}")

# 现在用真实图像测试
print("\n\n=== 使用真实图像测试 ===")
test_images_dir = r"c:\Users\wangzhicheng\Desktop\MotionRecognizeBrief(1)\MotionRecognizeBrief\Demos\MotionRecognizeTester\bin\x64\Release"
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if image_files:
    img_path = os.path.join(test_images_dir, image_files[0])
    print(f"测试图像: {img_path}")
    
    img = cv2.imread(img_path)
    if img is not None:
        print(f"原始尺寸: {img.shape}")
        
        # 预处理: resize, BGR->RGB, HWC->CHW, 归一化
        resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = chw[np.newaxis, ...]  # [1, 3, 640, 640]
        
        outputs = session.run(None, {input_name: input_data})
        
        out = outputs[0]
        print(f"\n输出形状: {out.shape}")
        
        if len(out.shape) == 3 and out.shape[2] >= 5:
            confs = out[0, :, 4]
            max_idx = np.argmax(confs)
            max_conf = confs[max_idx]
            print(f"最高置信度: {max_conf:.4f} (检测 {max_idx})")
            
            # 统计置信度分布
            above_05 = np.sum(confs > 0.5)
            above_025 = np.sum(confs > 0.25)
            above_01 = np.sum(confs > 0.1)
            print(f"\n置信度分布:")
            print(f"  > 0.5: {above_05} 个")
            print(f"  > 0.25: {above_025} 个")
            print(f"  > 0.1: {above_01} 个")
            
            if max_conf > 0.1:
                det = out[0, max_idx]
                print(f"\n最佳检测的详细信息:")
                print(f"  bbox: x1={det[0]:.1f}, y1={det[1]:.1f}, x2={det[2]:.1f}, y2={det[3]:.1f}")
                print(f"  conf: {det[4]:.4f}")
                
                if out.shape[2] == 57:
                    print(f"\n  关键点 (17个 x 3):")
                    for k in range(17):
                        kx = det[5 + k*3 + 0]
                        ky = det[5 + k*3 + 1]
                        kc = det[5 + k*3 + 2]
                        print(f"    点 {k}: x={kx:.1f}, y={ky:.1f}, conf={kc:.4f}")
else:
    print("没有找到测试图像")
