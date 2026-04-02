using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MotionRecognizeTester
{
    internal class Program
    {
        // YOLO 标准骨架连接顺序
        static readonly int[,] Skeleton = new int[,] {
            {5,7},{7,9},{6,8},{8,10},{5,6},{5,11},{6,12},
            {11,12},{11,13},{13,15},{12,14},{14,16}
        };

        // 手部 21 个关键点连接顺序
        static readonly int[,] HandSkeleton = new int[,] {
            {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
            {0,9},{9,10},{10,11},{11,12},{0,13},{13,14},{14,15},{15,16},
            {0,17},{17,18},{18,19},{19,20}
        };

        const float SCORE_THRESHOLD = 0.55f;
        const int SMOOTH_WINDOW = 7;
        static Queue<string> actionQueue = new Queue<string>();

        static void Main(string[] args)
        {
            Console.WriteLine("正在初始化动作识别与手部模型...");
            MotionAdapter.Initialize();

            // 手部模型
            HandLandmarkDetector handDetector = new HandLandmarkDetector("hand_landmark_model.onnx");
            PalmDetector palmDetector = new PalmDetector("palm_detection_mediapipe.onnx");

            VideoCapture capture = new VideoCapture("sample.mp4");
            if (!capture.IsOpened()) { Console.WriteLine("无法打开视频文件"); return; }

            // 修复点 1: 更换为兼容性更好的 XVID 编码器，并显式指定 Size
            Size frameSize = new Size(capture.FrameWidth, capture.FrameHeight);
            using (capture)
            using (VideoWriter writer = new VideoWriter("final_result.mp4", VideoWriter.FourCC("XVID"), capture.Fps, frameSize))
            using (Mat frame = new Mat())
            {
                while (capture.Read(frame) && !frame.Empty())
                {
                    // 修复点 2: 核心内存安全修复。
                    // 强制确保图像是 8位3通道 BGR 格式，防止因单通道或4通道导致的内存越界
                    if (frame.Type() != MatType.CV_8UC3)
                    {
                        Cv2.CvtColor(frame, frame, ColorConversionCodes.GRAY2BGR);
                    }

                    // 修复点 3: 显式计算 buffer 长度 (W * H * 3)，不再依赖 frame.Total()
                    int expectedLength = frame.Width * frame.Height * 3;
                    byte[] buffer = new byte[expectedLength];

                    // 将 Mat 数据安全拷贝到托管字节数组
                    Marshal.Copy(frame.Data, buffer, 0, expectedLength);

                    // 传递给 DLL (内部已做 Pinned 固定)
                    MotionAdapter.OnSceneUpdate(frame.Width, frame.Height, buffer);

                    // 2. 获取结果并简化输出
                    var results = MotionAdapter.QueryResult();
                    var sorted = results.OrderByDescending(r => r.score).ToList();

                    // --- 精简版控制台输出：直接在一行内显示所有分值 ---
                    string debugInfo = "";
                    foreach (var res in results)
                    {
                        // 格式：动作名:分数%
                        debugInfo += $"{res.name[0]}: {res.score:P0} | ";
                    }
                    // 使用 \r 回到行首，实现原地刷新，不刷屏
                    Console.Write($"\r{debugInfo} 推理结果: {sorted[0].name,-12}");

                    // 保持原有的平滑逻辑
                    string rawAction = (sorted.Count > 0 && sorted[0].score >= SCORE_THRESHOLD) ? sorted[0].name : "Background";

                    actionQueue.Enqueue(rawAction);
                    if (actionQueue.Count > SMOOTH_WINDOW) actionQueue.Dequeue();
                    string smoothAction = actionQueue.GroupBy(a => a).OrderByDescending(g => g.Count()).First().Key;

                  

                    // 3. 绘制人体骨架
                    var keypoints = MotionAdapter.GetKeypoints();
                    // 绘制骨架连接线
                    for (int i = 0; i < Skeleton.GetLength(0); i++)
                    {
                        int a = Skeleton[i, 0];
                        int b = Skeleton[i, 1];
                        if (a < keypoints.Count && b < keypoints.Count)
                        {
                            if (keypoints[a].X > 0 && keypoints[b].X > 0)
                                Cv2.Line(frame, keypoints[a], keypoints[b], Scalar.Yellow, 2);
                        }
                    }
                    // 绘制所有骨架点
                    for (int i = 0; i < keypoints.Count; i++)
                    {
                        if (keypoints[i].X > 0 && keypoints[i].Y > 0)
                        {
                            Cv2.Circle(frame, keypoints[i], 5, Scalar.Red, -1);
                            Cv2.PutText(frame, i.ToString(), new Point(keypoints[i].X + 8, keypoints[i].Y + 5), 
                                HersheyFonts.HersheyPlain, 0.8, Scalar.White, 1);
                        }
                    }

                    // 4. 检测手掌位置 (高效方式：先检测手掌，再检测关键点)
                    var palms = palmDetector.Detect(frame);
                    foreach (var palm in palms)
                    {
                        // 根据手掌检测框进行关键点检测
                        HandResult hResult = handDetector.Infer(frame, palm.roi);
                        if (hResult.Score > 0.2f)
                        {
                            Scalar color = hResult.IsRight ? Scalar.Red : Scalar.Green;
                            
                            // 绘制所有手部关键点
                            foreach (Point p in hResult.Points)
                            {
                                Cv2.Circle(frame, p, 4, color, -1);
                            }

                            // 绘制手部骨架连接线
                            for (int j = 0; j < HandSkeleton.GetLength(0); j++)
                            {
                                int p1 = HandSkeleton[j, 0];
                                int p2 = HandSkeleton[j, 1];
                                if (p1 < hResult.Points.Count && p2 < hResult.Points.Count)
                                {
                                    Cv2.Line(frame, hResult.Points[p1], hResult.Points[p2], color, 2);
                                }
                            }
                        }
                    }
                    
                    // 5. 同时检测腕部附近的手部 (保留原有功能)
                    int wristRoiSize = 160;
                    int[] wristIndices = { 9, 10 };
                    foreach (int idx in wristIndices)
                    {
                        if (idx < keypoints.Count && keypoints[idx].X > 0)
                        {
                            Rect rawRoi = new Rect(keypoints[idx].X - wristRoiSize / 2, keypoints[idx].Y - wristRoiSize / 2, wristRoiSize, wristRoiSize);
                            Rect safeRoi = rawRoi.Intersect(new Rect(0, 0, frame.Width, frame.Height));

                            if (safeRoi.Width > 10 && safeRoi.Height > 10)
                            {
                                HandResult hResult = handDetector.Infer(frame, safeRoi);
                                if (hResult.Score > 0.1f)
                                {
                                    Scalar color = hResult.IsRight ? Scalar.Red : Scalar.Blue;
                                    Cv2.Rectangle(frame, safeRoi, Scalar.Cyan, 1);
                                    foreach (Point p in hResult.Points) Cv2.Circle(frame, p, 3, color, -1);

                                    // 绘制手部骨架连接线
                                    for (int j = 0; j < HandSkeleton.GetLength(0); j++)
                                    {
                                        int p1 = HandSkeleton[j, 0];
                                        int p2 = HandSkeleton[j, 1];
                                        Cv2.Line(frame, hResult.Points[p1], hResult.Points[p2], color, 1);
                                    }
                                }
                            }
                        }
                    }

                    // 6. 渲染 UI
                    Cv2.Rectangle(frame, new Rect(0, 0, 450, 60), Scalar.Black, -1);
                    Cv2.PutText(frame, smoothAction, new Point(20, 40), HersheyFonts.HersheyComplex, 1.0, Scalar.Lime, 2);

                    writer.Write(frame);
                    Cv2.ImShow("Action recognition", frame);
                    if (Cv2.WaitKey(1) == 27) break;
                }
            }
            Console.WriteLine("处理完成，结果已保存至 final_result.mp4");
        }
    }

    public class HandLandmarkDetector
    {
        private InferenceSession session;
        public HandLandmarkDetector(string modelPath)
        {
            var options = new SessionOptions();
            try { options.AppendExecutionProvider_CUDA(0); } catch { }
            session = new InferenceSession(modelPath, options);
        }

        public HandResult Infer(Mat src, Rect roi)
        {
            using (Mat crop = new Mat(src, roi))
            using (Mat resized = new Mat())
            {
                Cv2.Resize(crop, resized, new Size(224, 224));
                Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);
                resized.ConvertTo(resized, MatType.CV_32FC3, 1f / 255f);

                var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                for (int y = 0; y < 224; y++)
                    for (int x = 0; x < 224; x++)
                    {
                        Vec3f px = resized.At<Vec3f>(y, x);
                        inputTensor[0, 0, y, x] = px.Item0;
                        inputTensor[0, 1, y, x] = px.Item1;
                        inputTensor[0, 2, y, x] = px.Item2;
                    }

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
                using (var outputs = session.Run(inputs))
                {
                    var xyz = outputs.First(o => o.Name == "xyz_x21").AsEnumerable<float>().ToArray();
                    var score = outputs.First(o => o.Name == "hand_score").AsEnumerable<float>().First();
                    var lr = outputs.First(o => o.Name == "lefthand_0_or_righthand_1").AsEnumerable<float>().First();

                    List<Point> pts = new List<Point>();
                    for (int i = 0; i < 21; i++)
                    {
                        // 映射回原图坐标
                        int px = roi.X + (int)(xyz[i * 3 + 0] * roi.Width);
                        int py = roi.Y + (int)(xyz[i * 3 + 1] * roi.Height);
                        pts.Add(new Point(px, py));
                    }
                    return new HandResult { Points = pts, Score = score, IsRight = lr > 0.5f };
                }
            }
        }
    }

    public class HandResult { public List<Point> Points; public float Score; public bool IsRight; }

    public class PalmDetection
    {
        public Rect roi;
        public float score;
    }

    public class PalmDetector
    {
        private InferenceSession session;
        public PalmDetector(string modelPath)
        {
            var options = new SessionOptions();
            try { options.AppendExecutionProvider_CUDA(0); } catch { }
            session = new InferenceSession(modelPath, options);
        }

        public List<PalmDetection> Detect(Mat frame)
        {
            using (Mat resized = new Mat())
            {
                Cv2.Resize(frame, resized, new Size(256, 256));
                Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);
                resized.ConvertTo(resized, MatType.CV_32FC3, 1f / 255f);

                var inputTensor = new DenseTensor<float>(new[] { 1, 3, 256, 256 });
                for (int y = 0; y < 256; y++)
                    for (int x = 0; x < 256; x++)
                    {
                        Vec3f px = resized.At<Vec3f>(y, x);
                        inputTensor[0, 0, y, x] = px.Item0;
                        inputTensor[0, 1, y, x] = px.Item1;
                        inputTensor[0, 2, y, x] = px.Item2;
                    }

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
                List<PalmDetection> results = new List<PalmDetection>();

                try
                {
                    using (var outputs = session.Run(inputs))
                    {
                        // 获取检测框和置信度
                        var detections = outputs.FirstOrDefault(o => o.Name == "detections");
                        if (detections == null) return results;

                        var boxes = detections.AsEnumerable<float>().ToArray();
                        
                        // MediaPipe 手掌检测输出格式: [bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax, score, ...]
                        // 每个检测对象通常是 12 个值（4个框坐标 + 配置信息）
                        int stride = 12;
                        for (int i = 0; i < boxes.Length; i += stride)
                        {
                            if (i + 4 < boxes.Length)
                            {
                                float ymin = boxes[i] * frame.Height;
                                float xmin = boxes[i + 1] * frame.Width;
                                float ymax = boxes[i + 2] * frame.Height;
                                float xmax = boxes[i + 3] * frame.Width;
                                float score = boxes[i + 4];

                                if (score > 0.5f)  // 置信度阈值
                                {
                                    int x = Math.Max(0, (int)xmin);
                                    int y = Math.Max(0, (int)ymin);
                                    int w = Math.Min(frame.Width - x, (int)(xmax - xmin));
                                    int h = Math.Min(frame.Height - y, (int)(ymax - ymin));

                                    // 扩展 ROI 以获得更好的关键点检测
                                    int pad = (int)(Math.Max(w, h) * 0.2);
                                    x = Math.Max(0, x - pad);
                                    y = Math.Max(0, y - pad);
                                    w = Math.Min(frame.Width - x, w + pad * 2);
                                    h = Math.Min(frame.Height - y, h + pad * 2);

                                    results.Add(new PalmDetection 
                                    { 
                                        roi = new Rect(x, y, w, h),
                                        score = score
                                    });
                                }
                            }
                        }
                    }
                }
                catch { }

                return results;
            }
        }
    }
}