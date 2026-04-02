using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenCvSharp;
using Newtonsoft.Json.Linq;

namespace MotionRecognizeTester
{
    public class MotionResult
    {
        public string name;
        public float score;
    }

    public static class MotionAdapter
    {
        private const string DllName = "MotionRecognizer.dll";

        // 对应 C++: void initialize()
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void initialize();

        // 对应 C++: void onSceneUpdate(int width, int height, unsigned char* data)
        // 注意顺序：Width, Height, Data
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void onSceneUpdate(int w, int h, IntPtr data);

        // 对应 C++: wchar_t* queryResult()
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr queryResult();

        // 对应 C++: wchar_t* extractBone()
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr extractBone();

        // 对应 C++: wchar_t* extractHands()
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr extractHands();

        public static void Initialize() => initialize();

        /// <summary>
        /// 接收 byte[] 数据，固定内存后传递给 C++
        /// </summary>
        public static void OnSceneUpdate(int w, int h, byte[] data)
        {
            if (data == null || data.Length == 0) return;

            // 固定内存，防止 GC 移动数组地址，确保 C++ 访问安全
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                onSceneUpdate(w, h, handle.AddrOfPinnedObject());
            }
            finally
            {
                handle.Free();
            }
        }

        public static List<MotionResult> QueryResult()
        {
            IntPtr ptr = queryResult();
            if (ptr == IntPtr.Zero) return new List<MotionResult>();

            // 使用 Uni 编码匹配 C++ 的 wchar_t*
            string raw = Marshal.PtrToStringUni(ptr);
            var list = new List<MotionResult>();

            if (string.IsNullOrEmpty(raw)) return list;

            string[] lines = raw.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                var p = line.Split(':');
                if (p.Length == 2 && float.TryParse(p[1], out float s))
                {
                    list.Add(new MotionResult { name = p[0], score = s });
                }
            }
            return list;
        }

        public static List<Point> GetKeypoints()
        {
            IntPtr ptr = extractBone();
            if (ptr == IntPtr.Zero) return new List<Point>();

            // 使用 Uni 编码匹配 C++ 的 wchar_t*
            string json = Marshal.PtrToStringUni(ptr);
            var list = new List<Point>();

            try
            {
                // 解析 C++ 返回的 JSON: {"joints":[{"x":123,"y":456},...]}
                var root = JObject.Parse(json);
                var joints = root["joints"];
                if (joints != null)
                {
                    foreach (var j in joints)
                    {
                        list.Add(new Point((int)j["x"], (int)j["y"]));
                    }
                }
            }
            catch
            {
                // 解析失败返回空列表
            }
            return list;
        }

        /// <summary>
        /// 获取手部关键点 (21点 x 手数)
        /// </summary>
        public static List<List<Point>> GetHands()
        {
            IntPtr ptr = extractHands();
            if (ptr == IntPtr.Zero) return new List<List<Point>>();

            string json = Marshal.PtrToStringUni(ptr);
            var result = new List<List<Point>>();

            try
            {
                // 解析 C++ 返回的 JSON: {"hands":[{"points":[{"x":1,"y":2},...],"score":0.9,"isRight":1},...],"count":2}
                var root = JObject.Parse(json);
                var hands = root["hands"];
                if (hands != null)
                {
                    foreach (var hand in hands)
                    {
                        var handPoints = new List<Point>();
                        var points = hand["points"];
                        if (points != null)
                        {
                            foreach (var p in points)
                            {
                                handPoints.Add(new Point((int)p["x"], (int)p["y"]));
                            }
                        }
                        result.Add(handPoints);
                    }
                }
            }
            catch
            {
                // 解析失败返回空列表
            }
            return result;
        }
    }
}