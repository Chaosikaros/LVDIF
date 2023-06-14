using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using UnityEditor;

namespace ChaosIkaros.LVDIF
{
    public class CudaUtility
    {
        /// <summary>
        /// Output overview info of CUDA device in editor console
        /// </summary>
        [DllImport("CudaUnity")]
        public static extern void GetDeviceInfo();

        /// <summary>
        /// Get number of cuda device number
        /// </summary>
        [DllImport("CudaUnity")]
        public static extern int GetDeviceNumber();

        /// <summary>
        /// Set cuda device by device ID
        /// </summary>
        [DllImport("CudaUnity")]
        public static extern void SetDevice(int deviceID);

        /// <summary>
        /// Get cudaLimitMallocHeapSize in editor console. default size = 8388608bit, bit = 8mb
        /// </summary>
        [DllImport("CudaUnity")]
        public static extern int GetMallocHeapSize();

        /// <summary>
        /// Set MallocHeapSize in bit, default size = 8388608bit, bit = 8mb
        /// </summary>
        /// <param name="MallocHeapSize"></param>
        /// <returns></returns>
        [DllImport("CudaUnity")]
        public static extern int SetMallocHeapSize(int MallocHeapSize);

        /// <summary>
        /// Set debug level of CUDA lib: debugLevel = 0 means has no debug level, max level = 10
        /// </summary>
        /// <param name="debugLevel"></param>
        [DllImport("CudaUnity")]
        public static extern void SetDebugLevel(int debugLevel);

        /// <summary>
        /// Cannot bigger than 1024, default Max Block Size = 256
        /// </summary>
        /// <param name="input"></param>
        [DllImport("CudaUnity")]
        public static extern void SetMaxBlockSize(int input);

        /// <summary>
        /// Whether or not to display each CUDA error in a separate Windows MessageBox, default = false
        /// </summary>
        /// <param name="display"></param>
        [DllImport("CudaUnity")]
        public static extern void DisplayErrorInMessageBox(bool display);

        /// <summary>
        /// Whether or not to display all catched errors in an extra Windows console, default = true
        /// </summary>
        /// <param name="display"></param>
        [DllImport("CudaUnity")]
        public static extern void DisplayErrorInExtraConsole(bool display);

        /// <summary>
        /// Set debug call back function in CUDA lib
        /// </summary>
        /// <param name="callback"></param>
        [DllImport("CudaUnity")]
        public static extern void SetCallback(DebugLogCallback callback);

        /// <summary>
        ///  delegate of debug call back 
        /// </summary>
        /// <param name="debugLog"></param>
        public delegate void DebugLogCallback(IntPtr debugLog);

        /// <summary>
        /// debug call back from CUDA lib
        /// </summary>
        /// <param name="debugLog"></param>
        public static void DebugLogCallbackFuction(IntPtr debugLog)
        {
            Debug.Log("DebugLog from CPP: " + Marshal.PtrToStringAuto(debugLog));
        }

        static DebugLogCallback callback;

        public static void InitializedCudaDebugCallBack()
        {
            callback = DebugLogCallbackFuction;
            SetCallback(callback);
        }
        //public static void ExportSDF(Texture3D exportSDF, float[] sdfData, int voxelSize,string name)
        //{
        //    exportSDF.SetPixelData(sdfData, 0);
        //    AssetDatabase.CreateAsset(exportSDF, "Assets/Large Volume Data Interaction Framework/Textures/SDF_" + voxelSize + "_" + name + ".asset");
        //}

    }
}
