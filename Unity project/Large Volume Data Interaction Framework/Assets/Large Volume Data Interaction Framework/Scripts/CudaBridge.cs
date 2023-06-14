using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ChaosIkaros.LVDIF
{
    public class CudaBridge
    {

        [DllImport("CudaUnity")]
        public static extern bool SetInputForVoxel(int chunkID, int inputRadius, Vector3 inputPos, bool eraserMode);
        [DllImport("CudaUnity")]
        public static extern bool SetInputBrushForVoxel(int chunkID, int inputRadius, int brushSize, Vector4[] inputPos,
            int colorType, float colorBrushOffset, bool eraserMode, int brushShape);

        [DllImport("CudaUnity")]
        public static extern bool SetMeshForVoxel(Vector3 gridSize, Vector3 boundsMax, Vector3 boundsMin, Vector3[] vertices, int[] triangles,
            Vector3[] normals, int verticeSize, int triangleSize, int normalSize);
        [DllImport("CudaUnity")]
        public static extern bool SetMarchingCubesChunks(int num, int minChunkSizeLog2I, int chunkThreadsI);
        [DllImport("CudaUnity")]
        public static extern void SetMarchingCubesKernelInThread(int chunkID, int voxelThreads, int triangleThreads, Vector3 CenterPos, float volumeWidth
            , float NoiseInterval, float IsoLevel, bool EnableSmooth, bool loadMesh, bool loadSDF, float[] sdfData,
            int gridSizeLog2OBox, bool exportTexture3D, string sdfFileName, float filterValue, int sleepTime, int maxUpatedChunk, int SVFSetting);
        [DllImport("CudaUnity")]
        public static extern void SaveSdfData(int chunkID, string fileName);
        [DllImport("CudaUnity")]
        public static extern bool GetExtractCubeVoxels(int chunkID, int size, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] float[] cubeVoxel);
        [DllImport("CudaUnity")]
        public static extern int GetExtractMarchingTriCount(int chunkID);
        [DllImport("CudaUnity")]
        public static extern void GetExtractMarchingCubesData(int chunkID, int size, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] Vector3[] vertices,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] Vector3[] normals, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] int[] triangles);

        [DllImport("CudaUnity")]
        public static unsafe extern void GetExtractMarchingCubesData(int chunkID, int size, void* vertices, void* normals, void* triangles);

        [DllImport("CudaUnity")]
        public static unsafe extern void GetExtractMarchingCubesChunkData(int chunkID, int childChunkID, int size, void* vertices, void* normals
            , void* triangles, void* colors, float volumeWidth, int triangleThreads, float IsoLevel, bool EnableSmooth);
        [DllImport("CudaUnity")]
        public static extern void GetExtractMarchingCubesChunkData(int chunkID, int childChunkID, int size,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] Vector3[] vertices,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] Vector3[] normals,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] int[] triangles
            , float volumeWidth, int triangleThreads, float IsoLevel, bool EnableSmooth);
        [DllImport("CudaUnity")]
        public static extern bool MallocMemoryForMC(int chunkID, Vector3 size, float volumeWidth);
        [DllImport("CudaUnity")]
        public static extern bool FreeMemoryForMC(int chunkID);
    }
}
