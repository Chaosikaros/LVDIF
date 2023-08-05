#include <string> 
#include <cstdarg>
#include <vector>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_timer.h"
#include <comutil.h>
#include <windows.h>
#include <time.h>
#include <tchar.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tchar.h>
#include <process.h>
#include <iostream>
#include <Windows.h>
#include "dbghelp.h"
#include <exception>
#include <stdexcept>
#include <thread>
#include <random>
#include <stack> 
#include <functional>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#pragma comment(lib,"Dbghelp.lib")
#pragma comment(lib, "comsuppw.lib")

#ifndef _CudaUnity_H_

#define _CudaUnity_H_
#define DLL_EXPORTS
#define getLastCudaError(msg) { CudaUnity::__getLastCudaError(msg, __FILE__, __LINE__);}
#define checkCudaErrors(ans) { CudaUnity::cudaAssert((ans), __FILE__, __LINE__); }

#define SKIP_EMPTY_VOXELS 0

#define NumberOfChunkPointer 1024
#define NumberOfCudaStreams 512
#define NumberOfMaxTriList 128 * 128 * 128
#define MaxArraySize 1024 * 1024 * 1024
#define MinVoxel -0.0000001f
#define ClampBound 4
#define ClampSize 2 * ClampBound * 1000 + 2
#define DictionarySize 65536

typedef unsigned __int64 size_t;

typedef unsigned int uint;

typedef unsigned char uchar;

__host__ __device__ struct Vector4
{
	float x, y, z, w;
};

__host__ __device__ struct Vector3
{
	float x, y, z;
};

__host__ __device__ struct BrushVoxel {
	uint brushID;
	uint lastBrushID;
	ushort backgroundColor;
	ushort lastBackgroundColor;
};

__host__ __device__ struct BrushInfo {
	float4 shape;
	ushort colorID;
	ushort type;
	uint brushID;
	BrushInfo(float4 shapeIn, ushort typeIn, ushort colorIn, uint brushIDIn)
	{
		shape = shapeIn;
		type = typeIn;
		colorID = colorIn;
		brushID = brushIDIn;
	}
};

__host__ __device__ struct Triangle {
	Vector3 posA, posB, posC;
	Vector3 normalA, normalB, normalC;
	int Color;
	uint Colors[15];
	uint3 cubeInfo;
	//uint Colors[8];
	//uint4 ColorA;
	//uint4 ColorB;
};

__host__ __device__ struct OctreeBbox {
	int chunkID;
	int octreeLeafID;
	int octreeLeafSize;
	int maxOctreeLeafSize;
	int octreeDepth;
	int octreeNodeID;
	//OctreeBbox* parentNode;
	//OctreeBbox* leafNodes;
	//int parentNodeID;
	int leafNodesID[8];
	int boxCount;
	bool empty;
	bool init;
	uint3 chunkPos;
	float3 bboxMin;
	float3 bboxMax;
	float3 bboxCenter;
	float3 bboxHalfsize;
	float3 bboxChunkSize;
	int triangleCount;
	int* triangleList;
	OctreeBbox()
	{
		for (int i = 0; i < 8; i++)
			leafNodesID[i] = -1;
		empty = true;
		init = false;
		chunkID = 0;
		//leafNodes = NULL;
	}
};

__host__ __device__ struct Bbox {
	float3 center;
	float3 bboxMin;
	float3 bboxMax;
};

__host__ __device__ struct McChunk {
	bool needUpdate;
	int chunkID;
	int voxelNumber;
	int tribufferStart = 0;
	int tribufferEnd = 0;
	BrushInfo* brushArray;
	BrushInfo* brushArrayTemp;
	float3 bboxMin;
	float3 bboxMax;
	uint3 chunkPos;
	uint3 chunkSize;
	Triangle* d_Tribuffer;
	int* d_triNum;
	int* h_triNum;
	uint activeVoxels;
	int* d_voxelVerts = NULL;
	int* d_voxelVertsScan = NULL;
	int* d_voxelOccupied = NULL;
	int* d_voxelOccupiedScan = NULL;
	int* d_compVoxelArray = NULL;
};

inline __host__ __device__ float invLerp(float from, float to, float value)
{
	return (value - from) / (to - from);
}
inline __host__ __device__ float signSDF(float x)
{
	if (x > 0) return 1;
	else return -1;
}

inline __host__ __device__ float sign(float x)
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}
inline __host__ __device__ float dot2(float3 a)
{
	return dot(a, a);
}

inline __host__ __device__ float remap(float origFrom, float origTo, float targetFrom, float targetTo, float value)
{
	float rel = invLerp(origFrom, origTo, value);
	return lerp(targetFrom, targetTo, rel);
}

inline __host__ __device__ Vector4 make_Vector4(float4 f)
{
	Vector4 t; t.x = f.x; t.y = f.y; t.z = f.z; t.w = f.w; return t;
};
inline __host__ __device__ float4 make_float4(Vector4 f)
{
	float4 t; t.x = f.x; t.y = f.y; t.z = f.z; t.w = f.w; return t;
};
inline __host__ __device__ Vector3 make_Vector3(float3 f)
{
	Vector3 t; t.x = f.x; t.y = f.y; t.z = f.z; return t;
};

inline __host__ __device__ float3 make_float3(Vector3 f)
{
	float3 t; t.x = f.x; t.y = f.y; t.z = f.z; return t;
};

inline __host__ __device__ float3 normalizeCheck(float3 v)
{
	float3 t = normalize(v);
	if (isnan(t.x))
		t.x = 0;
	if (isnan(t.y))
		t.y = 0;
	if (isnan(t.z))
		t.z = 0;
	return t;
};


__host__ __device__ struct TriVertice
{
	int triangleID;
	float3 vertice;

	TriVertice(int k, Vector3 v) : triangleID(k), vertice(make_float3(v.x, v.y, v.z)) {}
};

__host__ __device__ struct TriMesh {
	float3 posA, posB, posC;
	Vector3 normalA, normalB, normalC;
};


inline __host__ __device__ float GetSdfVectorValue(float inputSdf)
{
	int precisionClose = 3;
	int precisionFar = 1;
	float dist = inputSdf;
	int d = 1;
	if (fabs(inputSdf) < ClampBound)
		d = powf(10, precisionClose);
	else
		d = powf(10, precisionFar);
	dist = round(dist * d) / d;
	return dist;
}

inline __host__ __device__ ushort GetSdfVectorKey(float inputSdf)
{
	float inputSdfAbs = fabs(inputSdf);
	int mapSdfDictionaryKey = 0;
	if (inputSdf >= MinVoxel && inputSdf <= 0)
	{
		mapSdfDictionaryKey = 0;
	}
	else if (inputSdfAbs <= ClampBound + 0.002)
	{
		mapSdfDictionaryKey = (int)(2 * inputSdfAbs * 1000);
		if ((inputSdf < 0 && mapSdfDictionaryKey % 2 == 0) || (inputSdf > 0 && mapSdfDictionaryKey % 2 == 1))
			mapSdfDictionaryKey++;
	}
	else
	{
		mapSdfDictionaryKey = (int)(2 * (inputSdfAbs - ClampBound) * 10 + ClampSize);
		if ((inputSdf < 0 && mapSdfDictionaryKey % 2 == 0) || (inputSdf > 0 && mapSdfDictionaryKey % 2 == 1))
			mapSdfDictionaryKey++;
	}
	mapSdfDictionaryKey = clamp(mapSdfDictionaryKey, 0, DictionarySize - 1);
	return (ushort)mapSdfDictionaryKey;
}

#ifdef DLL_EXPORTS
#define EXPORTS_DLL _declspec(dllexport)
#endif

typedef void(__stdcall* CPPCallback)(BSTR output);
typedef void(__stdcall* McCallback)(int chunkID);
typedef void(__stdcall* SdfCallback)(float* sdf, int size);
typedef void(__stdcall* MeshChunkCallback)(int chunkID, bool state, int triCountInput);

extern"C" {
	EXPORTS_DLL void DisplayErrorInMessageBox(bool input);
	EXPORTS_DLL void DisplayErrorInExtraConsole(bool input);
	EXPORTS_DLL void SetCallback(CPPCallback callback);
	EXPORTS_DLL void GetDeviceInfo();
	EXPORTS_DLL int GetDeviceNumber();
	EXPORTS_DLL void SetDevice(int deviceID);
	EXPORTS_DLL void SetMaxBlockSize(int input);
	EXPORTS_DLL void SetDebugLevel(int input);
	EXPORTS_DLL int GetMallocHeapSize();
	EXPORTS_DLL int SetMallocHeapSize(int MallocHeapSize);

	EXPORTS_DLL void SaveSdfData(int chunkID, char* fileName);
	EXPORTS_DLL void GetSdfArray(int chunkID, int size, float* sdfData);
	EXPORTS_DLL void GetColorArray(int chunkID, int size, int* colorData);
	EXPORTS_DLL bool SetInputBrushForVoxel(int chunkID, int inputRadius, int brushSize, Vector4* inputPos, int colorType, 
		float colorBrushOffset, bool eraserMode, int brushShape);
	EXPORTS_DLL bool SetMeshForVoxel(Vector3 gridSize, Vector3 boundsMax, Vector3 boundsMin, Vector3* vertices,  int* triangles,
		Vector3* normals, int verticeSize, int triangleSize, int normalSize);
	EXPORTS_DLL bool SetMarchingCubesChunks(int num, int minChunkSizeLog2, int chunkThreads);
	EXPORTS_DLL void SetMarchingCubesKernelInThread(int chunkID, int size, int voxelThreads, int triangleThreads, float3 CenterPos, float GridW, float octreeBboxSizeOffset, 
		float IsoLevel, bool EnableSmooth, bool loadMesh, bool loadSdf, bool loadSdfFromUnity, ushort2* sdfData,
		int gridSizeLog2OBox, bool exportTexture3D, char* sdfFileName, float filterValue, int sleepTime, int maxUpatedChunk, int SVFSetting);
	EXPORTS_DLL int GetExtractMarchingTriCount(int chunkID);
	EXPORTS_DLL bool GetExtractCubeVoxels(int chunkID, int size, float* cubeVoxel);
	EXPORTS_DLL void SetCallbackForMeshChunk(MeshChunkCallback callback);
	EXPORTS_DLL void SetCallbackForMC(McCallback callback);
	EXPORTS_DLL void SetCallbackForSdfExport(SdfCallback callback);
	EXPORTS_DLL void GetExtractMarchingCubesData(int chunkID, int size, Vector3 *vertices, Vector3 *normals, int *triangles);
	EXPORTS_DLL void GetExtractMarchingCubesChunkData(int chunkID, int childChunkID, int size, Vector3* vertices, Vector3* normals, int* triangles, int* colors,
		float GridWI, int triangleThreads, float IsoLevel, bool EnableSmooth);
	EXPORTS_DLL bool MallocMemoryForMC(int chunkID, Vector3 size, float GridW);
	EXPORTS_DLL bool FreeMemoryForMC(int chunkID);
};

namespace CudaUnity
{
	EXPORTS_DLL void DebugLogToUnity(std::string output);
	EXPORTS_DLL void SetMaxBlockSize(int input);

	EXPORTS_DLL std::string sFormator(const std::string sFormat, ...);
	EXPORTS_DLL std::string GetStringOfLastError(cudaError_t err);
	EXPORTS_DLL inline void cudaAssert(cudaError_t code, const char *file, int line);
	EXPORTS_DLL inline void __getLastCudaError(const char *errorMessage, const char *file, const int line);

	EXPORTS_DLL void launch_sampleOctreeBbox(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax,
		int n_triangles, float* triangle_data, float* normal_data, OctreeBbox* octreeBboxes, float3 bboxChunkSize, float octreeBboxSizeOffset, int* triNum, bool calculateTri);
	EXPORTS_DLL void launch_sampleTriangle(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax, 
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize);
	EXPORTS_DLL void launch_sampleTriangleThread(dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize);
	EXPORTS_DLL void launch_sampleTriangleWindingNumber(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize);
	EXPORTS_DLL void launch_sampleTriangleThreadWindingNumber(dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize);
	EXPORTS_DLL void launch_sampleTriangleFilter(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize, float filterValue, int* fliterCounter, int currentLayer, int filterMode);
	EXPORTS_DLL void launch_insertShapeToVoxel(cudaStream_t cudaStreams, dim3 grid, dim3 threads, float2* sdfDictionary, ushort2* noiseData, uint3 gridSize, uint3 gridSizeShift,
		uint3 gridSizeMask, int inputRadius, BrushInfo* inputPos, BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing, int inputPosCount, int brushColorType, float colorBrushOffset, bool eraserMode
		, int brushShape, uint3 realGridSize, uint3 chunkPos, bool useChunk);
	EXPORTS_DLL void launch_classifyVoxel(int* triNum, cudaStream_t cudaStreams, float2* sdfDictionary, ushort2* noiseData, dim3 grid, dim3 threads,
		int *voxelVerts, int *voxelOccupied, uint3 gridSize, uint3 gridSizeShift, 
		uint3 gridSizeMask, uint numVoxels, float3 voxelSize, float isoValue, 
		uint3 realGridSize, uint3 chunkPos, bool useChunk);
	//EXPORTS_DLL void launch_countTriangles(cudaStream_t cudaStreams, dim3 grid, dim3 threads, int* triNum, float* noiseData, int *compactedVoxelArray, int *numVertsScanned,
	//	uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float isoValue, uint activeVoxels);
	EXPORTS_DLL void ThrustScanWrapper(int *output, int *input, int numElements);
	EXPORTS_DLL void launch_compactVoxels(cudaStream_t cudaStreams, dim3 grid, dim3 threads, int *compactedVoxelArray, int *voxelOccupied,
		int *voxelOccupiedScan, uint numVoxels);
	EXPORTS_DLL void launch_generateTriangles(cudaStream_t cudaStreams, int maxTri, int* triNum, float2* sdfDictionary, ushort2* noiseData, float3 CenterPos, int GridRes, int GridW, dim3 grid, int threads,
		int *compactedVoxelArray, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, Triangle* Tribuffer, bool EnableSmooth, 
		uint3 realGridSize, uint3 chunkPos, bool useChunk, BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing, int brushListCount);
};

class MyException {
public:
	CONTEXT Context;

	MyException() {
		RtlCaptureContext(&Context);
	}
};

extern HANDLE hConsole;
extern HANDLE hConIn;
extern HANDLE hConOut;
extern CPPCallback DebugLogCallBack;
extern bool msgBox;
extern bool msgConsole;
extern int debugLevel;
extern int cudaDeviceID;

#endif