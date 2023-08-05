#include "CudaUnity.h"
#include <conio.h>

using namespace CudaUnity;
using namespace std;
namespace CudaUnity {
	class MarchingCubesChunk {

	public:
		cudaEvent_t *stop_event;
		cudaEvent_t* kernelEvent;
		int eventflags = cudaEventBlockingSync;
		cudaStream_t *cudaStreams;
		McCallback mcCallBack = NULL;
		SdfCallback sdfCallBack = NULL;
		MeshChunkCallback meshChunkCallBack = NULL;
		std::thread kernelThread;
		StopWatchInterface *timer = NULL;
		uint3 gridSizeLog2 = make_uint3(5, 5, 5);
		uint3 gridSizeShift;
		uint3 gridSize;
		uint3 gridSizeVoxel;
		uint3 gridSizeMask;
		uint3 gridSizeLog2Chunk;
		uint3 gridSizeChunk;
		uint3 gridSizeMaskChunk;
		uint3 gridSizeShiftChunk;
		uint3 chunkSize;
		Bbox parentBbox;
		Bbox inputBbox;
		McChunk* mcChunks = NULL;
		int chunkNumber;
		int voxelNumInChunk;
		int** h_triNumAll;
		Triangle** h_TribufferAll;
		int* d_voxelVertsAll;
		int* d_voxelVertsScanAll;
		int* d_voxelOccupiedAll;
		int* d_voxelOccupiedScanAll;
		int** h_compVoxelArrayAll;
		OctreeBbox* octreeBboxChunks;
		OctreeBbox* h_octreeBboxNodes;
		int h_octreeBboxNodesCounter = 0;
		float3 voxelSize;
		uint numVoxels = 0;

		uint maxVerts = 0;
		//uint activeVoxels = 0;
		uint totalVerts = 0;
		
		float3 CenterPos;
		int triangleCounter = 0;
		//float *meshSampleData1D = NULL;
		//float* meshSampleData1DTemp = NULL;
		//float* meshSampleData1DTempDevice = NULL;
		float2* sdfDictionary1D = NULL;
		float2* sdfDictionary1DTemp = NULL;
		ushort2* noiseData1D = NULL;
		ushort2* noiseData1DTemp = NULL;

		//float *noiseData1DTempDevice = NULL;
		int *d_triNum;
		int *h_triNum;
		int *z_triNum;
		int lastTriNum;
		Triangle* NullTribuffer = NULL;
		Triangle* d_NullTribuffer = NULL;
		Triangle *Tribuffer = NULL;
		Triangle *d_Tribuffer = NULL;

		int* octreeScan = NULL;
		int* voxelScan = NULL;
		int* voxelScanTemp = NULL;
		int* compactVoxel = NULL;
		int* compactVoxelTemp = NULL;
		float* triangle_dataTemp = NULL;
		float* normal_dataTemp = NULL;
		float3 unit;
		float* meshVoxel1DOut = NULL;
		float3 bboxMax = make_float3(0);
		float3 bboxMin = make_float3(0);
		int meshTriangleSize;
		float* triangle_data = NULL;
		float* normal_data = NULL;
		bool currentEaserMode = false;
		int currentInputRadius = 0;
		BrushInfo* currentInputPosInBrush;

		BrushInfo* brushList;
		vector<BrushInfo> brushListTemp;
		BrushVoxel* brushVoxels;
		BrushVoxel* brushVoxelsTemp;
		bool enableColorSmoothing = false;

		int brushInputSize = 0;
		int brushColorType = 0;
		int brushShapeType = 0;
		float colorBrushOffset = 0;
		bool hasInput = false;
		bool hasBoolOp = false;
		bool canFree = false;
		int chunkID = 0;
		int triCount = 0;
		int chunkThreads;
		int minChunkSizeLog2;

		MarchingCubesChunk(int ID) { chunkID = ID; }
		MarchingCubesChunk(){}
		MarchingCubesChunk(MarchingCubesChunk &&) {}
		MarchingCubesChunk(const MarchingCubesChunk&) = default;
		MarchingCubesChunk& operator=(const MarchingCubesChunk&) = default;
		void InitMC(int ID, McCallback mcCallBackI, SdfCallback sdfCallBackI, MeshChunkCallback meshChunkCallBackI, int minChunkSizeLog2I, int chunkThreadsI)
		{
			minChunkSizeLog2 = minChunkSizeLog2I;
			chunkThreads = min(chunkThreadsI, NumberOfCudaStreams);
			chunkID = ID;
			mcCallBack = mcCallBackI;
			sdfCallBack = sdfCallBackI;
			meshChunkCallBack = meshChunkCallBackI;
		}
		template <class T>
		void dumpBuffer(T *d_buffer, int nelements, int size_element, const char *name)
		{
			uint bytes = nelements * size_element;
			T *h_buffer = (T *)malloc(bytes);
			checkCudaErrors(cudaMemcpyAsync(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			for (int i = 0; i < nelements; i++)
				DebugLogToUnity(sFormator("%s [%d]: %d\n", name, i, h_buffer[i]));

			free(h_buffer);
		}

		void SetTimer()
		{
			if (debugLevel >= 1)
			{
				sdkStartTimer(&timer);
				sdkResetTimer(&timer);
			}
		}

		void SaveSdfData(int chunkID, char* fileName)
		{
			int n = gridSize.x * gridSize.y * gridSize.z;
			string fileNameS = string(fileName);
			ushort2* noiseData1DOut = (ushort2*)malloc(n * sizeof(ushort2));
			checkCudaErrors(cudaMemcpyAsync(noiseData1DOut, noiseData1DTemp, n * sizeof(ushort2), cudaMemcpyDeviceToHost, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			ofstream sOut(fileNameS, ios::out | ios::binary);
			sOut.write(reinterpret_cast<char*>(noiseData1DOut), sizeof(ushort2) * n);
			sOut.close();
			free(noiseData1DOut);
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Save SDF for chunk [%d] : %s", chunkID, fileNameS.c_str()));
		}

		bool SetInputBrushForVoxel(int chunkID, int inputRadius, int brushSize, Vector4* inputPos,
			int colorType, float colorOffset, bool eraserMode, int brushShape)
		{
			hasInput = true;
			currentInputRadius = inputRadius;
			currentInputPosInBrush = (BrushInfo*)malloc(brushSize * sizeof(BrushInfo));
			for (int i = 0; i < brushSize; i++)
			{
				currentInputPosInBrush[i] = BrushInfo(make_float4(inputPos[i]), brushShape, colorType, 0);
				if (enableColorSmoothing)
				{
					currentInputPosInBrush[i].brushID = brushListTemp.size();
					//DebugLogToUnity(sFormator("BrushID [%d] : %d", chunkID, currentInputPosInBrush[i].brushID));
					brushListTemp.push_back(currentInputPosInBrush[i]);
				}
			}
			if (enableColorSmoothing)
			{
				cudaFree(brushList);
				checkCudaErrors(cudaMalloc((void**)&brushList, brushListTemp.size() * sizeof(BrushInfo)));
				BrushInfo* brushListTempArray = (BrushInfo*)malloc(brushListTemp.size() * sizeof(BrushInfo));;
				std::copy(brushListTemp.begin(), brushListTemp.end(), brushListTempArray);
				checkCudaErrors(cudaMemcpyAsync(brushList, brushListTempArray, brushListTemp.size() * sizeof(BrushInfo), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			}
			brushColorType = colorType;
			brushInputSize = brushSize;
			colorBrushOffset = colorOffset;
			currentEaserMode = eraserMode;
			brushShapeType = brushShape;
			return true;
		}

		bool SetMeshForVoxel(Vector3 gridSize, Vector3 boundsMax, Vector3 boundsMin, Vector3* vertices, int* triangles, Vector3* normals, int verticeSize, int triangleSize, int normalSize)
		{
			if (debugLevel >= 2)
			{
				for (int i = 0; i < triangleSize; i++)
					DebugLogToUnity(sFormator("triangles[%d]: %f %f %f", triangles[i],
						vertices[triangles[i]].x, vertices[triangles[i]].y, vertices[triangles[i]].z));
			}

			meshTriangleSize = triangleSize * 3;//输入的是vertice triangle的id 所以乘以3就行了
			triangle_dataTemp = (float *)malloc(meshTriangleSize * sizeof(float));
			normal_dataTemp = (float*)malloc(meshTriangleSize * sizeof(float));
			checkCudaErrors(cudaFree(triangle_data));
			checkCudaErrors(cudaMalloc((void**)&triangle_data, meshTriangleSize * sizeof(float)));
			checkCudaErrors(cudaFree(normal_data));
			checkCudaErrors(cudaMalloc((void**)&normal_data, meshTriangleSize * sizeof(float)));
			int j = 0;
			for (int i = 0; i < triangleSize; i++)
			{
				triangle_dataTemp[j] = vertices[triangles[i]].x;
				triangle_dataTemp[j + 1] = vertices[triangles[i]].y;
				triangle_dataTemp[j + 2] = vertices[triangles[i]].z;
				j += 3;
			}
			j = 0;
			for (int i = 0; i < triangleSize; i++)
			{
				normal_dataTemp[j] = normals[triangles[i]].x;
				normal_dataTemp[j + 1] = normals[triangles[i]].y;
				normal_dataTemp[j + 2] = normals[triangles[i]].z;
				j += 3;
			}
			checkCudaErrors(cudaMemcpyAsync(triangle_data, triangle_dataTemp, meshTriangleSize * sizeof(float), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			checkCudaErrors(cudaMemcpyAsync(normal_data, normal_dataTemp, meshTriangleSize * sizeof(float), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			bboxMax = make_float3(boundsMax.x, boundsMax.y, boundsMax.z);
			bboxMin = make_float3(boundsMin.x, boundsMin.y, boundsMin.z);

			float m_padding = 0.2f;
			bboxMin -= make_float3(m_padding);
			bboxMax += make_float3(m_padding);
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Bounding box cube for SDF max: %f %f %f, min: %f %f %f", bboxMax.x, bboxMax.y, bboxMax.z,
					bboxMin.x, bboxMin.y, bboxMin.z));
			return true;
		}

		int GridPosToInt(uint3 gridSize, uint3 input)
		{
			return (input.x * gridSize.y + input.y)* gridSize.z + input.z;
		}

		int RunCudaStreamPerThread(dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
			int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize)
		{
			cudaSetDevice(cudaDeviceID);
			launch_sampleTriangleThread(grid, threads, gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax,
				n_triangles, triangle_data, normal_data,sdfDictionary, noiseData, realGridSize, chunkPos, useChunk, octreeBboxes, octreeSize);
			cudaStreamSynchronize(0);
			return 0;
		}

		bool IsOverlapping1D(float xmax1, float xmin1, float xmax2, float xmin2)
		{
			return (xmax1 >= xmin2) && (xmax2 >= xmin1);
		}

		bool IsBBoxOverlap(float3 bboxMax, float3 bboxMin, float3 sBboxMax, float3 sBboxMin)
		{
			return (IsOverlapping1D(bboxMax.x, bboxMin.x, sBboxMax.x, sBboxMin.x)
				&& IsOverlapping1D(bboxMax.y, bboxMin.y, sBboxMax.y, sBboxMin.y)
				&& IsOverlapping1D(bboxMax.z, bboxMin.z, sBboxMax.z, sBboxMin.z));
		}

		void ScanOctreeBbox(OctreeBbox octreeBboxNode, int octreeDepth, int maxOctreeLeafSize, float3 bboxMax, float3 bboxMin, int brushID)
		{
			if (pow(2, octreeDepth) > maxOctreeLeafSize || !IsBBoxOverlap(octreeBboxNode.bboxMax, octreeBboxNode.bboxMin, bboxMax, bboxMin))
				return;
			//|| !IsBBoxOverlap(octreeBboxNode.bboxMax, octreeBboxNode.bboxMin, bboxMax, bboxMin)
			if (debugLevel >= 2)
				DebugLogToUnity(sFormator("Found OctreeNode in Leaf[%d], Depth[%d], MaxLeaf[%d]", octreeBboxNode.octreeLeafSize, octreeDepth, maxOctreeLeafSize));
			if (pow(2, octreeDepth + 1) > maxOctreeLeafSize)
			{
				for (int i = 0; i < 8; i++)
				{
					if (IsBBoxOverlap(h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].bboxMax, h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].bboxMin, bboxMax, bboxMin))
					{
						mcChunks[h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].chunkID].needUpdate = true;
						mcChunks[h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].chunkID].brushArrayTemp[brushID].shape.w = currentInputPosInBrush[brushID].shape.w;
						if(debugLevel >= 2)
							DebugLogToUnity(sFormator("Found OctreeNode [%d]", h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].chunkID));
					}
				}
			}
			else
			{
				if (!octreeBboxNode.empty)
					for (int i = 0; i < 8; i++)
						if (octreeBboxNode.leafNodesID[i] >= 0)
							ScanOctreeBbox(h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]],
								h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].octreeDepth, maxOctreeLeafSize, bboxMax, bboxMin, brushID);
						else if (debugLevel >= 2)
							DebugLogToUnity(sFormator("Found Null OctreeNode in Leaf[%d], Depth[%d], MaxLeaf[%d], leafNodesID[%d]", 
								octreeBboxNode.octreeLeafSize, octreeDepth, maxOctreeLeafSize, octreeBboxNode.leafNodesID[i]));
			}
		}

		void GetSDF(int size, float* sdf)
		{
			int n = gridSize.x * gridSize.y * gridSize.z;
			ushort2* ushort2Out = (ushort2*)malloc(n * sizeof(ushort2));
			checkCudaErrors(cudaMemcpyAsync(ushort2Out, noiseData1DTemp, n * sizeof(ushort2), cudaMemcpyDeviceToHost, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));

			for (int i = 0; i < n; i++)
				sdf[i] = sdfDictionary1D[ushort2Out[i].x].x;
			free(ushort2Out);
		}

		void GetVolumeColor(int size, int* colors)
		{
			int n = gridSize.x * gridSize.y * gridSize.z;
			ushort2* ushort2Out = (ushort2*)malloc(n * sizeof(ushort2));
			checkCudaErrors(cudaMemcpyAsync(ushort2Out, noiseData1DTemp, n * sizeof(ushort2), cudaMemcpyDeviceToHost, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));

			for (int i = 0; i < n; i++)
				colors[i] = (int)ushort2Out[i].y;
			free(ushort2Out);
		}

		int SetMarchingCubesKernel(int size, int voxelThreads, int triangleThreads, float3 CenterPosI, float GridWI,
			float octreeBboxSizeOffset, float IsoLevel, bool EnableSmooth, bool loadMesh, bool loadSdf, bool loadSdfFromUnity, ushort2* sdfData,
			int gridSizeLog2OBox, bool exportTexture3D, string sdfFileName, float filterValue, int sleepTime, int maxUpatedChunk, int SVFSetting)
		{
			cudaSetDevice(cudaDeviceID);
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d : start thread function: %f", chunkID, sdkGetTimerValue(&timer)));
			SetTimer();

			CenterPos = CenterPosI;
			//int threads = 128;
			dim3 grid(numVoxels / voxelThreads, 1, 1);
			// get around maximum grid size of 65535 in each dimension
			if (grid.x > 65535)
			{
				grid.y = grid.x / 32768;
				grid.x = 32768;
			}
			int n = gridSize.x * gridSize.y * gridSize.z;

			dim3 gridChunk(voxelNumInChunk / voxelThreads, 1, 1);
			// get around maximum grid size of 65535 in each dimension
			if (gridChunk.x > 65535)
			{
				gridChunk.y = gridChunk.x / 32768;
				gridChunk.x = 32768;
			}
			if (loadSdf)
			{
				if (loadSdfFromUnity)
				{
					noiseData1D = (ushort2*)malloc(n * sizeof(ushort2));
					for(int i=0;i<n;i++)
						noiseData1D[i] = make_ushort2(sdfData[i].x, sdfData[i].y);
				}
				else
				{
					noiseData1D = (ushort2*)malloc(n * sizeof(ushort2));
					ifstream sIn(sdfFileName, ios::in | ios::binary);
					sIn.read(reinterpret_cast<char*>(noiseData1D), sizeof(ushort2) * n);
					sIn.close();
				}
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Load SDF for chunk [%d] : %s", chunkID, sdfFileName.c_str()));
				if (debugLevel >= 3)
				{
					int n = gridSize.x * gridSize.y * gridSize.z;
					for (int i = 0; i < n; i++)
						DebugLogToUnity(sFormator("SDF at voxel [%d] : %f", i, sdfDictionary1D[noiseData1D[i].x].x));
				}

				//else
				//{
				//	for (int i = 0; i < n; i++)
				//		noiseData1D[i].x = sdfData[i];
				//}
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :Load SDF to CPU: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
				checkCudaErrors(cudaMemcpyAsync(noiseData1DTemp, noiseData1D, n * sizeof(ushort2), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));

				free(noiseData1D);
				//checkCudaErrors(cudaMemcpyAsync(noiseData1DTempDevice, noiseData1DTemp, n * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreams[0]));
				//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				if (exportTexture3D)
				{
					int n = gridSize.x * gridSize.y * gridSize.z;
					ushort2* ushort2Out = (ushort2*)malloc(n * sizeof(ushort2));
					checkCudaErrors(cudaMemcpyAsync(ushort2Out, noiseData1DTemp, n * sizeof(ushort2), cudaMemcpyDeviceToHost, cudaStreams[0]));
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));

					float* noiseData1DOut = (float*)malloc(n * sizeof(float));
					for (int i = 0; i < n; i++)
					{
						noiseData1DOut[i] = sdfDictionary1D[ushort2Out[i].x].x;
					}
					free(ushort2Out);
					sdfCallBack(noiseData1DOut, n);
					free(noiseData1DOut);
				}
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :Load SDF to GPU: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
			}
			else if (loadMesh)
			{
				//checkCudaErrors(cudaMemcpyAsync(noiseData1DTemp, noiseData1DTempDevice, n * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreams[0]));
				//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//checkCudaErrors(cudaEventSynchronize(stop_event[0]));

				uint3 adaptiveMapMax = gridSize - make_uint3(2);
				uint3 adaptiveMapMin = make_uint3(2);

				float bboxXRange = bboxMax.x - bboxMin.x;
				float bboxYRange = bboxMax.y - bboxMin.y;
				float bboxZRange = bboxMax.z - bboxMin.z;
				float3 bboxRange = make_float3(bboxXRange, bboxYRange, bboxZRange);
				float maxGridRange = adaptiveMapMax.x;
				float maxBboxRange = max(max(bboxRange.x, bboxRange.y), bboxRange.z);
				float adaptGridRange = 0;
				float halfGridSize = 0;
				if (maxBboxRange == bboxXRange)
				{
					adaptGridRange = maxGridRange * (bboxRange.y / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.y = adaptGridRange + halfGridSize;
					adaptiveMapMin.y = halfGridSize;
					adaptGridRange = maxGridRange * (bboxRange.z / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.z = adaptGridRange + halfGridSize;
					adaptiveMapMin.z = halfGridSize;
				}
				else if (maxBboxRange == bboxYRange)
				{
					adaptGridRange = maxGridRange * (bboxRange.x / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.x = adaptGridRange + halfGridSize;
					adaptiveMapMin.x = halfGridSize;
					adaptGridRange = maxGridRange * (bboxRange.z / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.z = adaptGridRange + halfGridSize;
					adaptiveMapMin.z = halfGridSize;
				}
				else
				{
					adaptGridRange = maxGridRange * (bboxRange.x / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.x = adaptGridRange + halfGridSize;
					adaptiveMapMin.x = halfGridSize;
					adaptGridRange = maxGridRange * (bboxRange.y / maxBboxRange);
					halfGridSize = 0.5 * (maxGridRange - adaptGridRange);
					adaptiveMapMax.y = adaptGridRange + halfGridSize;
					adaptiveMapMin.y = halfGridSize;
				}
				int triangleNumber = meshTriangleSize/9;

				uint3 gridSizeLog2O = make_uint3(gridSizeLog2OBox);
				int octreeLeafSize = 1 << gridSizeLog2O.x;
				uint3 gridSizeO = make_uint3(1 << gridSizeLog2O.x, 1 << gridSizeLog2O.y, 1 << gridSizeLog2O.z);
				uint3 gridSizeMaskO = make_uint3(gridSizeO.x - 1, gridSizeO.y - 1, gridSizeO.z - 1);
				uint3 gridSizeShiftO = make_uint3(0, gridSizeLog2O.x, gridSizeLog2O.x + gridSizeLog2O.y);
				int nO = octreeLeafSize * octreeLeafSize * octreeLeafSize;
				dim3 gridO(nO / voxelThreads, 1, 1);
				// get around maximum grid size of 65535 in each dimension
				if (gridO.x > 65535)
				{
					gridO.y = gridO.x / 32768;
					gridO.x = 32768;
				}
				//float octreeBboxSizeOffset = 2;
				float3 chunkSize = make_float3(gridSizeO.x / octreeLeafSize);
				float3 bboxChunkSize = (bboxMax - bboxMin) / octreeLeafSize;
				OctreeBbox* octreeBboxList = (OctreeBbox*)malloc(nO *sizeof(OctreeBbox));
				int idBox = 0;
				for (int x = 0; x < octreeLeafSize; x++)
				{
					for (int y = 0; y < octreeLeafSize; y++)
					{
						for (int z = 0; z < octreeLeafSize; z++)
						{
							float3 bboxPos = make_float3(bboxChunkSize.x * x,
								bboxChunkSize.y * y, bboxChunkSize.z * z);
							octreeBboxList[idBox].octreeLeafID = idBox;
							octreeBboxList[idBox].boxCount = nO;
							octreeBboxList[idBox].empty = true;
							octreeBboxList[idBox].bboxMax = bboxMin + bboxPos + bboxChunkSize;
							octreeBboxList[idBox].bboxMin = bboxMin + bboxPos;
							octreeBboxList[idBox].bboxCenter = bboxMin + bboxPos + 0.5f * bboxChunkSize;
							octreeBboxList[idBox].bboxHalfsize = 0.5f * bboxChunkSize;
							octreeBboxList[idBox].triangleCount = 0;
							idBox++;
						}
					}
				}
				OctreeBbox* octreeBboxListDevice = NULL;
				checkCudaErrors(cudaMalloc((void**)&octreeBboxListDevice, nO *sizeof(OctreeBbox)));
				checkCudaErrors(cudaMemcpyAsync(octreeBboxListDevice, octreeBboxList, nO * sizeof(OctreeBbox), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				checkCudaErrors(cudaMemcpyAsync(d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				launch_sampleOctreeBbox(cudaStreams[0], gridO, voxelThreads, gridSizeO, gridSizeShiftO, gridSizeMaskO, bboxMin, bboxMax,
					meshTriangleSize, triangle_data, normal_data, octreeBboxListDevice, bboxChunkSize, octreeBboxSizeOffset, d_triNum, true);
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));

				//free(octreeBboxList);
				//octreeBboxList = (OctreeBbox*)malloc(nO * sizeof(OctreeBbox));
				checkCudaErrors(cudaMemcpyAsync(octreeBboxList, octreeBboxListDevice, nO * sizeof(OctreeBbox), cudaMemcpyDeviceToHost, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				//int* triangleListDevicePointers[NumberOfMaxTriList];
				int** triangleListDevicePointers = (int**)malloc(nO * sizeof(int*));
				int* triangleList = (int*)malloc(triangleNumber * sizeof(int));
				for (int j = 0; j < triangleNumber; j++)
					triangleList[j] = -1;
				for (int i = 0; i < nO; i++)
				{
					octreeBboxList[i].triangleCount += 1;
					//DebugLogToUnity(sFormator("OctreeBbox ID %d : triangle count: %d", i, octreeBboxList[i].triangleCount));
					checkCudaErrors(cudaMalloc((void**)&triangleListDevicePointers[i], octreeBboxList[i].triangleCount * sizeof(int)));
					checkCudaErrors(cudaMemcpyAsync(triangleListDevicePointers[i], triangleList, octreeBboxList[i].triangleCount * sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					octreeBboxList[i].triangleList = triangleListDevicePointers[i];
				}
				//checkCudaErrors(cudaFree(octreeBboxListDevice));
				//checkCudaErrors(cudaMalloc((void**)&octreeBboxListDevice, nO * sizeof(OctreeBbox)));
				checkCudaErrors(cudaMemcpyAsync(octreeBboxListDevice, octreeBboxList, nO * sizeof(OctreeBbox), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				checkCudaErrors(cudaMemcpyAsync(d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				launch_sampleOctreeBbox(cudaStreams[0], gridO, voxelThreads, gridSizeO, gridSizeShiftO, gridSizeMaskO, bboxMin, bboxMax,
					meshTriangleSize, triangle_data, normal_data, octreeBboxListDevice, bboxChunkSize, octreeBboxSizeOffset, d_triNum, false);
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));

				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :sample Octree Bbox: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
				uint3 gridSizeLog2Chunk = make_uint3(minChunkSizeLog2);
				uint3 gridSizeChunk = make_uint3(1 << gridSizeLog2Chunk.x, 1 << gridSizeLog2Chunk.y, 1 << gridSizeLog2Chunk.z);
				if (gridSize.x <= gridSizeChunk.x)
				{
					launch_sampleTriangle(cudaStreams[0], grid, voxelThreads, gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax,
						adaptiveMapMin, adaptiveMapMax, meshTriangleSize, triangle_data, normal_data, sdfDictionary1DTemp, noiseData1DTemp, gridSize, gridSize, false, octreeBboxListDevice, nO);
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle: %f", chunkID, sdkGetTimerValue(&timer)));
					SetTimer();
				}
				else
				{
					uint3 gridSizeMaskChunk = make_uint3(gridSizeChunk.x - 1, gridSizeChunk.y - 1, gridSizeChunk.z - 1);
					uint3 gridSizeShiftChunk = make_uint3(0, gridSizeLog2Chunk.x, gridSizeLog2Chunk.x + gridSizeLog2Chunk.y);
					uint numVoxelsChunk = gridSizeChunk.x * gridSizeChunk.y * gridSizeChunk.z;
					dim3 gridChunk(numVoxelsChunk / voxelThreads, 1, 1);
					if (gridChunk.x > 65535)
					{
						gridChunk.y = gridChunk.x / 32768;
						gridChunk.x = 32768;
					}
					uint3 chunkSize = make_uint3(gridSize.x / gridSizeChunk.x, gridSize.y / gridSizeChunk.y, gridSize.z / gridSizeChunk.z);
					uint3 chunkPos = make_uint3(0);
					int streamID = 0;
					std::thread kernelThreads[NumberOfCudaStreams];
					for (int x = 0; x < chunkSize.x; x++)
					{
						for (int y = 0; y < chunkSize.y; y++)
						{
							for (int z = 0; z < chunkSize.z; z++)
							{
								chunkPos = make_uint3(x,y,z);
								if (streamID >= chunkThreads)
								{
									streamID = 0;
									for (int i = 0; i < chunkThreads; i++)
										kernelThreads[i].join();
								}

								kernelThreads[streamID] = std::thread(&MarchingCubesChunk::RunCudaStreamPerThread, this, gridChunk, voxelThreads, gridSizeChunk, gridSizeShiftChunk, gridSizeMaskChunk, bboxMin, bboxMax,
									adaptiveMapMin, adaptiveMapMax, meshTriangleSize, triangle_data, normal_data, sdfDictionary1DTemp, noiseData1DTemp, gridSize, chunkPos, true, octreeBboxListDevice, nO);
								streamID++;
								if (debugLevel >= 1)
									DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangleChunk %d / %d", chunkID,
										1 + z + y * chunkSize.z + x * chunkSize.y * chunkSize.z, chunkSize.x * chunkSize.y * chunkSize.z));
							}
						}
					}
					for (int i = 0; i < chunkThreads; i++)
						kernelThreads[i].join();

					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangleAll: %f", chunkID, sdkGetTimerValue(&timer)));
					SetTimer();
				}
				int* fliterCounter = new int[1];
				dim3 gridLayer(gridSize.x * gridSize.x / voxelThreads, 1, 1);
				// get around maximum grid size of 65535 in each dimension
				if (gridLayer.x > 65535)
				{
					gridLayer.y = grid.x / 32768;
					gridLayer.x = 32768;
				}
				bool enableFilter = true;
				int maxFilter = SVFSetting;
				if(enableFilter)
				for (int filterMode = 0; filterMode < maxFilter; filterMode++)
				{
					for (int x = 0; x < gridSize.x; x++)
					{
						int stopCounter = gridSize.x * gridSize.x;
						fliterCounter[0] = 1;
						while (fliterCounter[0] != 0)
						{
							stopCounter--;
							if (stopCounter < 0)
								break;
							fliterCounter[0] = 0;
							checkCudaErrors(cudaMemcpyAsync(d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
							checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
							checkCudaErrors(cudaEventSynchronize(stop_event[0]));
							launch_sampleTriangleFilter(cudaStreams[0], gridLayer, voxelThreads, gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax,
								adaptiveMapMin, adaptiveMapMax, meshTriangleSize, triangle_data, normal_data, sdfDictionary1DTemp, noiseData1DTemp,
								gridSize, gridSize, false, octreeBboxListDevice, nO, filterValue, d_triNum, x, filterMode);
							checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
							checkCudaErrors(cudaEventSynchronize(stop_event[0]));
							checkCudaErrors(cudaMemcpyAsync(fliterCounter, d_triNum, sizeof(int), cudaMemcpyDeviceToHost, cudaStreams[0]));
							checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
							checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						}
						//if (debugLevel >= 1)
						//	DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle Filter %d Layer: %d / %d; loops %d", chunkID,
						//		filterMode, x, gridSize.x, gridSize.x * gridSize.x - stopCounter));
					}
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle Filter %d Layer: %d / %d %f", chunkID,
							filterMode, filterMode, maxFilter, sdkGetTimerValue(&timer)));
					SetTimer();
				}
				//if (debugLevel >= 1)
				//	DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle Filter0All: %f", chunkID, sdkGetTimerValue(&timer)));
				//SetTimer();
				//filterMode = 1;
				//for (int x = 0; x < gridSize.x; x++)
				//{
				//	int stopCounter = gridSize.x * gridSize.x;
				//	//fliterCounter[0] = 1;
				//	//while (fliterCounter[0] != 0)
				//	//{
				//		stopCounter--;
				//		if (stopCounter < 0)
				//			break;
				//		fliterCounter[0] = 0;
				//		checkCudaErrors(cudaMemcpyAsync(d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
				//		checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//		checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				//		launch_sampleTriangleFilter(cudaStreams[0], gridLayer, voxelThreads, gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax,
				//			adaptiveMapMin, adaptiveMapMax, meshTriangleSize, triangle_data, normal_data, sdfDictionary1DTemp, noiseData1DTemp,
				//			gridSize, gridSize, false, octreeBboxListDevice, nO, filterValue, d_triNum, x, filterMode);
				//		checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//		checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				//		checkCudaErrors(cudaMemcpyAsync(fliterCounter, d_triNum, sizeof(int), cudaMemcpyDeviceToHost, cudaStreams[0]));
				//		checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//		checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				//	//}
				//	if (debugLevel >= 1)
				//		DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle Filter 1 Layer: %d / %d; loops %d", chunkID,
				//			x, gridSize.x, gridSize.x * gridSize.x - stopCounter));
				//}
				//if (debugLevel >= 1)
				//	DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle Filter1All: %f", chunkID, sdkGetTimerValue(&timer)));
				//SetTimer();
				//if (n < MaxArraySize)
				//{
				//	float* noiseData1DOut = (float*)malloc(n * sizeof(float));
				//	checkCudaErrors(cudaMemcpyAsync(noiseData1DOut, noiseData1DTemp, n * sizeof(float), cudaMemcpyDeviceToHost, cudaStreams[0]));
				//	checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				//	checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				//	sdfCallBack(noiseData1DOut, n);
				//	free(noiseData1DOut);
				//}
				//else
				//{
					sdfCallBack(NULL, n);
				//}

				free(triangleList);
				free(octreeBboxList);
				for (int i = 0; i < nO; i++)
				{
					checkCudaErrors(cudaFree(triangleListDevicePointers[i]));
				}
				checkCudaErrors(cudaFree(octreeBboxListDevice));
			}
			else if(hasInput)
			{
				for (int i = 0; i < chunkNumber; i++)
				{
					mcChunks[i].needUpdate = false;
					checkCudaErrors(cudaMalloc((void**)&mcChunks[i].brushArray, brushInputSize * sizeof(BrushInfo)));
				}
				BrushInfo* brushArrayTemp = (BrushInfo*)malloc(brushInputSize * sizeof(BrushInfo));
				for (int j = 0; j < brushInputSize; j++)
				{
					brushArrayTemp[j] = currentInputPosInBrush[j];
					//DebugLogToUnity(sFormator("BrushID [%d] : %d", chunkID, brushArrayTemp[j].brushID));
					//brushArrayTemp[j].brushID = currentInputPosInBrush[j].brushID;
					//brushArrayTemp[j].colorID = currentInputPosInBrush[j].colorID;
					//brushArrayTemp[j].shape = currentInputPosInBrush[j].shape;
					//brushArrayTemp[j].type = currentInputPosInBrush[j].type;
					brushArrayTemp[j].shape.w = -1;
				}
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :alloc brush buffer: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();

				if (chunkNumber == 1)
				{
					for (int i = 0; i < chunkNumber; i++)
					{
						if (debugLevel >= 2)
							DebugLogToUnity(sFormator("Child Chunk ID %d chunkbbox: bboxmax: [%f,%f,%f], bboxmin: [%f,%f,%f]",
								i, mcChunks[i].bboxMax.x, mcChunks[i].bboxMax.y, mcChunks[i].bboxMax.z,
								mcChunks[i].bboxMin.x, mcChunks[i].bboxMin.y, mcChunks[i].bboxMin.z));
						for (int j = 0; j < brushInputSize; j++)
						{
							inputBbox.center = make_float3(currentInputPosInBrush[j].shape.z, currentInputPosInBrush[j].shape.y, currentInputPosInBrush[j].shape.x);
							//Due to Unity's abnormal axis......It toke me 3 days to locate this bug......
							if (debugLevel >= 2)
								DebugLogToUnity(sFormator("Chunk ID %d input brush node [%d]: [%f,%f,%f]",
									chunkID, j, inputBbox.center.x, inputBbox.center.y, inputBbox.center.z));
							inputBbox.bboxMax = inputBbox.center + make_float3(currentInputRadius);
							inputBbox.bboxMin = inputBbox.center - make_float3(currentInputRadius);
							//inputBbox.bboxMax = clamp(inputBbox.bboxMax, parentBbox.bboxMin, parentBbox.bboxMax);
							//inputBbox.bboxMin = clamp(inputBbox.bboxMin, parentBbox.bboxMin, parentBbox.bboxMax);
							if (debugLevel >= 2)
								DebugLogToUnity(sFormator("Chunk ID %d inputbbox: bboxmax: [%f,%f,%f], bboxmin: [%f,%f,%f]",
									chunkID, inputBbox.bboxMax.x, inputBbox.bboxMax.y, inputBbox.bboxMax.z,
									inputBbox.bboxMin.x, inputBbox.bboxMin.y, inputBbox.bboxMin.z));
							if (IsBBoxOverlap(mcChunks[i].bboxMax, mcChunks[i].bboxMin, inputBbox.bboxMax, inputBbox.bboxMin))
							{
								mcChunks[i].needUpdate = true;
								brushArrayTemp[j].shape.w = currentInputPosInBrush[j].shape.w;
							}
							else
							{
								brushArrayTemp[j].shape.w = -1;
							}
						}
						if (mcChunks[i].needUpdate)
						{
							checkCudaErrors(cudaMemcpyAsync(mcChunks[i].brushArray, brushArrayTemp, brushInputSize * sizeof(BrushInfo), cudaMemcpyHostToDevice, cudaStreams[0]));
							checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
							checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						}
					}
				}
				else
				{
					for (int i = 0; i < chunkNumber; i++)
					{
						mcChunks[i].brushArrayTemp = (BrushInfo*)malloc(brushInputSize * sizeof(BrushInfo));
						memcpy(mcChunks[i].brushArrayTemp, brushArrayTemp, brushInputSize * sizeof(BrushInfo));
						//DebugLogToUnity(sFormator("BrushID [%d] : %d", chunkID, mcChunks[i].brushArrayTemp[0].brushID));
					}
					for (int j = 0; j < brushInputSize; j++)
					{
						inputBbox.center = make_float3(currentInputPosInBrush[j].shape.z, currentInputPosInBrush[j].shape.y, currentInputPosInBrush[j].shape.x);
						if (debugLevel >= 2)
							DebugLogToUnity(sFormator("Chunk ID %d input brush node [%d]: [%f,%f,%f]",
								chunkID, j, inputBbox.center.x, inputBbox.center.y, inputBbox.center.z));
						inputBbox.bboxMax = inputBbox.center + make_float3(currentInputRadius);
						inputBbox.bboxMin = inputBbox.center - make_float3(currentInputRadius);
						if (debugLevel >= 2)
							DebugLogToUnity(sFormator("Chunk ID %d inputbbox: bboxmax: [%f,%f,%f], bboxmin: [%f,%f,%f]",
								chunkID, inputBbox.bboxMax.x, inputBbox.bboxMax.y, inputBbox.bboxMax.z,
								inputBbox.bboxMin.x, inputBbox.bboxMin.y, inputBbox.bboxMin.z));
						ScanOctreeBbox(octreeBboxChunks[0], octreeBboxChunks[0].octreeDepth, octreeBboxChunks[0].maxOctreeLeafSize, inputBbox.bboxMax, inputBbox.bboxMin, j);
					}
					for (int i = 0; i < chunkNumber; i++)
					{
						if (debugLevel >= 2)
							DebugLogToUnity(sFormator("Child Chunk ID %d chunkbbox: bboxmax: [%f,%f,%f], bboxmin: [%f,%f,%f]",
								i, mcChunks[i].bboxMax.x, mcChunks[i].bboxMax.y, mcChunks[i].bboxMax.z,
								mcChunks[i].bboxMin.x, mcChunks[i].bboxMin.y, mcChunks[i].bboxMin.z));
						if (mcChunks[i].needUpdate)
						{
							checkCudaErrors(cudaMemcpyAsync(mcChunks[i].brushArray, mcChunks[i].brushArrayTemp, brushInputSize * sizeof(BrushInfo), cudaMemcpyHostToDevice, cudaStreams[0]));
							checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
							checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						}
						free(mcChunks[i].brushArrayTemp);
					}
				}
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :assign brush buffer: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
				int updatedChunkCounter = 0;
				for (int i = 0; i < chunkNumber; i++)
				{
					if (mcChunks[i].needUpdate)
					{
						updatedChunkCounter++;
						if (debugLevel >= 1)
							DebugLogToUnity(sFormator("Child Chunk ID %d : input-needUpdate = true", i));
						launch_insertShapeToVoxel(cudaStreams[0], gridChunk, voxelThreads, sdfDictionary1DTemp, noiseData1DTemp, gridSizeChunk,
							gridSizeShiftChunk, gridSizeMaskChunk, currentInputRadius, mcChunks[i].brushArray, brushList, brushVoxels, enableColorSmoothing, brushInputSize,
							brushColorType, colorBrushOffset, currentEaserMode, brushShapeType, gridSize, mcChunks[i].chunkPos, true);
						checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
						checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						if(updatedChunkCounter > maxUpatedChunk && updatedChunkCounter % maxUpatedChunk == 0)
							Sleep(sleepTime);
					}
					checkCudaErrors(cudaFree(mcChunks[i].brushArray));
				}
				free(brushArrayTemp);
				free(currentInputPosInBrush);
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d :launch_insertSphereToVoxel: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
			}
			else
			{
				for (int i = 0; i < chunkNumber; i++)
					mcChunks[i].needUpdate = false;
			}

			if (hasBoolOp)
			{
				hasBoolOp = false;
				//launch_insertSphereToVoxel(cudaStreams[0], grid, voxelThreads, noiseData1DTemp, gridSize,
				//	gridSizeShift, gridSizeMask, currentInputRadius, currentInputPos, currentEaserMode);
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			}
			//checkCudaErrors(cudaStreamSynchronize(cudaStreams[0]));
			//if (debugLevel >= 1)
			//	DebugLogToUnity(sFormator("Chunk ID %d :launch_sampleTriangle: %f", chunkID, sdkGetTimerValue(&timer)));
			//SetTimer();

			if (debugLevel >= 2)
			{
				ushort2* noiseData1DOut = (ushort2*)malloc(n * sizeof(ushort2));
				checkCudaErrors(cudaMemcpyAsync(noiseData1DOut, noiseData1DTemp, n * sizeof(ushort2), cudaMemcpyDeviceToHost, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				for (int i = 0; i < n; i++)
					DebugLogToUnity(sFormator("noiseData1DOut [%d] : [%f, %f] ", i, sdfDictionary1D[noiseData1DOut[i].x].x, sdfDictionary1D[noiseData1DOut[i].y].y));
				free(noiseData1DOut);
			}

			//checkCudaErrors(cudaMemcpyAsync(d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
			//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			//checkCudaErrors(cudaEventSynchronize(stop_event[0]));

			if (debugLevel >= 2)
				DebugLogToUnity(sFormator("Chunk ID %d : bboxmax: [%f,%f,%f], bboxmin: [%f,%f,%f]",
					chunkID, parentBbox.bboxMax.x, parentBbox.bboxMax.y, parentBbox.bboxMax.z, 
					parentBbox.bboxMin.x, parentBbox.bboxMin.y, parentBbox.bboxMin.z));

			for (int i = 0; i < chunkNumber; i++)
			{
				if (loadMesh)
				{
					mcChunks[i].needUpdate = true;
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Child Chunk ID %d : init-needUpdate = true", i));
				}
				if (mcChunks[i].needUpdate)
				{
					ReuseMemoryInChunk(mcChunks[i]);
					// calculate number of vertices need per voxel
					checkCudaErrors(cudaMemcpyAsync(mcChunks[i].d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					launch_classifyVoxel(mcChunks[i].d_triNum, cudaStreams[0],sdfDictionary1DTemp, noiseData1DTemp, gridChunk, voxelThreads,
						mcChunks[i].d_voxelVerts, mcChunks[i].d_voxelOccupied,
						gridSizeChunk, gridSizeShiftChunk, gridSizeMaskChunk,
						voxelNumInChunk, voxelSize, IsoLevel, gridSize, mcChunks[i].chunkPos, true);
					checkCudaErrors(cudaMemcpyAsync(mcChunks[i].h_triNum, mcChunks[i].d_triNum, sizeof(int), cudaMemcpyDeviceToHost, cudaStreams[0]));
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					checkCudaErrors(cudaMemcpyAsync(mcChunks[i].d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
					checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					//if (debugLevel >= 1)
					//	DebugLogToUnity(sFormator("Child Chunk ID %d :triNum = %d", i, mcChunks[i].h_triNum[0]));
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Child Chunk ID %d :launch_classifyVoxel: %f", i, sdkGetTimerValue(&timer)));
					SetTimer();
#if SKIP_EMPTY_VOXELS
					if (debugLevel >= 3)
						dumpBuffer(mcChunks[i].d_voxelVerts, voxelNumInChunk, sizeof(int), "voxelVerts");
					if (debugLevel >= 3)
						dumpBuffer(mcChunks[i].d_voxelOccupied, voxelNumInChunk, sizeof(int), "voxelOccupied");

					ThrustScanWrapper(mcChunks[i].d_voxelOccupiedScan, mcChunks[i].d_voxelOccupied, (int)voxelNumInChunk);
					//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
					//checkCudaErrors(cudaEventSynchronize(stop_event[0]));
					if (debugLevel >= 3)
						dumpBuffer(mcChunks[i].d_voxelOccupiedScan, voxelNumInChunk, sizeof(int), "voxelOccupiedScan");

					{
						int lastElement, lastScanElement;
						checkCudaErrors(cudaMemcpyAsync((void*)&lastElement,
							(void*)(mcChunks[i].d_voxelOccupied + voxelNumInChunk - 1),
							sizeof(int), cudaMemcpyDeviceToHost, cudaStreams[0]));
						checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
						checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						checkCudaErrors(cudaMemcpyAsync((void*)&lastScanElement,
							(void*)(mcChunks[i].d_voxelOccupiedScan + voxelNumInChunk - 1),
							sizeof(int), cudaMemcpyDeviceToHost, cudaStreams[0]));
						checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
						checkCudaErrors(cudaEventSynchronize(stop_event[0]));
						mcChunks[i].activeVoxels = lastElement + lastScanElement;
					}
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Child Chunk ID %d : activeVoxels： %d", i, mcChunks[i].activeVoxels));
					if (mcChunks[i].activeVoxels == 0)
					{
						if (debugLevel >= 1)
							DebugLogToUnity(sFormator("Child Chunk ID %d : has no voxel", i));
						// return if there are no full voxels
						//totalVerts = 0;
						//triCount = 0;
						//mcCallBack(chunkID);
						//return totalVerts;
					}

					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Child Chunk ID %d :PrefixSum: %f", i, sdkGetTimerValue(&timer)));
					SetTimer();
					// compact voxel index array
					CudaUnity::launch_compactVoxels(cudaStreams[0], gridChunk, voxelThreads, mcChunks[i].d_compVoxelArray, mcChunks[i].d_voxelOccupied, mcChunks[i].d_voxelOccupiedScan, voxelNumInChunk);
					if (debugLevel >= 1)
						DebugLogToUnity(sFormator("Child Chunk ID %d :launch_compactVoxels: %f", i, sdkGetTimerValue(&timer)));
					SetTimer();
					if (debugLevel >= 3)
						dumpBuffer(mcChunks[i].d_compVoxelArray, voxelNumInChunk, sizeof(int), "compVoxelArray");
#endif
				}

			}

			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d : Update All Chunk: %f", chunkID, sdkGetTimerValue(&timer)));
			SetTimer();

			h_triNum[0] = 0;
			for (int i = 0; i < chunkNumber; i++)
			{
				mcChunks[i].tribufferStart = h_triNum[0];
				mcChunks[i].tribufferEnd = mcChunks[i].h_triNum[0] + h_triNum[0];
				h_triNum[0] += mcChunks[i].h_triNum[0];
				if(mcChunks[i].needUpdate)
					meshChunkCallBack(mcChunks[i].chunkID, mcChunks[i].needUpdate, mcChunks[i].h_triNum[0]);
			}
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d : triNum = %d; time: %f", chunkID, h_triNum[0], sdkGetTimerValue(&timer)));
			SetTimer();

			//mcCallBack(chunkID);
			//return 0;

			if (hasInput)
				hasInput = false;

			if (h_triNum[0] > 0)
			{
				lastTriNum = h_triNum[0];
				triCount = h_triNum[0];
				mcCallBack(chunkID);
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d : triangle callback: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
				return h_triNum[0];
			}
			else
			{
				triCount = lastTriNum;
				mcCallBack(chunkID);
				if (debugLevel >= 1)
					DebugLogToUnity(sFormator("Chunk ID %d : triangle callback: %f", chunkID, sdkGetTimerValue(&timer)));
				SetTimer();
				return lastTriNum;
			}
		}

		void SetMarchingCubesThread(int size, int voxelThreads,
			int triangleThreads, float3 CenterPosI, float GridWI, float octreeBboxSizeOffset,
			float IsoLevel, bool EnableSmooth, bool loadMesh, bool loadSdf, bool loadSdfFromUnity, ushort2* sdfData,
			int gridSizeLog2OBox, bool exportTexture3D, string sdfFileName, float filterValue, int sleepTime, int maxUpatedChunk, int SVFSetting)
		{
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d :start thread", chunkID));
			SetTimer();
			kernelThread = std::thread(&MarchingCubesChunk::SetMarchingCubesKernel, this, size, voxelThreads, triangleThreads,
				CenterPosI, GridWI, octreeBboxSizeOffset, IsoLevel, EnableSmooth, loadMesh,loadSdf, loadSdfFromUnity,
				sdfData, gridSizeLog2OBox, exportTexture3D, sdfFileName, filterValue, sleepTime, maxUpatedChunk, SVFSetting);
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d :end to start thread: %f", chunkID, sdkGetTimerValue(&timer)));
			SetTimer();
			kernelThread.detach();
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Chunk ID %d :detached thread: %f", chunkID, sdkGetTimerValue(&timer)));
			SetTimer();
		}

		bool IsNeedUpdate(int childChunkID)
		{
			return mcChunks[childChunkID].needUpdate;
		}

		void GetExtractMarchingCubesChunkData(int childChunkID, int size, Vector3* vertices, Vector3* normals, int* triangles, int* colors
			, float GridWI, int triangleThreads, float IsoLevel, bool EnableSmooth)
		{
			cudaSetDevice(cudaDeviceID);
			int i = 0;
			int totalTriNum = mcChunks[childChunkID].tribufferEnd - mcChunks[childChunkID].tribufferStart;

			checkCudaErrors(cudaMemcpyAsync(mcChunks[childChunkID].d_triNum, z_triNum, sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));

			checkCudaErrors(cudaMalloc((void**)&mcChunks[childChunkID].d_Tribuffer, mcChunks[childChunkID].h_triNum[0] * sizeof(Triangle)));
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Child Chunk ID %d :reset CUDA buffers: %f", childChunkID, sdkGetTimerValue(&timer)));
			SetTimer();
#if SKIP_EMPTY_VOXELS
			dim3 gridChunk2((int)ceil(mcChunks[childChunkID].activeVoxels / triangleThreads), 1, 1);
#else
			dim3 gridChunk2((int)ceil(voxelNumInChunk / triangleThreads), 1, 1);
#endif
			while (gridChunk2.x > 65535)
			{
				gridChunk2.x /= 2;
				gridChunk2.y *= 2;
			}
			launch_generateTriangles(cudaStreams[0], mcChunks[childChunkID].h_triNum[0], mcChunks[childChunkID].d_triNum, 
				sdfDictionary1DTemp, noiseData1DTemp, CenterPos, gridSize.x,
				GridWI, gridChunk2, triangleThreads,mcChunks[childChunkID].d_compVoxelArray, 
				gridSizeChunk, gridSizeShiftChunk, gridSizeMaskChunk,
				voxelSize, IsoLevel, mcChunks[childChunkID].activeVoxels,
				maxVerts, mcChunks[childChunkID].d_Tribuffer, EnableSmooth
				, gridSize, mcChunks[childChunkID].chunkPos, true, brushList,
				brushVoxels, enableColorSmoothing, brushListTemp.size());
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]))

			Tribuffer = (Triangle*)malloc(totalTriNum * sizeof(Triangle));
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Child Chunk ID %d :freeAndAlloc(Tribuffer): %f", childChunkID, sdkGetTimerValue(&timer)));
			SetTimer();

			checkCudaErrors(cudaMemcpyAsync(Tribuffer, mcChunks[childChunkID].d_Tribuffer, totalTriNum * sizeof(Triangle), cudaMemcpyDeviceToHost, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			if (debugLevel >= 1)
				DebugLogToUnity(sFormator("Child Chunk ID %d :cudaMemcpy(Tribuffer): %f", childChunkID, sdkGetTimerValue(&timer)));
			SetTimer();
			for (int j = 0; j < totalTriNum; j++)
			{
				vertices[i * 3] = Tribuffer[j].posA;
				vertices[i * 3 + 1] = Tribuffer[j].posB;
				vertices[i * 3 + 2] = Tribuffer[j].posC;
				normals[i * 3] = Tribuffer[j].normalA;
				normals[i * 3 + 1] = Tribuffer[j].normalB;
				normals[i * 3 + 2] = Tribuffer[j].normalC;
				triangles[i * 3] = i * 3;
				triangles[i * 3 + 1] = i * 3 + 1;
				triangles[i * 3 + 2] = i * 3 + 2;

				if (enableColorSmoothing)
				{
					for (int k = 0; k < 15; k++)
					{
						colors[j * 15 + k] = Tribuffer[j].Colors[k];
					}
				}
				for (int k = 0; k < 11; k++)
				{
					if (k < 8)
						colors[j * 11 + k] = Tribuffer[j].Colors[k];
					else if (k == 8)
						colors[j * 11 + k] = Tribuffer[j].cubeInfo.x;
					else if (k == 9)
						colors[j * 11 + k] = Tribuffer[j].cubeInfo.y;
					else if (k == 10)
						colors[j * 11 + k] = Tribuffer[j].cubeInfo.z;
				}

				//for (int k = 0; k < 8; k++)
				//{
				//	colors[j * 8 + k] = Tribuffer[j].Colors[k];
				//}
				//colors[i * 3] = Tribuffer[j].Color;
				//colors[i * 3 + 1] = Tribuffer[j].Color;
				//colors[i * 3 + 2] = Tribuffer[j].Color;
				//for (int k = 0; k < 8; k++)
				//	colors[j * 8 + k] = Tribuffer[j].Color;
				//colors[j * 8 + 1] = Tribuffer[j].ColorA.y;
				//colors[j * 8 + 2] = Tribuffer[j].ColorA.z;
				//colors[j * 8 + 3] = Tribuffer[j].ColorA.w;
				//colors[j * 8 + 4] = Tribuffer[j].ColorB.x;
				//colors[j * 8 + 5] = Tribuffer[j].ColorB.y;
				//colors[j * 8 + 6] = Tribuffer[j].ColorB.z;
				//colors[j * 8 + 7] = Tribuffer[j].ColorB.w;
				i += 1;
				//if (debugLevel == 0)
				//{
				//	DebugLogToUnity(sFormator("TotalTriNum %d Tribuffer [%d] : posA(%f, %f, %f), posB(%f, %f, %f), posC(%f, %f, %f)", totalTriNum, j,
				//		Tribuffer[j].posA.x, Tribuffer[j].posA.y, Tribuffer[j].posA.z,
				//		Tribuffer[j].posB.x, Tribuffer[j].posB.y, Tribuffer[j].posB.z,
				//		Tribuffer[j].posC.x, Tribuffer[j].posC.y, Tribuffer[j].posC.z));
				//	DebugLogToUnity(sFormator("TotalTriNum %d Tribuffer [%d]: norA(%f, %f, %f), norB(%f, %f, %f), norC(%f, %f, %f)", totalTriNum, j,
				//		Tribuffer[j].normalA.x, Tribuffer[j].normalA.y, Tribuffer[j].normalA.z,
				//		Tribuffer[j].normalB.x, Tribuffer[j].normalB.y, Tribuffer[j].normalB.z,
				//		Tribuffer[j].normalC.x, Tribuffer[j].normalC.y, Tribuffer[j].normalC.z));
				//}
			}
			triangleCounter = 0;
			if (debugLevel >= 2)
			{
				for (int i = 0; i < totalTriNum; i++)
				{
					triangleCounter += 1;
					DebugLogToUnity(sFormator("Tri Counter %d Tribuffer [%d] : posA(%f, %f, %f), posB(%f, %f, %f), posC(%f, %f, %f)", triangleCounter, i,
						Tribuffer[i].posA.x, Tribuffer[i].posA.y, Tribuffer[i].posA.z,
						Tribuffer[i].posB.x, Tribuffer[i].posB.y, Tribuffer[i].posB.z,
						Tribuffer[i].posC.x, Tribuffer[i].posC.y, Tribuffer[i].posC.z));
					DebugLogToUnity(sFormator("Tri Counter %d Tribuffer [%d]: norA(%f, %f, %f), norB(%f, %f, %f), norC(%f, %f, %f)", triangleCounter, i,
						Tribuffer[i].normalA.x, Tribuffer[i].normalA.y, Tribuffer[i].normalA.z,
						Tribuffer[i].normalB.x, Tribuffer[i].normalB.y, Tribuffer[i].normalB.z,
						Tribuffer[i].normalC.x, Tribuffer[i].normalC.y, Tribuffer[i].normalC.z));
				}
			}
			checkCudaErrors(cudaFree(mcChunks[childChunkID].d_Tribuffer));
			free(Tribuffer);
		}

		void GetExtractMarchingCubesData(int size, Vector3 *vertices, Vector3 *normals, int *triangles)
		{
			int i = 0;
			for (int j = 0; j < h_triNum[0]; j++)
			{
				vertices[i * 3] = Tribuffer[j].posA;
				vertices[i * 3 + 1] = Tribuffer[j].posB;
				vertices[i * 3 + 2] = Tribuffer[j].posC;
				normals[i * 3] = Tribuffer[j].normalA;
				normals[i * 3 + 1] = Tribuffer[j].normalB;
				normals[i * 3 + 2] = Tribuffer[j].normalC;
				triangles[i * 3] = i * 3;
				triangles[i * 3 + 1] = i * 3 + 1;
				triangles[i * 3 + 2] = i * 3 + 2;
				i += 1;
			}
		}

		bool GetExtractCubeVoxels(float* cubeVoxel, int size)
		{
			//int n = gridSize.x * gridSize.y * gridSize.z;
			//checkCudaErrors(cudaMemcpyAsync(cubeVoxel, noiseData1DTemp, n * sizeof(float), cudaMemcpyDeviceToHost, cudaStreams[0]));
			//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			//checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			return true;
		}

		void InitOCtreeBbox(OctreeBbox& octreeBboxNode, int octreeDepth, int maxOctreeLeafSize)
		{
			if (pow(2, octreeDepth) > maxOctreeLeafSize)
				return;
			if (octreeBboxNode.init == false)
			{
				octreeBboxNode.init = true;
				octreeBboxNode.octreeDepth = octreeDepth;
				octreeBboxNode.maxOctreeLeafSize = maxOctreeLeafSize;
				octreeBboxNode.octreeLeafSize = pow(2, octreeDepth);
				octreeBboxNode.empty = false;
				OctreeBbox* leafNodes = (OctreeBbox*)malloc(8 * sizeof(OctreeBbox));
				float3 bboxChunkSize = (parentBbox.bboxMax - parentBbox.bboxMin) / octreeBboxNode.octreeLeafSize;
				for (int i = 0; i < 8; i++)
				{
					//octreeBboxNode.leafNodes[i].parentNode = (OctreeBbox*)malloc(1 * sizeof(OctreeBbox));
					//octreeBboxNode.leafNodes[i].parentNode[0] = octreeBboxNode;
					octreeBboxNode.leafNodesID[i] = h_octreeBboxNodesCounter;
					leafNodes[i].bboxChunkSize = bboxChunkSize;
					leafNodes[i].octreeDepth = octreeBboxNode.octreeDepth + 1;
					leafNodes[i].boxCount = 8;
					leafNodes[i].empty = false;
					leafNodes[i].init = false;
					leafNodes[i].chunkID = 0;
					if (debugLevel >= 2)
						DebugLogToUnity(sFormator("Insert OctreeNode [%d] in Leaf[%d], Depth[%d], MaxLeaf[%d]",
							h_octreeBboxNodesCounter, octreeBboxNode.octreeLeafSize, octreeDepth, maxOctreeLeafSize));

				}
				int counter = 0;
				for (int x = 0; x < 2; x++)
				{
					for (int y = 0; y < 2; y++)
					{
						for (int z = 0; z < 2; z++)
						{
							leafNodes[counter].chunkPos = make_uint3(x, y, z);
							float3 bboxPos = make_float3(bboxChunkSize.x * x,
								bboxChunkSize.y * y, bboxChunkSize.z * z);
							leafNodes[counter].bboxMax = octreeBboxNode.bboxMin + bboxPos + bboxChunkSize;
							leafNodes[counter].bboxMin = octreeBboxNode.bboxMin + bboxPos;
							if (debugLevel >= 2)
							{
								DebugLogToUnity(sFormator("OctreeBbox octreeDepth[%d] counter", octreeDepth));
								DebugLogToUnity(sFormator("OctreeBbox octreeDepth[%d]: bboxMax(%f, %f, %f), bboxMin(%f, %f, %f)", octreeDepth,
									leafNodes[counter].bboxMax.x, leafNodes[counter].bboxMax.y, leafNodes[counter].bboxMax.z,
									leafNodes[counter].bboxMin.x, leafNodes[counter].bboxMin.y, leafNodes[counter].bboxMin.z));
							}
							if (pow(2, octreeDepth + 1) > maxOctreeLeafSize)
							{
								for (int i = 0; i < chunkNumber; i++)
								{
									if (length(mcChunks[i].bboxMax - leafNodes[counter].bboxMax) == 0 &&
										length(mcChunks[i].bboxMin - leafNodes[counter].bboxMin) == 0)
									{
										leafNodes[counter].chunkID = i;
										if (debugLevel >= 2)
											DebugLogToUnity(sFormator("OctreeBbox octreeDepth[%d] assign Chunk[%d]", octreeDepth, leafNodes[counter].chunkID));
										break;
									}
								}
							}
							octreeBboxNode.chunkID = leafNodes[counter].chunkID;
							octreeBboxNode.leafNodesID[counter] = h_octreeBboxNodesCounter;
							h_octreeBboxNodes[h_octreeBboxNodesCounter].chunkID = leafNodes[counter].chunkID;
							h_octreeBboxNodes[h_octreeBboxNodesCounter] = leafNodes[counter];
							if (debugLevel >= 2)
								DebugLogToUnity(sFormator("OctreeBbox[%d] octreeDepth[%d] assign Chunk[%d] leafNodesID[%d]", 
									counter, octreeDepth, h_octreeBboxNodes[h_octreeBboxNodesCounter].chunkID, octreeBboxNode.leafNodesID[counter]));
							counter++;
							h_octreeBboxNodesCounter++;
						}
					}
				}
			}
			for (int i = 0; i < 8; i++)
				InitOCtreeBbox(h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]],
					h_octreeBboxNodes[octreeBboxNode.leafNodesID[i]].octreeDepth, maxOctreeLeafSize);
		}

		bool MallocMemoryForMC(Vector3 size, float GridWI)
		{
			cudaSetDevice(cudaDeviceID);
			cudaStreams = (cudaStream_t*)malloc(chunkThreads * sizeof(cudaStream_t));
			stop_event = (cudaEvent_t*)malloc(chunkThreads * sizeof(cudaEvent_t));
			kernelEvent = (cudaEvent_t*)malloc(chunkThreads * sizeof(cudaEvent_t));
			for (int i = 0; i < chunkThreads; i++)
			{
				checkCudaErrors(cudaEventCreateWithFlags(&kernelEvent[i], cudaEventDisableTiming));
				checkCudaErrors(cudaEventCreateWithFlags(&stop_event[i], eventflags));
				checkCudaErrors(cudaStreamCreateWithFlags(&cudaStreams[i], cudaStreamNonBlocking));
			}

			canFree = true;
			sdkCreateTimer(&timer);
			gridSizeLog2 = make_uint3(size.x, size.y, size.z);
			//DebugLogToUnity(sFormator("gridSizeLog2 = (%d, %d, %d)", gridSizeLog2.x, gridSizeLog2.y, gridSizeLog2.z));
			gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
			gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
			gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

			numVoxels = gridSize.x*gridSize.y*gridSize.z;
			
			voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
			maxVerts = 4 * gridSize.x * gridSize.y * gridSize.z;
			int voxelNum = gridSize.x * gridSize.y * gridSize.z;

			int voxelThreads = 128;
			gridSizeLog2Chunk = make_uint3(minChunkSizeLog2);
			gridSizeChunk = make_uint3(1 << gridSizeLog2Chunk.x, 1 << gridSizeLog2Chunk.y, 1 << gridSizeLog2Chunk.z);
			gridSizeMaskChunk = make_uint3(gridSizeChunk.x - 1, gridSizeChunk.y - 1, gridSizeChunk.z - 1);
			gridSizeShiftChunk = make_uint3(0, gridSizeLog2Chunk.x, gridSizeLog2Chunk.x + gridSizeLog2Chunk.y);
			voxelNumInChunk = gridSizeChunk.x * gridSizeChunk.y * gridSizeChunk.z;
			dim3 gridChunk(chunkNumber / voxelThreads, 1, 1);
			if (gridChunk.x > 65535)
			{
				gridChunk.y = gridChunk.x / 32768;
				gridChunk.x = 32768;
			}
			chunkSize = make_uint3(gridSize.x / gridSizeChunk.x, gridSize.y / gridSizeChunk.y, gridSize.z / gridSizeChunk.z);
			uint3 chunkPos = make_uint3(0);
			chunkNumber = chunkSize.x * chunkSize.y * chunkSize.z;
			mcChunks = (McChunk*)malloc(chunkNumber * sizeof(McChunk));	
			//DebugLogToUnity(sFormator("Chunk size: %d,%d,%d", chunkSize.x, chunkSize.y, chunkSize.z));
			//DebugLogToUnity(sFormator("voxelNumInChunk %d", voxelNumInChunk));
			parentBbox.bboxMin = make_float3(0);
			parentBbox.bboxMax = make_float3(gridSize - make_uint3(1));
			float3 bboxChunkSize = (parentBbox.bboxMax - parentBbox.bboxMin) / chunkSize.x;

			if (enableColorSmoothing)
			{
				brushVoxelsTemp = (BrushVoxel*)malloc(numVoxels * sizeof(BrushVoxel));
				for (int i = 0; i < numVoxels; i++)
				{
					brushVoxelsTemp[i].brushID = 999999999;
					brushVoxelsTemp[i].lastBrushID = 999999999;
					brushVoxelsTemp[i].backgroundColor = 0;
					brushVoxelsTemp[i].lastBackgroundColor = 0;
				}
				checkCudaErrors(cudaMalloc((void**)&brushVoxels, numVoxels * sizeof(BrushVoxel)));
				checkCudaErrors(cudaMemcpyAsync(brushVoxels, brushVoxelsTemp, numVoxels * sizeof(BrushVoxel), cudaMemcpyHostToDevice, cudaStreams[0]));
				checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
				checkCudaErrors(cudaEventSynchronize(stop_event[0]));
				brushListTemp.clear();
			}

			h_TribufferAll = (Triangle**)malloc(chunkNumber * sizeof(Triangle*));
			h_triNumAll = (int**)malloc(chunkNumber * sizeof(int*));
#if SKIP_EMPTY_VOXELS
			h_compVoxelArrayAll = (int**)malloc(chunkNumber * sizeof(int*));
			checkCudaErrors(cudaMalloc((void**)&d_voxelVertsAll, sizeof(int) * voxelNumInChunk));
			checkCudaErrors(cudaMalloc((void**)&d_voxelVertsScanAll, sizeof(int) * voxelNumInChunk));
			checkCudaErrors(cudaMalloc((void**)&d_voxelOccupiedAll, sizeof(int) * voxelNumInChunk));
			checkCudaErrors(cudaMalloc((void**)&d_voxelOccupiedScanAll, sizeof(int) * voxelNumInChunk));
#endif

			for (int i = 0; i < chunkNumber; i++)
			{
				mcChunks[i].chunkID = i;
				mcChunks[i].needUpdate = true;
				mcChunks[i].chunkSize = chunkSize;
				mcChunks[i].h_triNum = new int[1];
				mcChunks[i].h_triNum[0] = 0;
				checkCudaErrors(cudaMalloc((void**)&h_TribufferAll[i], sizeof(Triangle)));
				checkCudaErrors(cudaMalloc((void**)&h_triNumAll[i], sizeof(int)));
				mcChunks[i].d_triNum = h_triNumAll[i];
				mcChunks[i].d_Tribuffer = h_TribufferAll[i];
#if SKIP_EMPTY_VOXELS
				checkCudaErrors(cudaMalloc((void**)&h_compVoxelArrayAll[i], sizeof(int) * voxelNumInChunk));
				mcChunks[i].d_voxelVerts = d_voxelVertsAll;
				mcChunks[i].d_voxelVertsScan = d_voxelVertsScanAll;
				mcChunks[i].d_voxelOccupied = d_voxelOccupiedAll;
				mcChunks[i].d_voxelOccupiedScan = d_voxelOccupiedScanAll;
				mcChunks[i].d_compVoxelArray = h_compVoxelArrayAll[i];
#endif
			}


			int counter = 0;
			for (int x = 0; x < chunkSize.x; x++)
			{
				for (int y = 0; y < chunkSize.y; y++)
				{
					for (int z = 0; z < chunkSize.z; z++)
					{
						mcChunks[counter].chunkPos = make_uint3(x, y, z);
						float3 bboxPos = make_float3(bboxChunkSize.x * x,
							bboxChunkSize.y * y, bboxChunkSize.z * z);
						mcChunks[counter].bboxMax = parentBbox.bboxMin + bboxPos + bboxChunkSize;
						mcChunks[counter].bboxMin = parentBbox.bboxMin + bboxPos;
						counter++;
					}
				}
			}
			int octreeDepthTemp = 1;
			int octreeDepthNodeCounter = 0;
			while ((pow(2, octreeDepthTemp) <= chunkSize.x))
			{
				octreeDepthNodeCounter += pow(2, octreeDepthTemp) * pow(2, octreeDepthTemp) * pow(2, octreeDepthTemp);
				octreeDepthTemp++;
				//if (debugLevel >= 2)
				//	DebugLogToUnity(sFormator("OctreeBbox node number: %d", octreeDepthNodeCounter));
			}
			octreeBboxChunks = (OctreeBbox*)malloc(1 * sizeof(OctreeBbox));
			h_octreeBboxNodes = (OctreeBbox*)malloc(octreeDepthNodeCounter * sizeof(OctreeBbox));
			h_octreeBboxNodesCounter = 0;
			octreeBboxChunks[0].octreeDepth = 0;
			if (chunkSize.x == 1)
			{
				octreeBboxChunks[0].empty = true;
				octreeBboxChunks[0].init = true;
			}
			else
			{
				octreeBboxChunks[0].maxOctreeLeafSize = chunkSize.x;
				octreeBboxChunks[0].octreeLeafSize = 2;
				octreeBboxChunks[0].empty = false;
				octreeBboxChunks[0].init = false;
				octreeBboxChunks[0].chunkPos = make_uint3(0);
				octreeBboxChunks[0].bboxMax = parentBbox.bboxMax;
				octreeBboxChunks[0].bboxMin = parentBbox.bboxMin;
				octreeBboxChunks[0].bboxChunkSize = (parentBbox.bboxMax - parentBbox.bboxMin) / 2;
				//octreeBboxChunks[0].parentNode = (OctreeBbox*)malloc(1 * sizeof(OctreeBbox));
				//octreeBboxChunks[0].parentNode[0] = octreeBboxChunks[0];
				octreeBboxChunks[0].octreeDepth = 1;
				InitOCtreeBbox(octreeBboxChunks[0], octreeBboxChunks[0].octreeDepth, octreeBboxChunks[0].maxOctreeLeafSize);
			}
			//meshSampleData1D = (float*)malloc(voxelNum * sizeof(float));
			//for (int i = 0; i < voxelNum; i++)
			//	meshSampleData1D[i] = -0.0000001;
			sdfDictionary1D = (float2*)malloc(DictionarySize * sizeof(float2));
			sdfDictionary1D[0] = make_float2(MinVoxel, 0);
			sdfDictionary1D[1] = make_float2(0, 1);
			for (int i = 2; i < DictionarySize; i+=2)
			{
				if (i <= ClampSize)
				{
					int clampID = (int)i / 2;
					sdfDictionary1D[i] = make_float2(clampID * 0.001, i);
					sdfDictionary1D[i+1] = make_float2(-clampID * 0.001, i + 1);
				}
				else
				{
					int clampID = (int)(i - ClampSize) / 2 - 2;
					sdfDictionary1D[i] = make_float2(ClampBound + clampID * 0.1, i);
					sdfDictionary1D[i + 1] = make_float2(-(ClampBound + clampID * 0.1), i + 1);
				}
				if (debugLevel >= 2)
					DebugLogToUnity(sFormator("sdfDictionary1D [%d]: [%f], [%d]: [%f] ",
						i, sdfDictionary1D[i].x, i+1, sdfDictionary1D[i + 1].x));
			}	

			if (debugLevel >= 2)
			{
				for (int i = 0; i < DictionarySize; i++)
				{
					DebugLogToUnity(sFormator("sdfDictionary1D Key [%f] [%f] [%d] [%d]",
						sdfDictionary1D[i].x, GetSdfVectorValue(sdfDictionary1D[i].x), GetSdfVectorKey(GetSdfVectorValue(sdfDictionary1D[i].x)), GetSdfVectorKey(sdfDictionary1D[i].x)));
				}
				for (float i = -5; i < 5; i += 0.001)
				{
					DebugLogToUnity(sFormator("float Test: [%f] sdfDictionary1D Value [%f] sdfDictionary1D Key[%d]",
						i, GetSdfVectorValue(i), GetSdfVectorKey(GetSdfVectorValue(i))));
				}
			}
			//float inputSdf = 0;
			//float inputSdfAbs = fabs(inputSdf);
			//int mapSdfDictionaryKey = 0;
			//if (inputSdf >= MinVoxel && inputSdf <=0)
			//{
			//	mapSdfDictionaryKey = 0;
			//}else if (inputSdfAbs <= ClampBound + 0.002)
			//{
			//	if (inputSdf > 0)
			//		mapSdfDictionaryKey = (int)(2 * inputSdfAbs * 1000);
			//	else
			//		mapSdfDictionaryKey = (int)(2 * inputSdfAbs * 1000) + 1;
			//}
			//else
			//{
			//	if (inputSdf > 0)
			//		mapSdfDictionaryKey = (int)(2 * (inputSdfAbs - ClampBound) * 10 + ClampSize + 1);
			//	else
			//		mapSdfDictionaryKey = (int)(2 * (inputSdfAbs - ClampBound) * 10 + ClampSize + 2);
			//}
			//mapSdfDictionaryKey = clamp(mapSdfDictionaryKey, 0, DictionarySize - 1);

			noiseData1D = (ushort2*)malloc(voxelNum * sizeof(ushort2));
			for (int i = 0; i < voxelNum; i++)
				noiseData1D[i] = make_ushort2(0,0);
			
			//checkCudaErrors(cudaHostAlloc((void**)&noiseData1DTemp, n * sizeof(float), cudaHostAllocDefault));
			h_triNum = new int[1];
			z_triNum = new int[1];
			z_triNum[0] = 0;
			NullTribuffer = new Triangle[1];
			NullTribuffer[0].posA = make_Vector3(make_float3(0));
			NullTribuffer[0].posB = make_Vector3(make_float3(0));
			NullTribuffer[0].posC = make_Vector3(make_float3(0));
			NullTribuffer[0].normalA = make_Vector3(make_float3(0));
			NullTribuffer[0].normalB = make_Vector3(make_float3(0));
			NullTribuffer[0].normalC = make_Vector3(make_float3(0));
			checkCudaErrors(cudaMalloc(&d_NullTribuffer, z_triNum[0] * sizeof(Triangle)));

			gridSizeVoxel = make_uint3(gridSize.x, gridSize.y, gridSize.z) - make_uint3(2);
			checkCudaErrors(cudaMalloc(&d_triNum, sizeof(int)));
			checkCudaErrors(cudaMalloc((void**)&noiseData1DTemp, voxelNum * sizeof(ushort2)));
			//checkCudaErrors(cudaMalloc((void**)&noiseData1DTempDevice, voxelNum * sizeof(float)));
			//checkCudaErrors(cudaMalloc((void**)&meshSampleData1DTemp, voxelNum * sizeof(float)));
			//checkCudaErrors(cudaMalloc((void**)&meshSampleData1DTempDevice, voxelNum * sizeof(float)));
			checkCudaErrors(cudaMemcpyAsync(noiseData1DTemp, noiseData1D, voxelNum * sizeof(ushort2), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			//checkCudaErrors(cudaMemcpyAsync(meshSampleData1DTemp, noiseData1D, voxelNum * sizeof(float), cudaMemcpyHostToDevice, cudaStreams[0]));
			//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			//checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			free(noiseData1D);

			checkCudaErrors(cudaMalloc((void**)&sdfDictionary1DTemp, DictionarySize * sizeof(float2)));
			checkCudaErrors(cudaMemcpyAsync(sdfDictionary1DTemp, sdfDictionary1D, DictionarySize * sizeof(float2), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));


			checkCudaErrors(cudaMemcpyAsync(d_NullTribuffer, NullTribuffer, z_triNum[0] * sizeof(Triangle), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			//checkCudaErrors(cudaMemcpyAsync(noiseData1DTempDevice, noiseData1DTemp, voxelNum * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			//checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			//checkCudaErrors(cudaMemcpyAsync(meshSampleData1DTempDevice, noiseData1DTemp, voxelNum * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			//checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			//checkCudaErrors(cudaEventSynchronize(stop_event[0]));

#if SKIP_EMPTY_VOXELS
			voxelScan = (int*)malloc(sizeof(int) * voxelNumInChunk);
			for (int i = 0; i < voxelNumInChunk; i++)
				voxelScan[i] = -1;
			checkCudaErrors(cudaMalloc((void**)&voxelScanTemp, sizeof(int) * voxelNumInChunk));
			checkCudaErrors(cudaMemcpyAsync(voxelScanTemp, voxelScan, voxelNumInChunk * sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			compactVoxel = (int*)malloc(sizeof(int) * voxelNumInChunk);
			for (int i = 0; i < voxelNumInChunk; i++)
				compactVoxel[i] = 0;
			checkCudaErrors(cudaMalloc((void**)&compactVoxelTemp, sizeof(int) * voxelNumInChunk));
			checkCudaErrors(cudaMemcpyAsync(compactVoxelTemp, compactVoxel, voxelNumInChunk * sizeof(int), cudaMemcpyHostToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));	
			free(compactVoxel);
			free(voxelScan);
#endif
			ReuseMemory();
			return true;
		}
		void ReuseMemoryInChunk(McChunk chunk)
		{
			int i = chunk.chunkID;
			int n = chunk.chunkSize.x * chunk.chunkSize.y * chunk.chunkSize.z;
#if SKIP_EMPTY_VOXELS
			checkCudaErrors(cudaMemcpyAsync(d_voxelVertsAll, voxelScanTemp, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			checkCudaErrors(cudaMemcpyAsync(d_voxelVertsScanAll, voxelScanTemp, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			checkCudaErrors(cudaMemcpyAsync(d_voxelOccupiedAll, voxelScanTemp, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			checkCudaErrors(cudaMemcpyAsync(d_voxelOccupiedScanAll, voxelScanTemp, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
			checkCudaErrors(cudaMemcpyAsync(h_compVoxelArrayAll[i], compactVoxelTemp, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreams[0]));
			checkCudaErrors(cudaEventRecord(stop_event[0], cudaStreams[0]));
			checkCudaErrors(cudaEventSynchronize(stop_event[0]));
#endif
		}
		void FreeMemoryInChunk(McChunk chunk)
		{
			//checkCudaErrors(cudaFree(chunk.d_Tribuffer));
			checkCudaErrors(cudaFree(chunk.d_triNum));
#if SKIP_EMPTY_VOXELS
			checkCudaErrors(cudaFree(chunk.d_compVoxelArray));
#endif
		}
		void ReuseMemory()
		{
			for (int i = 0; i < chunkNumber; i++)
			{
				ReuseMemoryInChunk(mcChunks[i]);
			}
		}

		bool FreeMemoryForMC()
		{
			//if (!canFree)
			//	return false;
			hasInput = false;
			canFree = false;
			//for (int i = 0; i < chunkThreads; i++)
			//{
			//	checkCudaErrors(cudaEventRecord(stop_event[i], cudaStreams[i]));
			//	checkCudaErrors(cudaEventSynchronize(stop_event[i]));
			//}
			sdkDeleteTimer(&timer);
			free(triangle_dataTemp);
			free(normal_dataTemp);
			//free(noiseData1D);
			//free(meshSampleData1D);
			//free(voxelScan);
			//free(compactVoxel);
			//free(Tribuffer);
			free(sdfDictionary1D);
			//free(octreeBboxChunks.leafNodes);
			checkCudaErrors(cudaFree(triangle_data));
			checkCudaErrors(cudaFree(normal_data));
			if (enableColorSmoothing)
			{
				brushListTemp.clear();
				checkCudaErrors(cudaFree(brushVoxels));
				cudaFree(brushList);
				free(brushVoxelsTemp);
			}

			//checkCudaErrors(cudaFree(meshSampleData1DTemp));
			//checkCudaErrors(cudaFree(meshSampleData1DTempDevice));
			//checkCudaErrors(cudaFree(noiseData1DTempDevice));
			checkCudaErrors(cudaFree(noiseData1DTemp));
			checkCudaErrors(cudaFree(sdfDictionary1DTemp));
			checkCudaErrors(cudaFree(d_NullTribuffer));
			checkCudaErrors(cudaFree(d_Tribuffer));
			checkCudaErrors(cudaFree(d_triNum));
#if SKIP_EMPTY_VOXELS
			checkCudaErrors(cudaFree(voxelScanTemp));
			checkCudaErrors(cudaFree(compactVoxelTemp));
			checkCudaErrors(cudaFree(d_voxelVertsAll));
			checkCudaErrors(cudaFree(d_voxelVertsScanAll));
			checkCudaErrors(cudaFree(d_voxelOccupiedAll));
			checkCudaErrors(cudaFree(d_voxelOccupiedScanAll));
#endif
			for (int i = 0; i < chunkNumber; i++)
			{
				FreeMemoryInChunk(mcChunks[i]);
			}
			//for (int i = 0; i < chunkThreads; i++)
			//{
			//	checkCudaErrors(cudaStreamDestroy(cudaStreams[i]));
			//	checkCudaErrors(cudaEventDestroy(stop_event[i]));
			//	checkCudaErrors(cudaEventDestroy(kernelEvent[i]));
			//}
			return true;
		}
		//Class Destructor
		~MarchingCubesChunk()
		{
			//FreeMemoryForMC();
		};
	};
}