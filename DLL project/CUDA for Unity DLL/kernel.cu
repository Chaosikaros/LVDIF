#include "device_functions.h"
#include "driver_functions.h"
#include "device_launch_parameters.h"
#include "CudaUnity.h"
#include "mcTables.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace std;
using namespace CudaUnity;

namespace CudaUnity
{
    int block_size = 256;

	void SetMaxBlockSize(int input)
	{
		block_size = input;
	}

    // only works for power of 2 sizes
	__device__ uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
	{
		uint3 gridPos;
		gridPos.x = i & gridSizeMask.x;
		gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
		gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
		return gridPos;
	}

	__device__ uint3 calcGridPosLayerX(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
	{
		uint3 gridPos;
		gridPos.x = 0;
		gridPos.y = i & gridSizeMask.x;
		gridPos.z = (i >> gridSizeShift.y) & gridSizeMask.y;
		return gridPos;
	}

	__device__ uint3 calcGridPosLayerY(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
	{
		uint3 gridPos;
		gridPos.x = i & gridSizeMask.x;
		gridPos.y = 0;
		gridPos.z = (i >> gridSizeShift.y) & gridSizeMask.y;
		return gridPos;
	}

	__device__ uint3 calcGridPosLayerZ(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
	{
		uint3 gridPos;
		gridPos.x = i & gridSizeMask.x;
		gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
		gridPos.z = 0;
		return gridPos;
	}
	__device__ bool volumeIDCheckBound(uint3 p, uint3 gridSize)
	{
		if (p.x <= 0 || p.y <= 0 || p.z <= 0)
			return false;
		if (p.x >= gridSize.x || p.y >= gridSize.y || p.z >= gridSize.z)
			return false;
		return true;
	}
	__device__ int sampleVolumeID(uint3 p, uint3 gridSize)
	{
		p.x = min(p.x, gridSize.x);
		p.y = min(p.y, gridSize.y);
		p.z = min(p.z, gridSize.z);
		int i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
		//    return (float) data[i] / 255.0f;
		return i;
	}

	__device__ float sampleVolumeColor(float2* sdfDictionary, ushort2* data, uint3 p, uint3 gridSize)
	{
		if (p.x == 0 || p.y == 0 || p.z == 0 ||
			p.x >= gridSize.x - 2 || p.y >= gridSize.y - 2 || p.z >= gridSize.z - 2)
			return 0;
		p.x = min(p.x, gridSize.x);
		p.y = min(p.y, gridSize.y);
		p.z = min(p.z, gridSize.z);
		uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
		//    return (float) data[i] / 255.0f;
		return sdfDictionary[data[i].y].y;
	}

	__device__ float sampleVolume(float2* sdfDictionary, ushort2* data, uint3 p, uint3 gridSize)
	{
		if (p.x == 0 || p.y == 0 || p.z == 0 ||
			p.x == gridSize.x || p.y == gridSize.y || p.z == gridSize.z)
			return MinVoxel;
		p.x = min(p.x, gridSize.x);
		p.y = min(p.y, gridSize.y);
		p.z = min(p.z, gridSize.z);
		uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
		//    return (float) data[i] / 255.0f;
		return sdfDictionary[data[i].x].x;
	}

	__device__ __host__ inline float3 TexelToPosition(int x, int y, int z, uint3 adaptiveMapMin, uint3 adaptiveMapMax, float3 _MinBounds, float3 _MaxBounds)
	{
		float xCoord = remap(adaptiveMapMin.x, adaptiveMapMax.x, _MinBounds.x, _MaxBounds.x, x);
		float yCoord = remap(adaptiveMapMin.y, adaptiveMapMax.y, _MinBounds.y, _MaxBounds.y, y);
		float zCoord = remap(adaptiveMapMin.z, adaptiveMapMax.z, _MinBounds.z, _MaxBounds.z, z);
		return make_float3(xCoord, yCoord, zCoord);
	}

	__device__ __host__ inline bool RayIntersectsTriangle(float3 o, float3 d, float3 v0, float3 v1, float3 v2)
	{
		const float EPSILON = 0.0000001;

		float3 e1, e2, h, s, q;
		float a, f, u, v, t;
		e1 = v1 - v0;
		e2 = v2 - v0;

		h = cross(d, e2);
		a = dot(e1, h);

		if (abs(a) < EPSILON)
		{
			return false; // ray is parallel to triangle
		}

		f = 1.0 / a;
		s = o - v0;
		u = f * dot(s, h);

		if (u < 0.0 || u > 1.0)
			return false;

		q = cross(s, e1);
		v = f * dot(d, q);

		if (v < 0.0 || u + v > 1.0)
			return false;

		t = f * dot(e2, q);

		if (t >= 0.0)
		{
			return true;
		}
		return false;
	}

	__device__ __host__ inline float3 RayIntersectsPointTriangle(float3 o, float3 d, float3 v0, float3 v1, float3 v2)
	{
		const float EPSILON = 0.0000001;

		float3 e1, e2, h, s, q;
		float a, f, u, v, t;

		e1 = v1 - v0;
		e2 = v2 - v0;

		h = cross(d, e2);
		a = dot(e1, h);

		if (abs(a) < EPSILON)
		{
			return make_float3(0); // ray is parallel to triangle
		}

		f = 1.0 / a;
		s = o - v0;
		u = f * dot(s, h);

		if (u < 0.0 || u > 1.0)
			return make_float3(0);

		q = cross(s, e1);
		v = f * dot(d, q);

		if (v < 0.0 || u + v > 1.0)
			return make_float3(0);

		t = f * dot(e2, q);

		if (t >= 0.0)
		{
			return  o + d * t;;
		}
		return make_float3(0);
	}

	__device__ __host__ inline float DistanceToTriangle(float3 p, float3 a, float3 b, float3 c)
	{
		float3 ba = b - a;
		float3 pa = p - a;
		float3 cb = c - b;
		float3 pb = p - b;
		float3 ac = a - c;
		float3 pc = p - c;

		float3 nor = cross(ba, ac);

		return sqrt(
			(sign(dot(cross(ba, nor), pa)) +
				sign(dot(cross(cb, nor), pb)) +
				sign(dot(cross(ac, nor), pc)) < 2.0)
			?
			min(min(
				dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
				dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
				dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc))
			:
			dot(nor, pa) * dot(nor, pa) / dot2(nor));
	}
	__device__ __host__ inline float3 ClosestPointOnTriangle(float3 sourcePosition, float3 aT, float3 bT, float3 cT)
	{
		float3 edge0 = bT - aT;
		float3 edge1 = cT - aT;
		float3 v0 = aT - sourcePosition;

		float a = dot(edge0,edge0);
		float b = dot(edge0, edge1);
		float c = dot(edge1, edge1);
		float d = dot(edge0, v0);
		float e = dot(edge1, v0);

		float det = a * c - b * b;
		float s = b * e - c * d;
		float t = b * d - a * e;

		if (s + t < det)
		{
			if (s < 0.f)
			{
				if (t < 0.f)
				{
					if (d < 0.f)
					{
						s = clamp(-d / a, 0.f, 1.f);
						t = 0.f;
					}
					else
					{
						s = 0.f;
						t = clamp(-e / c, 0.f, 1.f);
					}
				}
				else
				{
					s = 0.f;
					t = clamp(-e / c, 0.f, 1.f);
				}
			}
			else if (t < 0.f)
			{
				s = clamp(-d / a, 0.f, 1.f);
				t = 0.f;
			}
			else
			{
				float invDet = 1.f / det;
				s *= invDet;
				t *= invDet;
			}
		}
		else
		{
			if (s < 0.f)
			{
				float tmp0 = b + d;
				float tmp1 = c + e;
				if (tmp1 > tmp0)
				{
					float numer = tmp1 - tmp0;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					t = clamp(-e / c, 0.f, 1.f);
					s = 0.f;
				}
			}
			else if (t < 0.f)
			{
				if (a + d > b + e)
				{
					float numer = c + e - b - d;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					s = clamp(-e / c, 0.f, 1.f);
					t = 0.f;
				}
			}
			else
			{
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1.f - s;
			}
		}

		return aT + s * edge0 + t * edge1;
	}
	__device__ int PlaneBoxOverlap(float3 normal, float3 vert, float3 maxbox)
	{
		float  v;
		float3 vmin, vmax;
		v = vert.x;
		if (normal.x > 0.0f)
		{
			vmin.x = -maxbox.x - v;
			vmax.x = maxbox.x - v;
		}
		else
		{
			vmin.x = maxbox.x - v;
			vmax.x = -maxbox.x - v;
		}
		v = vert.y;
		if (normal.y > 0.0f)
		{
			vmin.y = -maxbox.y - v;
			vmax.y = maxbox.y - v;
		}
		else
		{
			vmin.y = maxbox.y - v;
			vmax.y = -maxbox.y - v;
		}
		v = vert.z;
		if (normal.z > 0.0f)
		{
			vmin.z = -maxbox.z - v;
			vmax.z = maxbox.z - v;
		}
		else
		{
			vmin.z = maxbox.z - v;
			vmax.z = -maxbox.z - v;
		}

		if (dot(normal, vmin) > 0.0f) return 0;

		if (dot(normal, vmax) >= 0.0f) return 1;

		return 0;
	}

	__device__ bool TriBoxOverlap(float3 boxcenter, float3 boxhalfsize, float3 aT, float3 bT, float3 cT)
	{

		float3 v0, v1, v2;
		float minV, maxV, p0, p1, p2, rad, fex, fey, fez;
		float3 normal, e0, e1, e2;
		float a, b, fa, fb;
		v0 = aT - boxcenter;
		v1 = bT - boxcenter;
		v2 = cT - boxcenter;

		e0 = v1 - v0;
		e1 = v2 - v1;
		e2 = v0 - v2;

		fex = fabsf(e0.x); fey = fabsf(e0.y); fez = fabsf(e0.z);

		a = e0.z; b = e0.y; fa = fez; fb = fey;
		p0 = a * v0.y - b * v0.z;
		p2 = a * v2.y - b * v2.z;
		if (p0 < p2) { minV = p0; maxV = p2; }
		else { minV = p2; maxV = p0; }
		rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e0.z; b = e0.x; fa = fez; fb = fex;
		p0 = -a * v0.x + b * v0.z;
		p2 = -a * v2.x + b * v2.z;
		if (p0 < p2) { minV = p0; maxV = p2; }
		else { minV = p2; maxV = p0; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e0.y; b = e0.x; fa = fey; fb = fex;
		p1 = a * v1.x - b * v1.y;
		p2 = a * v2.x - b * v2.y;
		if (p2 < p1) { minV = p2; maxV = p1; }
		else { minV = p1; maxV = p2; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
		if (minV > rad || maxV < -rad) return 0;

		fex = fabsf(e1.x); fey = fabsf(e1.y); fez = fabsf(e1.z);

		a = e1.z; b = e1.y; fa = fez; fb = fey;
		p0 = a * v0.y - b * v0.z;
		p2 = a * v2.y - b * v2.z;
		if (p0 < p2) { minV = p0; maxV = p2; }
		else { minV = p2; maxV = p0; }
		rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e1.z; b = e1.x; fa = fez; fb = fex;
		p0 = -a * v0.x + b * v0.z;
		p2 = -a * v2.x + b * v2.z;
		if (p0 < p2) { minV = p0; maxV = p2; }
		else { minV = p2; maxV = p0; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e1.y; b = e1.x; fa = fey; fb = fex;
		p0 = a * v0.x - b * v0.y;
		p1 = a * v1.x - b * v1.y;
		if (p0 < p1) { minV = p0; maxV = p1; }
		else { minV = p1; maxV = p0; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
		if (minV > rad || maxV < -rad) return 0;

		fex = fabsf(e2.x); fey = fabsf(e2.y); fez = fabsf(e2.z);

		a = e2.z; b = e2.y; fa = fez; fb = fey;
		p0 = a * v0.y - b * v0.z;
		p1 = a * v1.y - b * v1.z;
		if (p0 < p1) { minV = p0; maxV = p1; }
		else { minV = p1; maxV = p0; }
		rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e2.z; b = e2.x; fa = fez; fb = fex;
		p0 = -a * v0.x + b * v0.z;
		p1 = -a * v1.x + b * v1.z;
		if (p0 < p1) { minV = p0; maxV = p1; }
		else { minV = p1; maxV = p0; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
		if (minV > rad || maxV < -rad) return 0;

		a = e2.y; b = e2.x; fa = fey; fb = fex;
		p1 = a * v1.x - b * v1.y;
		p2 = a * v2.x - b * v2.y;
		if (p2 < p1) { minV = p2; maxV = p1; }
		else { minV = p1; maxV = p2; }
		rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
		if (minV > rad || maxV < -rad) return 0;

		minV = min(v0.x, min(v1.x, v2.x));
		maxV = max(v0.x, max(v1.x, v2.x));
		if (minV > boxhalfsize.x || maxV < -boxhalfsize.x) return 0;

		minV = min(v0.y, min(v1.y, v2.y));
		maxV = max(v0.y, max(v1.y, v2.y));
		if (minV > boxhalfsize.y || maxV < -boxhalfsize.y) return 0;

		minV = min(v0.z, min(v1.z, v2.z));
		maxV = max(v0.z, max(v1.z, v2.z));
		if (minV > boxhalfsize.z || maxV < -boxhalfsize.z) return 0;

		normal = cross(e0, e1);
		if (!PlaneBoxOverlap(normal, v0, boxhalfsize)) return 0;

		return 1;

	}
	__global__ void sampleOctreeBbox(uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax,
		int n_triangles, float* triangle_data, float* normal_data, OctreeBbox* octreeBboxes, float3 bboxChunkSize, float octreeBboxSizeOffset, int* boxNum, bool calculateTri) {

		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
		int x = gridPos.x;
		int y = gridPos.y;
		int z = gridPos.z;
		int idBox = atomicAdd(boxNum, 1) - 1;
		idBox = clamp(idBox,0, octreeBboxes[0].boxCount - 1);
		//float3 bboxPos = make_float3(bboxChunkSize.x * x,
		//	bboxChunkSize.y * y, bboxChunkSize.z * z);
		//octreeBboxes[idBox].octreeLeafID = idBox;
		//octreeBboxes[idBox].bboxMax = bboxMin + bboxPos + bboxChunkSize;
		//octreeBboxes[idBox].bboxMin = bboxMin + bboxPos;
		//octreeBboxes[idBox].bboxCenter = bboxMin + bboxPos + 0.5f * bboxChunkSize;
		//octreeBboxes[idBox].bboxHalfsize = 0.5f * bboxChunkSize;
		float3 a, b, c;
		//float3 aP, bP, cP, aA, bA, cA, d, e, f;
		//float aE, bE, cE, aEP, bEP, cEP, aEA, bEA, cEA;
		int triNum = 0;
		//bool checkTriangle;
		//bool checkOrphanedTriangle;
		//bool checkNormal;
		for (int i = 0; i < n_triangles; i += 9)
		{
			a = make_float3(triangle_data[clamp(i, 0, n_triangles - 1)], triangle_data[clamp(i + 1, 0, n_triangles - 1)], triangle_data[clamp(i + 2, 0, n_triangles - 1)]);
			b = make_float3(triangle_data[clamp(i + 3, 0, n_triangles - 1)], triangle_data[clamp(i + 4, 0, n_triangles - 1)], triangle_data[clamp(i + 5, 0, n_triangles - 1)]);
			c = make_float3(triangle_data[clamp(i + 6, 0, n_triangles - 1)], triangle_data[clamp(i + 7, 0, n_triangles - 1)], triangle_data[clamp(i + 8, 0, n_triangles - 1)]);

			//d = make_float3(normal_data[clamp(i, 0, n_triangles - 1)], normal_data[clamp(i + 1, 0, n_triangles - 1)], normal_data[clamp(i + 2, 0, n_triangles - 1)]);
			//e = make_float3(normal_data[clamp(i + 3, 0, n_triangles - 1)], normal_data[clamp(i + 4, 0, n_triangles - 1)], normal_data[clamp(i + 5, 0, n_triangles - 1)]);
			//f = make_float3(normal_data[clamp(i + 6, 0, n_triangles - 1)], normal_data[clamp(i + 7, 0, n_triangles - 1)], normal_data[clamp(i + 8, 0, n_triangles - 1)]);
			//aE = length(b - c);
			//bE = length(a - c);
			//cE = length(a - b);
			//checkTriangle = (aE != 0 && bE != 0 && cE != 0)&& length(cross(b - a, c- a)) != 0;
			////remove edges with no length and triangles with no area
			//checkOrphanedTriangle = false;
			//if (i >= 9 && i < n_triangles - 9)
			//{
			//	aP = make_float3(triangle_data[clamp(i - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 1 - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 2 - 9, 0, n_triangles - 1)]);
			//	bP = make_float3(triangle_data[clamp(i + 3 - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 4 - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 5 - 9, 0, n_triangles - 1)]);
			//	cP = make_float3(triangle_data[clamp(i + 6 - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 7 - 9, 0, n_triangles - 1)], triangle_data[clamp(i + 8 - 9, 0, n_triangles - 1)]);

			//	aA = make_float3(triangle_data[clamp(i + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 1 + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 2 + 9, 0, n_triangles - 1)]);
			//	bA = make_float3(triangle_data[clamp(i + 3 + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 4 + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 5 + 9, 0, n_triangles - 1)]);
			//	cA = make_float3(triangle_data[clamp(i + 6 + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 7 + 9, 0, n_triangles - 1)], triangle_data[clamp(i + 8 + 9, 0, n_triangles - 1)]);

			//	aEP = length(bP - cP);
			//	bEP = length(aP - cP);
			//	cEP = length(aP - bP);

			//	aEA = length(bA - cA);
			//	bEA = length(aA - cA);
			//	cEA = length(aA - bA);

			//	checkOrphanedTriangle = aE != aEA && aE != bEA && aE != cEA && aE != aEP && aE != bEP && aE != cEP
			//		&& bE != aEA && bE != bEA && bE != cEA && bE != aEP && bE != bEP && bE != cEP
			//		&& cE != aEA && cE != bEA && cE != cEA && cE != aEP && cE != bEP && cE != cEP;
			//	//remove orphaned triangles 
			//}

			//checkNormal = (length(d - e) != 0 && length(d - f) != 0 && length(e - f) != 0)
			//	&& length(cross(e - d, f - d)) != 0;
			if (TriBoxOverlap(octreeBboxes[idBox].bboxCenter, octreeBboxes[idBox].bboxHalfsize * octreeBboxSizeOffset, a, b, c))
			{	
				octreeBboxes[idBox].empty = false;
				if (!calculateTri)
					octreeBboxes[idBox].triangleList[triNum] = i;
				triNum++;
				if (calculateTri)
					octreeBboxes[idBox].triangleCount = triNum;
			}
		}
	}

	void launch_sampleOctreeBbox(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax,
		int n_triangles, float* triangle_data, float* normal_data, OctreeBbox* octreeBboxes, float3 bboxChunkSize, float octreeBboxSizeOffset, int* triNum, bool calculateTri)
	{
		sampleOctreeBbox << <grid, threads, 0, cudaStreams >> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax,
			n_triangles, triangle_data, normal_data, octreeBboxes, bboxChunkSize, octreeBboxSizeOffset, triNum, calculateTri);
		getLastCudaError("Check sampleOctreeBbox Kernel");
	}
	__global__ void sampleTriangleWindingNumber(uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize) {

		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
		if (useChunk)
		{
			gridPos.x += gridSize.x * chunkPos.x;
			gridPos.y += gridSize.y * chunkPos.y;
			gridPos.z += gridSize.z * chunkPos.z;
		}
		int x = gridPos.x;
		int y = gridPos.y;
		int z = gridPos.z;

		if (gridPos.x <= 1 || gridPos.y <= 1 || gridPos.z <= 1)
			return;
		if (gridPos.x >= realGridSize.x - 1 || gridPos.y >= realGridSize.y - 1 || gridPos.z >= realGridSize.z - 1)
			return;

		float3 p = TexelToPosition(x, y, z, adaptiveMapMin, adaptiveMapMax, bboxMin, bboxMax);
		float minDistance = 1000000.0;
		float dist;
		int intersectsID = 0;
		float sdfRange = 1.73205 * realGridSize.x;
		float bboxRange = length(bboxMin - bboxMax);
		float windingNumber = 0;
		float3 a, b, c, A, B, C, intersectsDirectionTemp;
		float aL, bL, cL, determinant, atan2X;
		float PI = 3.141592653589793238462643383279502884;
		for (int i = 0; i < n_triangles; i += 9)
		{
			a = make_float3(triangle_data[clamp(i, 0, n_triangles - 1)], triangle_data[clamp(i + 1, 0, n_triangles - 1)], triangle_data[clamp(i + 2, 0, n_triangles - 1)]);
			b = make_float3(triangle_data[clamp(i + 3, 0, n_triangles - 1)], triangle_data[clamp(i + 4, 0, n_triangles - 1)], triangle_data[clamp(i + 5, 0, n_triangles - 1)]);
			c = make_float3(triangle_data[clamp(i + 6, 0, n_triangles - 1)], triangle_data[clamp(i + 7, 0, n_triangles - 1)], triangle_data[clamp(i + 8, 0, n_triangles - 1)]);
			intersectsDirectionTemp = ClosestPointOnTriangle(p, a, b, c) - p;
			dist = remap(-bboxRange, bboxRange, -sdfRange, sdfRange, length(intersectsDirectionTemp));
			dist = GetSdfVectorValue(dist);
			if (dist < minDistance)
			{
				minDistance = dist;
			}
			A = a - p;
			B = b - p;
			C = c - p;
			aL = length(A);
			bL = length(B);
			cL = length(C);
			determinant = A.x * B.y * C.z + A.y * B.z * C.x + A.z * B.x * C.y -
				A.z * B.y * C.x - A.y * B.x * C.z - A.x * B.z * C.y;
			atan2X = (aL * bL * cL) + cL * dot(A, B) + aL * dot(B, C) + bL * dot(C, A);
			windingNumber += atan2(determinant, atan2X);
		}
		if(windingNumber < 1.9 * PI)
			minDistance *= -1;
		noiseData[(x * realGridSize.y + y) * realGridSize.z + z].x = GetSdfVectorKey(minDistance);
	}

	void launch_sampleTriangleWindingNumber(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize)
	{
		sampleTriangleWindingNumber << <grid, threads, 0, cudaStreams >> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax, n_triangles,
			triangle_data, normal_data, sdfDictionary, noiseData, realGridSize, chunkPos, useChunk, octreeBboxes, octreeSize);
		getLastCudaError("Check sampleTriangleWindingNumber Kernel");
	}

	void launch_sampleTriangleThreadWindingNumber(dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize)
	{
		sampleTriangleWindingNumber << <grid, threads >> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax, n_triangles,
			triangle_data, normal_data, sdfDictionary, noiseData, realGridSize, chunkPos, useChunk, octreeBboxes, octreeSize);
		getLastCudaError("Check sampleTriangleWindingNumber Kernel");
	}
	__global__ void sampleTriangleFilter(uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes,
		int octreeSize, float filterValue, int* fliterCounter, int currentLayer, int filterMode) {

		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
		
		uint3 gridPos = calcGridPosLayerX(i, gridSizeShift, gridSizeMask);
		int x = currentLayer;
		int yL = gridPos.y;
		int zL = gridPos.z;
		float dC = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL, zL), realGridSize);
		bool reverse = false;
		if (filterMode == 3)
		{
			//if (abs(dC) <= filterValue)
			//{
				int signSum = 0;
				float dYp = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL + 1, zL), realGridSize);
				if (sign(dC) != sign(dYp))
					signSum++;
				float dYn = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL - 1, zL), realGridSize);
				if (sign(dC) != sign(dYn))
					signSum++;
				float dZp = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL, zL + 1), realGridSize);
				if (sign(dC) != sign(dZp))
					signSum++;
				float dZn = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL, zL - 1), realGridSize);
				if (sign(dC) != sign(dZn))
					signSum++;
				float dXp = sampleVolume(sdfDictionary, noiseData, make_uint3(x + 1, yL, zL), realGridSize);
				if (sign(dC) != sign(dXp))
					signSum++;
				float dXn = sampleVolume(sdfDictionary, noiseData, make_uint3(x - 1, yL, zL), realGridSize);
				if (sign(dC) != sign(dXn))
					signSum++;
				if (signSum >= 4)
					reverse = true;
			//}
		}
		else if (abs(dC) > filterValue)
		{
			if (filterMode == 0)
			{
				float dCP = sampleVolume(sdfDictionary, noiseData, make_uint3(x - 1, yL, zL), realGridSize);
				if (sign(dC) != sign(dCP))
					reverse = true;
			}
			else if (filterMode == 1)
			{
				int filterRange1 = 1;
				int filterRange2 = 1;
				if (abs(dC) > filterValue)
				{
					int signSum = 0;
					for (int yi = -filterRange1; yi <= filterRange1; yi++)
					{
						for (int zi = -filterRange1; zi <= filterRange1; zi++)
						{
							float sampleP = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL + yi, zL + zi), realGridSize);
							signSum += sign(sampleP);
						}
					}
					if(sign(signSum) != sign(dC))
						for (int yi = -filterRange2; yi <= filterRange2 && !reverse; yi++)
						{
							for (int zi = -filterRange2; zi <= filterRange2 && !reverse; zi++)
							{
								if (yi == 0 && zi == 0)
									continue;
								float sampleP = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL + yi, zL + zi), realGridSize);
								if (sign(sampleP) != sign(dC) && abs(sampleP) > abs(dC))
									reverse = true;
							}
						}
				}
			}
			else if (filterMode == 2)
			{
				float dYp = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL + 1, zL), realGridSize);
				if (sign(dC) != sign(dYp))
				{
					float dYn = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL - 1, zL), realGridSize);
					if (sign(dC) != sign(dYn))
					{
						float dZp = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL, zL + 1), realGridSize);
						if (sign(dC) != sign(dZp))
						{
							float dZn = sampleVolume(sdfDictionary, noiseData, make_uint3(x, yL, zL - 1), realGridSize);
							if (sign(dC) != sign(dZn))
								reverse = true;
						}
					}
				}
			}
		}
		if (reverse)
		{
			noiseData[zL * realGridSize.x * realGridSize.y + yL * realGridSize.x + x].x = GetSdfVectorKey(GetSdfVectorValue(-dC));
			//noiseData[(x * realGridSize.y + yL) * realGridSize.z + zL].x = GetSdfVectorKey(GetSdfVectorValue(-dC)); //Wrong index (different with sampleVolume)
			if(filterMode != 3)
				atomicAdd(fliterCounter, 1);
		}
	}

	void launch_sampleTriangleFilter(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize, float filterValue, int* fliterCounter, int currentLayer, int filterMode)
	{
		sampleTriangleFilter << <grid, threads, 0, cudaStreams >> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax, n_triangles,
			triangle_data, normal_data, sdfDictionary, noiseData, realGridSize, chunkPos, useChunk, octreeBboxes, octreeSize, filterValue, fliterCounter, currentLayer, filterMode);
		getLastCudaError("Check sampleTriangleFilter Kernel");
	}

	__global__ void sampleTriangle(uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax, 
		int n_triangles, float* triangle_data, float* normal_data,  float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize) {
		
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
		if (useChunk)
		{
			gridPos.x += gridSize.x * chunkPos.x;
			gridPos.y += gridSize.y * chunkPos.y;
			gridPos.z += gridSize.z * chunkPos.z;
		}
		int x = gridPos.x;
		int y = gridPos.y;
		int z = gridPos.z;

		if (gridPos.x <= 1 || gridPos.y <= 1 || gridPos.z <= 1)
			return;
		if (gridPos.x >= realGridSize.x - 1 || gridPos.y >= realGridSize.y - 1 || gridPos.z >= realGridSize.z - 1)
			return;
	
		float3 p = TexelToPosition(x, y, z, adaptiveMapMin, adaptiveMapMax, bboxMin, bboxMax);

		float minOctreeDis = 100000;
		int octreeLeafID = 0;

		for (int octreeVoxelID = 0; octreeVoxelID < octreeSize; octreeVoxelID++)
		{
			if (octreeBboxes[octreeVoxelID].octreeLeafID == -1)
				break;
			if (!octreeBboxes[octreeVoxelID].empty)
			{
				float tempDist = length(p - octreeBboxes[octreeVoxelID].bboxCenter);
				if (tempDist < minOctreeDis)
				{
					minOctreeDis = tempDist;
					octreeLeafID = octreeBboxes[octreeVoxelID].octreeLeafID;
				}
			}
		}
		float minDistance = 1000000.0;
		float dist;
		int intersectsID = 0;
		float sdfRange =  1.73205 * realGridSize.x;
		float bboxRange = length(bboxMin - bboxMax);
		float3 intersectsDirection, a, b, c, A, B, C, faceNormal, intersectsDirectionTemp;
		//float windingNumber = 0;
		//float aL, bL, cL, determinant, atan2X;
		//float PI = 3.141592653589793238462643383279502884;
		for (uint i = 0; i < n_triangles/9; i++)
		{
			if (octreeLeafID == -1 || octreeLeafID > octreeBboxes[0].boxCount - 1)
				octreeLeafID = 0;
			//if (i >= octreeBboxes[octreeLeafID].triangleCount)
			//	break;
			int j = octreeBboxes[octreeLeafID].triangleList[i];
			if (j == -1)
				break;
			a = make_float3(triangle_data[clamp(j, 0, n_triangles-1)], triangle_data[clamp(j + 1, 0, n_triangles - 1)], triangle_data[clamp(j + 2, 0, n_triangles - 1)]);
			b = make_float3(triangle_data[clamp(j + 3, 0, n_triangles - 1)], triangle_data[clamp(j + 4, 0, n_triangles - 1)], triangle_data[clamp(j + 5, 0, n_triangles - 1)]);
			c = make_float3(triangle_data[clamp(j + 6, 0, n_triangles - 1)], triangle_data[clamp(j + 7, 0, n_triangles - 1)], triangle_data[clamp(j + 8, 0, n_triangles - 1)]);
			intersectsDirectionTemp = ClosestPointOnTriangle(p, a, b, c) - p;
			//dist = length(intersectsDirectionTemp);
		    dist = remap(-bboxRange, bboxRange,-sdfRange, sdfRange, length(intersectsDirectionTemp));
			dist = GetSdfVectorValue(dist);
			if (dist < minDistance)
			{
				minDistance = dist;
				intersectsDirection = intersectsDirectionTemp;
				intersectsID = j;
			}
		}
		a = make_float3(normal_data[clamp(intersectsID, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 1, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 2, 0, n_triangles - 1)]);
		b = make_float3(normal_data[clamp(intersectsID + 3, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 4, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 5, 0, n_triangles - 1)]);
		c = make_float3(normal_data[clamp(intersectsID + 6, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 7, 0, n_triangles - 1)], normal_data[clamp(intersectsID + 8, 0, n_triangles - 1)]);
		faceNormal = (a + b + c) / 3;
		if (dot(intersectsDirection, faceNormal) < 0)
		{
			minDistance *= -1;
		}
		noiseData[(x * realGridSize.y + y) * realGridSize.z + z].x = GetSdfVectorKey(minDistance);
	}

	void launch_sampleTriangle(cudaStream_t cudaStreams, dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize)
	{
		sampleTriangle << <grid, threads, 0, cudaStreams >> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax, n_triangles, 
			triangle_data, normal_data, sdfDictionary, noiseData, realGridSize, chunkPos,useChunk, octreeBboxes, octreeSize);
		getLastCudaError("Check sampleTriangle Kernel");
	}

	void launch_sampleTriangleThread(dim3 grid, dim3 threads, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 bboxMin, float3 bboxMax, uint3 adaptiveMapMin, uint3 adaptiveMapMax,
		int n_triangles, float* triangle_data, float* normal_data, float2* sdfDictionary, ushort2* noiseData, uint3 realGridSize, uint3 chunkPos, bool useChunk, OctreeBbox* octreeBboxes, int octreeSize)
	{
		sampleTriangle << <grid, threads>> > (gridSize, gridSizeShift, gridSizeMask, bboxMin, bboxMax, adaptiveMapMin, adaptiveMapMax, n_triangles,
			triangle_data, normal_data, sdfDictionary, noiseData, realGridSize, chunkPos, useChunk, octreeBboxes, octreeSize);
		getLastCudaError("Check sampleTriangle Kernel");
	}

	inline __host__ __device__ float opUnion(float d1, float d2)
	{
		return min(d1, d2);
	}

	inline __host__ __device__ float opSubtraction(float d1, float d2)
	{
		return max(-d1, d2);
	}

	inline __host__ __device__ float opIntersection(float d1, float d2)
	{
		return max(d1, d2);
	}

	inline __host__ __device__ float opSmoothUnion(float d1, float d2, float k)
	{
		float h = max(k - abs(d1 - d2), 0.0);
		return min(d1, d2) - h * h * 0.25 / k;
	}

	inline __host__ __device__ float opSmoothSubtraction(float d1, float d2, float k)
	{
		float h = max(k - abs(-d1 - d2), 0.0);
		return max(-d1, d2) + h * h * 0.25 / k;
	}

	inline __host__ __device__ float opSmoothIntersection(float d1, float d2, float k)
	{
		float h = max(k - abs(d1 - d2), 0.0);
		return max(d1, d2) + h * h * 0.25 / k;
	}

	__global__ void insertShapeToVoxel(float2* sdfDictionary, ushort2* noiseData, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, int inputRadius,
		 BrushInfo* inputPos, BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing, int inputPosCount, int brushColorType, float colorBrushOffset, bool eraserMode, int brushShape, uint3 realGridSize, uint3 chunkPos, bool useChunk)
	{
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
		if (useChunk)
		{
			gridPos.x += gridSize.x * chunkPos.x;
			gridPos.y += gridSize.y * chunkPos.y;
			gridPos.z += gridSize.z * chunkPos.z;
		}
		if (gridPos.x <= 1 || gridPos.y <= 1 || gridPos.z <= 1)
			return;
		if (gridPos.x >= realGridSize.x - 1 || gridPos.y >= realGridSize.y - 1 || gridPos.z >= realGridSize.z - 1)
			return;
		int z = gridPos.x;
		int y = gridPos.y;
		int x = gridPos.z;
		int id = (x * realGridSize.y + y) * realGridSize.z + z;
		float halfS = realGridSize.x / 2;
		//float halfR = 0.45f * inputRadius;
		//float halfR = inputRadius;
		float checkLength;
		float brushDistance = 0;

		float checkLengthLast;
		float brushDistanceLast = 0;
		//brushVoxel[id].brushID = 0;
		for (int j = 0; j < inputPosCount; j++)
		{
			if (inputPos[j].shape.w > 0)
			{
				if (enableColorSmoothing)
				{
					if (brushVoxel[id].brushID == 999999999)
					{
						brushVoxel[id].brushID = inputPos[j].brushID;
						brushVoxel[id].lastBrushID = inputPos[j].brushID;
						brushVoxel[id].backgroundColor = 0;
						brushVoxel[id].lastBackgroundColor = 0;
					}
					brushDistanceLast = length(make_float3(x, y, z) - make_float3(brushList[brushVoxel[id].lastBrushID].shape.x,
						brushList[brushVoxel[id].lastBrushID].shape.y, brushList[brushVoxel[id].lastBrushID].shape.z));

					if (brushList[brushVoxel[id].lastBrushID].type == 0)
					{
						checkLengthLast = brushDistanceLast - brushList[brushVoxel[id].lastBrushID].shape.w;
					}
					else if (brushList[brushVoxel[id].lastBrushID].type == 1)
					{
						float3 brushVector = fabs((make_float3(x, y, z) - make_float3(brushList[brushVoxel[id].lastBrushID].shape.x,
							brushList[brushVoxel[id].lastBrushID].shape.y, brushList[brushVoxel[id].lastBrushID].shape.z)));
						float3 checkVector = brushVector - make_float3(brushList[brushVoxel[id].lastBrushID].shape.w);
						checkLengthLast = length(fmaxf(checkVector, make_float3(0))) + min(max(checkVector.x, max(checkVector.y, checkVector.z)), 0.0);
					}
				}

				brushDistance = length(make_float3(x, y, z) - make_float3(inputPos[j].shape.x, inputPos[j].shape.y, inputPos[j].shape.z));
				if (inputPos[j].type == 0)
				{
					checkLength = brushDistance - inputPos[j].shape.w;
				}
				else if (inputPos[j].type == 1)
				{
					float3 brushVector = fabs((make_float3(x, y, z) - make_float3(inputPos[j].shape.x, inputPos[j].shape.y, inputPos[j].shape.z)));
					float3 checkVector = brushVector - make_float3(inputPos[j].shape.w);
					checkLength = length(fmaxf(checkVector, make_float3(0))) + min(max(checkVector.x, max(checkVector.y, checkVector.z)), 0.0);
				}

				if (eraserMode)
				{
					float dist = opUnion(sdfDictionary[noiseData[id].x].x, checkLength);
					noiseData[id].x = GetSdfVectorKey(GetSdfVectorValue(dist));
				}
				else
				{
					//if (enableColorSmoothing)
					//{
					//	if (checkLength > colorBrushOffset && checkLengthLast > colorBrushOffset)
					//	{
					//		//if (brushList[brushVoxel[id].lastBrushID].colorID == brushList[brushVoxel[id].brushID].colorID)
					//			brushVoxel[id].lastBackgroundColor = brushVoxel[id].lastBackgroundColor;
					//		//else
					//		//	brushVoxel[id].lastBackgroundColor = brushVoxel[id].backgroundColor;
					//	}
					//}

					if (checkLength <= colorBrushOffset)
					{
						if (enableColorSmoothing)
						{
							if (checkLengthLast > colorBrushOffset)
								brushVoxel[id].lastBackgroundColor = brushVoxel[id].lastBackgroundColor;
							else
								brushVoxel[id].lastBackgroundColor = brushVoxel[id].backgroundColor;

							brushVoxel[id].lastBrushID = brushVoxel[id].brushID;

							brushVoxel[id].backgroundColor = noiseData[id].y;
							brushVoxel[id].brushID = inputPos[j].brushID;
						}
						noiseData[id].y = brushColorType;
					}
					float dist = opSubtraction(checkLength, sdfDictionary[noiseData[id].x].x);;
					noiseData[id].x = GetSdfVectorKey(GetSdfVectorValue(dist));
				}
			}
		}
		//if(noiseData[(x * gridSize.y + y)* gridSize.z + z] == 0.5)
		//	if (checkLength <= inputRadius)
		//		noiseData[(x * gridSize.y + y)* gridSize.z + z] = 0;

		//if (checkLength == inputRadius)
		//	noiseData[(x * gridSize.y + y)* gridSize.z + z] = 0.5;
		//else if (checkLength < inputRadius)
		//	noiseData[(x * gridSize.y + y)* gridSize.z + z] = 0;
	}

	void launch_insertShapeToVoxel(cudaStream_t cudaStreams, dim3 grid, dim3 threads, float2* sdfDictionary, ushort2* noiseData,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, int inputRadius, BrushInfo* inputPos, BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing,
		int inputPosCount, int brushColorType, float colorBrushOffset,
		bool eraserMode, int brushShape, uint3 realGridSize, uint3 chunkPos, bool useChunk)
	{
		insertShapeToVoxel << <grid, threads, 0, cudaStreams >> > (sdfDictionary, noiseData, gridSize, gridSizeShift, gridSizeMask, 
			inputRadius, inputPos, brushList, brushVoxel, enableColorSmoothing, inputPosCount, brushColorType, colorBrushOffset, eraserMode, brushShape, realGridSize, chunkPos, useChunk);
		getLastCudaError("launch_insertShapeToVoxel");
	}
	
	__global__ void classifyVoxel(int* triNum, float2* sdfDictionary, ushort2* noiseData, int *voxelVerts, int *voxelOccupied,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,float3 voxelSize, float isoValue, uint3 realGridSize, uint3 chunkPos, bool useChunk)
	{

		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
		if (useChunk)
		{
			gridPos.x += gridSize.x * chunkPos.x;
			gridPos.y += gridSize.y * chunkPos.y;
			gridPos.z += gridSize.z * chunkPos.z;
		}
		int x = gridPos.x;
		int y = gridPos.y;
		int z = gridPos.z;

		uint3 cornerCoords[8];
		cornerCoords[0] = gridPos + make_uint3(0, 0, 0);
		cornerCoords[1] = gridPos + make_uint3(1, 0, 0);
		cornerCoords[2] = gridPos + make_uint3(1, 0, 1);
		cornerCoords[3] = gridPos + make_uint3(0, 0, 1);
		cornerCoords[4] = gridPos + make_uint3(0, 1, 0);
		cornerCoords[5] = gridPos + make_uint3(1, 1, 0);
		cornerCoords[6] = gridPos + make_uint3(1, 1, 1);
		cornerCoords[7] = gridPos + make_uint3(0, 1, 1);

		int cubeindex = 0;
		for (int l = 0; l < 8; l++)
		{
			if (sampleVolume(sdfDictionary, noiseData, cornerCoords[l], realGridSize) < isoValue)
				cubeindex |= 1 << l;
		}

		uint numVerts = numVertsTable[cubeindex];

		if (i < numVoxels)
		{
#if SKIP_EMPTY_VOXELS
			voxelVerts[i] = numVerts;
			voxelOccupied[i] = (numVerts > 0);
#endif
			if (numVerts != 0)
				atomicAdd(triNum, numVerts / 3);
		}
	}

	void launch_classifyVoxel(int* triNum, cudaStream_t cudaStreams, float2* sdfDictionary, ushort2* noiseData, dim3 grid, dim3 threads,
		int *voxelVerts, int *voxelOccupied, uint3 gridSize, uint3 gridSizeShift,
		uint3 gridSizeMask, uint numVoxels, float3 voxelSize, float isoValue, uint3 realGridSize, uint3 chunkPos, bool useChunk)
	{
		classifyVoxel << <grid, threads, 0, cudaStreams >> > (triNum, sdfDictionary, noiseData, voxelVerts, voxelOccupied,
			gridSize, gridSizeShift, gridSizeMask, numVoxels, voxelSize, isoValue, realGridSize,chunkPos, useChunk);
		getLastCudaError("Check classifyVoxel Kernel");
	}

	__global__ void compactVoxels(int *compactedVoxelArray, int *voxelOccupied, int *voxelOccupiedScan, uint numVoxels)
	{
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		if (voxelOccupied[i] && (i < numVoxels))
		{
			compactedVoxelArray[voxelOccupiedScan[i]] = i;
		}
	}
	
	void launch_compactVoxels(cudaStream_t cudaStreams, dim3 grid, dim3 threads, int *compactedVoxelArray, int *voxelOccupied, int *voxelOccupiedScan, uint numVoxels)
	{
		compactVoxels << <grid, threads, 0, cudaStreams >> > (compactedVoxelArray, voxelOccupied,
			voxelOccupiedScan, numVoxels);
		getLastCudaError("Check compactVoxels Kernel");
	}

	__global__ void generateTriangles(int maxTri, int* triNum, float2* sdfDictionary, ushort2* noiseData, float3 CenterPos, int GridRes, int GridW,
		int *compactedVoxelArray, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, Triangle* Tribuffer, bool EnableSmooth, 
		uint3 realGridSize, uint3 chunkPos, bool useChunk, BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing, int brushListCount)
	{
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

		if (i > activeVoxels - 1)
		{
			// can't return here because of syncthreads()
			i = activeVoxels - 1;
		}
		//if (compactedVoxelArray[i] >= 0)
		//{
#if SKIP_EMPTY_VOXELS
			uint voxel = compactedVoxelArray[i];
#else
			uint voxel = i;
#endif

			uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);
			if (useChunk)
			{
				gridPos.x += gridSize.x * chunkPos.x;
				gridPos.y += gridSize.y * chunkPos.y;
				gridPos.z += gridSize.z * chunkPos.z;
			}
			int x = gridPos.x;
			int y = gridPos.y;
			int z = gridPos.z;

			//if (gridPos.x <= 0 || gridPos.y <= 0 || gridPos.z <= 0)
			//	return;
			//if (gridPos.x >= realGridSize.x - 1 || gridPos.y >= realGridSize.y - 1 || gridPos.z >= realGridSize.z - 1)
			//	return;
			uint3 cornerCoords[8];
			cornerCoords[0] = gridPos + make_uint3(0, 0, 0);
			cornerCoords[1] = gridPos + make_uint3(1, 0, 0);
			cornerCoords[2] = gridPos + make_uint3(1, 0, 1);
			cornerCoords[3] = gridPos + make_uint3(0, 0, 1);
			cornerCoords[4] = gridPos + make_uint3(0, 1, 0);
			cornerCoords[5] = gridPos + make_uint3(1, 1, 0);
			cornerCoords[6] = gridPos + make_uint3(1, 1, 1);
			cornerCoords[7] = gridPos + make_uint3(0, 1, 1);

			int cubeindex = 0;
			for (int l = 0; l < 8; l++)
			{
				if (sampleVolume(sdfDictionary, noiseData, cornerCoords[l], realGridSize) < isoValue)
					cubeindex |= 1 << l;
			}

			float3 finalPos, finalNormal, centerOffset;
			int connectIndex1, connectIndex2;
			float voxelW = (float)GridW / (float)GridRes;// GridW = 2, GridRes = volumeSize.x (if x = y = z)
			centerOffset = make_float3(gridPos.x, gridPos.y, gridPos.z) * voxelW + CenterPos - make_float3(1.0f) * (float)GridW / 2.0f;//(float)GridW / 2.0f = 1 if (GridW = 2)
			//centerOffset = make_float3(gridPos.x, gridPos.y, gridPos.z) * voxelW + CenterPos;
			float3 p1, p2;
			float3 posA, posB, posC, normalA, normalB, normalC;
			int currentTriCounter = 0;
			uint3 offsetX = make_uint3(1, 0, 0);
			uint3 offsetY = make_uint3(0, 1, 0);
			uint3 offsetZ = make_uint3(0, 0, 1);
			//float3 cubeInfo = make_float3(gridPos.x, gridPos.y, gridPos.z);
			for (int i = 0; i < 16; i += 3)
			{
				if (TriangleTable[cubeindex][i] == 255)
				{
					break;
				}

				Triangle tempTri;
				currentTriCounter += 1;

				for (int k = 0; k < 3; k++) {
					int edgeIndex = TriangleTable[cubeindex][i + k];
					int temp1 = cornerIndexAFromEdge[edgeIndex];
					int temp2 = cornerIndexBFromEdge[edgeIndex];
					uint3 coordA = cornerCoords[temp1];
					uint3 coordB = cornerCoords[temp2];

					//p1 = make_float3(coordA);
					//p2 = make_float3(coordB);

					connectIndex1 = cornerIndexAFromEdge[edgeIndex];
					connectIndex2 = cornerIndexBFromEdge[edgeIndex];

					p1.x = VertexPointDirF[connectIndex1][0];
					p1.y = VertexPointDirF[connectIndex1][1];
					p1.z = VertexPointDirF[connectIndex1][2];

					p2.x = VertexPointDirF[connectIndex2][0];
					p2.y = VertexPointDirF[connectIndex2][1];
					p2.z = VertexPointDirF[connectIndex2][2];

					p1 *= voxelW;
					p2 *= voxelW;
					p1 /= 2.0f;
					p2 /= 2.0f;

					float densityA = sampleVolume(sdfDictionary, noiseData, coordA, realGridSize);
					float densityB = sampleVolume(sdfDictionary, noiseData, coordB, realGridSize);

					float t = (isoValue - densityA) / (densityB - densityA);
					if (isnan(t))
						t = 0;

					finalPos = p1 + t * (p2 - p1);

					finalPos += centerOffset;
					if (k == 0) {
						posC = finalPos;
					}
					else if (k == 1) {
						posB = finalPos;
					}
					else if (k == 2) {
						posA = finalPos;
					}
					if (EnableSmooth) {
						float dxA = sampleVolume(sdfDictionary, noiseData, coordA + offsetX, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordA - offsetX, realGridSize);
						float dyA = sampleVolume(sdfDictionary, noiseData, coordA + offsetY, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordA - offsetY, realGridSize);
						float dzA = sampleVolume(sdfDictionary, noiseData, coordA + offsetZ, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordA - offsetZ, realGridSize);
						float dxB = sampleVolume(sdfDictionary, noiseData, coordB + offsetX, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordB - offsetX, realGridSize);
						float dyB = sampleVolume(sdfDictionary, noiseData, coordB + offsetY, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordB - offsetY, realGridSize);
						float dzB = sampleVolume(sdfDictionary, noiseData, coordB + offsetZ, realGridSize) - sampleVolume(sdfDictionary, noiseData, coordB - offsetZ, realGridSize);
						// Normal:
						float3 normalTempA = normalizeCheck(make_float3(dxA, dyA, dzA));
						float3 normalTempB = normalizeCheck(make_float3(dxB, dyB, dzB));
						finalNormal = -normalizeCheck(normalTempA + t * (normalTempB - normalTempA));
						//finalNormal = normalTempA;
						// ID
						//int indexA = indexFromCoord(coordA);
						//int indexB = indexFromCoord(coordB);

						//Vertex vertex;
						//vertex.id = int2(min(indexA, indexB), max(indexA, indexB));

						if (k == 0) {
							normalC = finalNormal;
						}
						else if (k == 1) {
							normalB = finalNormal;
						}
						else if (k == 2) {
							normalA = finalNormal;
						}
					}
				}

				if (!EnableSmooth) {
					float3 tempNormal = cross(posB - posA, posC - posA);
					normalA = tempNormal;
					normalB = tempNormal;
					normalC = tempNormal;
				}

				tempTri.posA = make_Vector3(posA);
				tempTri.posB = make_Vector3(posB);
				tempTri.posC = make_Vector3(posC);
				tempTri.normalA = make_Vector3(normalA);
				tempTri.normalB = make_Vector3(normalB);
				tempTri.normalC = make_Vector3(normalC);

				//tempTri.ColorA.x = sampleVolumeColor(noiseData, cornerCoords[0], realGridSize);
				//tempTri.ColorA.y = sampleVolumeColor(noiseData, cornerCoords[1], realGridSize);
				//tempTri.ColorA.z = sampleVolumeColor(noiseData, cornerCoords[2], realGridSize);
				//tempTri.ColorA.w = sampleVolumeColor(noiseData, cornerCoords[3], realGridSize);
				//tempTri.ColorB.x = sampleVolumeColor(noiseData, cornerCoords[4], realGridSize);
				//tempTri.ColorB.y = sampleVolumeColor(noiseData, cornerCoords[5], realGridSize);
				//tempTri.ColorB.z = sampleVolumeColor(noiseData, cornerCoords[6], realGridSize);
				//tempTri.ColorB.w = sampleVolumeColor(noiseData, cornerCoords[7], realGridSize);

				if (enableColorSmoothing)
				{
					int voxelid = (gridPos.z * realGridSize.y + gridPos.y) * realGridSize.z + gridPos.x;
					// same as int id = (x * realGridSize.y + y) * realGridSize.z + z; (x = gridPos.z, z = gridPos.x)

					if (brushListCount > 0 && brushVoxel[voxelid].brushID < (uint)brushListCount)
					{
						tempTri.Colors[0] = brushList[brushVoxel[voxelid].brushID].shape.x * 1000;
						tempTri.Colors[1] = brushList[brushVoxel[voxelid].brushID].shape.y * 1000;
						tempTri.Colors[2] = brushList[brushVoxel[voxelid].brushID].shape.z * 1000;
						tempTri.Colors[3] = brushList[brushVoxel[voxelid].brushID].shape.w * 1000;
						tempTri.Colors[4] = brushList[brushVoxel[voxelid].brushID].colorID;
						tempTri.Colors[5] = brushList[brushVoxel[voxelid].brushID].type;

						tempTri.Colors[6] = brushList[brushVoxel[voxelid].lastBrushID].shape.x * 1000;
						tempTri.Colors[7] = brushList[brushVoxel[voxelid].lastBrushID].shape.y * 1000;
						tempTri.Colors[8] = brushList[brushVoxel[voxelid].lastBrushID].shape.z * 1000;
						tempTri.Colors[9] = brushList[brushVoxel[voxelid].lastBrushID].shape.w * 1000;
						tempTri.Colors[10] = brushList[brushVoxel[voxelid].lastBrushID].colorID;
						tempTri.Colors[11] = brushList[brushVoxel[voxelid].lastBrushID].type;

						tempTri.Colors[12] = brushVoxel[voxelid].backgroundColor;
						tempTri.Colors[13] = brushVoxel[voxelid].lastBackgroundColor;
						tempTri.Colors[14] = brushVoxel[voxelid].brushID;

					}
					else
					{
						uint colorID = sampleVolumeColor(sdfDictionary, noiseData, cornerCoords[0], realGridSize);
						for (int k = 0; k < 15; k++)
						{
							tempTri.Colors[k] = colorID;
						}
						tempTri.Colors[14] = brushVoxel[voxelid].brushID;
					}
				}
				else
				for (int k = 0; k < 8; k++)
				{
					tempTri.Colors[k] = sampleVolumeColor(sdfDictionary, noiseData, cornerCoords[k], realGridSize);
				}
				tempTri.cubeInfo = gridPos;
				//for (int k = 0; k < 9; k ++)
				//{
				//	tempTri.Colors[k] = tColor[k];
				//}
				//tempTri.Color = sampleVolumeColor(noiseData, gridPos, realGridSize);
				//tempTri.Color = 1;
				int id = atomicAdd(triNum, 1);

				//tempTri.posA.x = voxel;
				//tempTri.posA.y = cubeindex;
				if (id >= 0 && id < maxTri)
					Tribuffer[id] = tempTri;
			}
		//}
	}

	void launch_generateTriangles(cudaStream_t cudaStreams, int maxTri, int* triNum, float2* sdfDictionary, ushort2* noiseData, float3 CenterPos, int GridRes, int GridW, dim3 grid, int threads,
		int *compactedVoxelArray, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, Triangle* TribufferI, bool EnableSmooth, uint3 realGridSize, uint3 chunkPos, bool useChunk, 
		BrushInfo* brushList, BrushVoxel* brushVoxel, bool enableColorSmoothing, int brushListCount)
	{
		generateTriangles << <grid, threads, 0, cudaStreams >> > (maxTri, triNum, sdfDictionary, noiseData, CenterPos, GridRes, GridW,
			compactedVoxelArray,
			gridSize, gridSizeShift, gridSizeMask,
			voxelSize, isoValue, activeVoxels,
			maxVerts, TribufferI, EnableSmooth, realGridSize, chunkPos, useChunk, brushList, brushVoxel, enableColorSmoothing, brushListCount);
		getLastCudaError("Check generateTriangles Kernel");
	}

	void ThrustScanWrapper(int *output, int *input, int numElements)
	{
		thrust::exclusive_scan(thrust::device_ptr<int>(input),
			thrust::device_ptr<int>(input + numElements),
			thrust::device_ptr<int>(output));
	}
}