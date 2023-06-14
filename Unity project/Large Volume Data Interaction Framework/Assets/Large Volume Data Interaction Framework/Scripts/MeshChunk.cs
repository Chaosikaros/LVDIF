using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using Unity.Collections;
using ChaosIkaros;
using UnityEngine.Rendering;
using System.Linq;
//using Unity.Burst;
//using Unity.Jobs;

namespace ChaosIkaros.LVDIF
{
    public class MeshChunk : MonoBehaviour
    {
        //public static VertexAttributeDescriptor[] meshBufferLayout = new VertexAttributeDescriptor[]
        //{
        //     new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3),
        //     new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3),
        //     new VertexAttributeDescriptor(VertexAttribute.Color, VertexAttributeFormat.UNorm8, 4),
        //};
        public CudaMarchingCubesChunk parentChunk;
        public bool enableTimer = false;
        public bool logTimer = false;
        //public List<Color> colorTypes;
        public Vector3 cubeSize;
        public Vector3 unitSize;
        public Vector3 localCenter;
        public bool needUpate = false;
        public bool doneUpdate = false;
        public int chunkID = 0;
        public int triCount = 0;
        Vector3[] vertices;
        Vector3[] normals;
        int[] triangles;
        [NativeDisableContainerSafetyRestriction]
        NativeArray<Vector3> verticesNative;
        [NativeDisableContainerSafetyRestriction]
        NativeArray<Vector3> normalsNative;
        [NativeDisableContainerSafetyRestriction]
        NativeArray<int> trianglesNative;
        [NativeDisableContainerSafetyRestriction]
        NativeArray<int> colorsNative;
        GraphicsBuffer colorsBuffer;
        GraphicsBuffer verticesBuffer;
        GraphicsBuffer volumeColorsBuffer;
        MeshRenderer meshRenderer;
        Mesh mesh;
        MeshFilter meshFilter;
        MeshCollider meshCollider;
        int colorSmoothingMode = 0;// mode 1 is unfinished
                                   //public JobHandle jobHandle;
                                   // Start is called before the first frame update
        void Start()
        {
#if LVDIF_Haptic
        gameObject.tag = "Chunk";
#endif
            meshCollider = gameObject.AddComponent<MeshCollider>();
            meshFilter = gameObject.AddComponent<MeshFilter>();
            meshRenderer = gameObject.GetComponent<MeshRenderer>();
        }

        public void InitMeshChunk(int id, CudaMarchingCubesChunk chunk, Vector3 center)
        {
            localCenter = center;
            parentChunk = chunk;
            chunkID = id;
            mesh = new Mesh();
            gameObject.AddComponent<MeshRenderer>().material = parentChunk.mat;
            cubeSize = CudaMarchingCubesChunks.cudaMCManager.ScaleRatio
                * CudaMarchingCubesChunks.cudaMCManager.meshChunkSize * CudaMarchingCubesChunks.cudaMCManager.parentScale;
            //colorTypes = new List<Color>();
            //colorTypes.AddRange(CudaMarchingCubesChunks.cudaMCManager.volumeColors);
        }

        //public async Task UpdateMeshBufferTask(int triCountInput, int parentChunkID, int chunkID, float volumeWidth,
        //    float IsoLevel, int triangleThreads, bool EnableSmooth, NativeArray<Vector3> verticesNative,
        //    NativeArray<Vector3> normalsNative, NativeArray<int> trianglesNative, NativeArray<int> colorsNative)
        //{
        //    triCount = triCountInput;
        //    if (triCount < 1)
        //        return;
        //    unsafe
        //    {
        //        void* ptrV = NativeArrayUnsafeUtility.GetUnsafePtr(verticesNative);
        //        void* ptrN = NativeArrayUnsafeUtility.GetUnsafePtr(normalsNative);
        //        void* ptrT = NativeArrayUnsafeUtility.GetUnsafePtr(trianglesNative);
        //        void* ptrC = NativeArrayUnsafeUtility.GetUnsafePtr(colorsNative);
        //        CudaBridge.GetExtractMarchingCubesChunkData(parentChunkID, chunkID, triCount * 3, ptrV, ptrN, ptrT, ptrC,
        //            volumeWidth, triangleThreads, IsoLevel, EnableSmooth);
        //    }
        //}

        //public async void UpdateMeshTask(int triCountInput)
        //{
        //    triCount = triCountInput;
        //    if (triCount < 1)
        //        return;
        //    verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        //    normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        //    trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        //    colorsNative = new NativeArray<int>(triCount * 8, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        //    await UpdateMeshBufferTask(triCountInput, parentChunk.chunkID, chunkID,
        //                CudaMarchingCubesChunks.cudaMCManager.volumeWidth,CudaMarchingCubesChunks.cudaMCManager.IsoLevel, 
        //                CudaMarchingCubesChunks.cudaMCManager.triangleThreads,CudaMarchingCubesChunks.cudaMCManager.EnableSmooth,
        //                verticesNative, normalsNative, trianglesNative, colorsNative);
        //    UpdateMesh();
        //}

        public void UpdateMeshBufferThread(int triCountInput, int parentChunkID, int chunkID, float volumeWidth,
            float IsoLevel, int triangleThreads, bool EnableSmooth, NativeArray<Vector3> verticesNative,
            NativeArray<Vector3> normalsNative, NativeArray<int> trianglesNative, NativeArray<int> colorsNative)
        {
            triCount = triCountInput;
            if (triCount < 1)
                return;
            unsafe
            {
                void* ptrV = NativeArrayUnsafeUtility.GetUnsafePtr(verticesNative);
                void* ptrN = NativeArrayUnsafeUtility.GetUnsafePtr(normalsNative);
                void* ptrT = NativeArrayUnsafeUtility.GetUnsafePtr(trianglesNative);
                void* ptrC = NativeArrayUnsafeUtility.GetUnsafePtr(colorsNative);
                CudaBridge.GetExtractMarchingCubesChunkData(parentChunkID, chunkID, triCount * 3, ptrV, ptrN, ptrT, ptrC,
                    volumeWidth, triangleThreads, IsoLevel, EnableSmooth);
            }
            doneUpdate = true;
        }

        public bool UpdateMeshBuffer(int triCountInput)
        {
            doneUpdate = true;
            triCount = triCountInput;
            if (triCount < 1)
                return false;

            doneUpdate = false;
            verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            if (colorSmoothingMode == 1)
                colorsNative = new NativeArray<int>(triCount * 15, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            else
                colorsNative = new NativeArray<int>(triCount * 11, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            ThreadManager.RunOriginalAction(() => UpdateMeshBufferThread(triCountInput, parentChunk.chunkID, chunkID,
                    CudaMarchingCubesChunks.cudaMCManager.volumeWidth, CudaMarchingCubesChunks.cudaMCManager.IsoLevel,
                    CudaMarchingCubesChunks.cudaMCManager.triangleThreads, CudaMarchingCubesChunks.cudaMCManager.EnableSmooth,
                    verticesNative, normalsNative, trianglesNative, colorsNative));
            //vertices = new Vector3[triCount * 3];
            //normals = new Vector3[triCount * 3];
            //triangles = new int[triCount * 3];
            //CudaBridge.GetExtractMarchingCubesChunkData(parentChunk.chunkID, chunkID, triCount * 3, vertices, normals, triangles);

            //volumeColorsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, BrushManager.holder.volumeColors.Count, 16);
            //volumeColorsBuffer.SetData(BrushManager.holder.volumeColors);
            //meshRenderer.material.SetBuffer("volumeColorsBuffer", volumeColorsBuffer);

            //verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            //normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            //trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            //colorsNative = new NativeArray<int>(triCount * 8, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            //UpdateArray();

            //verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //colorsNative = new NativeArray<int>(triCount * 8, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //var job = new DllJob();
            //job.verticesNative = verticesNative;
            //job.normalsNative = normalsNative;
            //job.trianglesNative = trianglesNative;
            //job.colorsNative = colorsNative;

            //job.parentChunkID = parentChunk.chunkID;
            //job.chunkID = chunkID;
            //job.triCount = triCount;

            //job.volumeWidth = CudaMarchingCubesChunks.cudaMCManager.volumeWidth;
            //job.triangleThreads = CudaMarchingCubesChunks.cudaMCManager.triangleThreads;
            //job.IsoLevel = CudaMarchingCubesChunks.cudaMCManager.IsoLevel;
            //job.EnableSmooth = CudaMarchingCubesChunks.cudaMCManager.EnableSmooth;

            //jobHandle = job.Schedule();
            return true;
        }

        //[BurstCompile(CompileSynchronously = true, DisableSafetyChecks = true)]
        //public struct DllJob : IJob
        //{
        //    [WriteOnly]
        //    public NativeArray<Vector3> verticesNative;
        //    [WriteOnly]
        //    public NativeArray<Vector3> normalsNative;
        //    [WriteOnly]
        //    public NativeArray<int> trianglesNative;
        //    [WriteOnly]
        //    public NativeArray<int> colorsNative;

        //    [ReadOnly]
        //    public int parentChunkID;
        //    [ReadOnly]
        //    public int chunkID;
        //    [ReadOnly]
        //    public int triCount;
        //    [ReadOnly]
        //    public float volumeWidth;
        //    [ReadOnly]
        //    public float IsoLevel;
        //    [ReadOnly]
        //    public int triangleThreads;
        //    [ReadOnly]
        //    public bool EnableSmooth;
        //    public void Execute()
        //    {
        //        unsafe
        //        {
        //            void* ptrV = NativeArrayUnsafeUtility.GetUnsafePtr(verticesNative);
        //            void* ptrN = NativeArrayUnsafeUtility.GetUnsafePtr(normalsNative);
        //            void* ptrT = NativeArrayUnsafeUtility.GetUnsafePtr(trianglesNative);
        //            void* ptrC = NativeArrayUnsafeUtility.GetUnsafePtr(colorsNative);
        //            CudaBridge.GetExtractMarchingCubesChunkData(parentChunkID, chunkID, triCount * 3, ptrV, ptrN, ptrT, ptrC,
        //                    volumeWidth, triangleThreads, IsoLevel, EnableSmooth);
        //        }
        //    }
        //}


        private void UpdateArray()
        {
            unsafe
            {
                void* ptrV = NativeArrayUnsafeUtility.GetUnsafePtr(verticesNative);
                void* ptrN = NativeArrayUnsafeUtility.GetUnsafePtr(normalsNative);
                void* ptrT = NativeArrayUnsafeUtility.GetUnsafePtr(trianglesNative);
                void* ptrC = NativeArrayUnsafeUtility.GetUnsafePtr(colorsNative);
                CudaBridge.GetExtractMarchingCubesChunkData(parentChunk.chunkID, chunkID, triCount * 3, ptrV, ptrN, ptrT, ptrC,
                        CudaMarchingCubesChunks.cudaMCManager.volumeWidth, CudaMarchingCubesChunks.cudaMCManager.triangleThreads,
                        CudaMarchingCubesChunks.cudaMCManager.IsoLevel, CudaMarchingCubesChunks.cudaMCManager.EnableSmooth);
            }
        }

        public void UpdateMesh()
        {
            if (triCount < 1)
            {
                meshRenderer.enabled = false;
                meshCollider.enabled = false;
                return;
            }
            meshRenderer.enabled = true;
            meshCollider.enabled = true;
            //jobHandle.Complete();
            //var verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //var normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //var trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            //var colorsNative = new NativeArray<int>(triCount * 8, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            //job.Schedule().Complete();
            //mesh.Clear();
            //mesh.triangles = triangles;
            //mesh.vertices = vertices;
            //mesh.normals = normals;
            //meshFilter.sharedMesh = mesh;
            //meshCollider.sharedMesh = mesh;
            MeshUpdateFlags flags =
            //    MeshUpdateFlags.Default;
            //MeshUpdateFlags.DontNotifyMeshUsers
            //|
            MeshUpdateFlags.DontRecalculateBounds
            | MeshUpdateFlags.DontResetBoneBounds
            | MeshUpdateFlags.DontValidateIndices;

            mesh.SetVertexBufferParams(triCount * 3,
                new VertexAttributeDescriptor(VertexAttribute.Position, stream: 0),
                new VertexAttributeDescriptor(VertexAttribute.Normal, stream: 1));
            //mesh.SetVertexBufferParams(triCount * 3, meshBufferLayout);
            mesh.SetIndexBufferParams(triCount * 3, IndexFormat.UInt32);
            mesh.SetIndexBufferData(trianglesNative, 0, 0, trianglesNative.Length, flags);
            var submesh = new SubMeshDescriptor(0, triCount * 3, MeshTopology.Triangles);
            submesh.bounds = new Bounds(Vector3.zero, new Vector3(10, 10, 10));
            mesh.SetSubMesh(0, submesh);
            mesh.bounds = submesh.bounds;
            mesh.SetVertexBufferData(verticesNative, 0, 0, verticesNative.Length, 0, flags);
            mesh.SetVertexBufferData(normalsNative, 0, 0, normalsNative.Length, 1, flags);

            if (colorsBuffer != null)
                colorsBuffer.Dispose();

            if (verticesBuffer != null)
                verticesBuffer.Dispose();

            if (volumeColorsBuffer != null)
                volumeColorsBuffer.Dispose();

            if (colorSmoothingMode == 1)
                colorsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, triCount * 15, 4 * 15);
            else
                colorsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, triCount * 11, 4 * 11);
            colorsBuffer.SetData(colorsNative);
            meshRenderer.material.SetBuffer("colorsBuffer", colorsBuffer);
            verticesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, triCount * 3, 4 * 3);
            verticesBuffer.SetData(verticesNative);

            volumeColorsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, BrushManager.holder.volumeColors.Count, 16);
            volumeColorsBuffer.SetData(BrushManager.holder.volumeColors);
            meshRenderer.material.SetBuffer("volumeColorsBuffer", volumeColorsBuffer);

            meshRenderer.material.SetBuffer("verticesBuffer", verticesBuffer);
            //meshRenderer.material.SetColorArray("volumeColorsBuffer", BrushManager.holder.volumeColors);
            //meshRenderer.material.SetInt("_TriangleNum", triCount);
            //meshRenderer.material.SetInt("_VertexNum", triCount * 3);
            //meshRenderer.material.SetInt("_ColorNum", BrushManager.holder.volumeColors.Count);
            meshRenderer.material.SetFloat("_ColorSmoothing", colorSmoothingMode);
            meshRenderer.material.SetFloat("_ColorBrushOffset", BrushManager.holder.colorBrushOffset);
            meshRenderer.material.SetVector("_UnitSize", parentChunk.unitSize);
            meshRenderer.material.SetVector("_VoxelOriginal", parentChunk.voxelOriginal);
            meshRenderer.material.SetVector("_BrushCenterOffset", Vector3.zero);
            meshRenderer.material.SetVector("_CenterOffset", parentChunk.transform.position -
                CudaMarchingCubesChunks.cudaMCManager.volumeWidth * 0.5f * Vector3.one * parentChunk.transform.localScale.x * CudaMarchingCubesChunks.cudaMCManager.transform.localScale.x);
            // center offset = tranform.position - half of volumeWidth * Vector3.one * localScale
            //meshRenderer.material.SetVector("Voxel Original", parentChunk.voxelOriginal);
            //Color[] colors = new Color[triCount * 3];
            //for (int i = 0; i < triCount * 3; i+= 3)
            //{
            //    //colors[i] = colorTypes[colorsNative[i]];
            //    for (int k = 0; k < 8; k++)
            //        colors[i] += BrushManager.holder.volumeColors[colorsNative[(i / 3) * 8 + k]];
            //    colors[i] = colors[i] / 8;
            //    colors[i + 1] = colors[i];
            //    colors[i + 2] = colors[i];
            //}

            //mesh.colors = colors;
            //mesh.Optimize();
            meshFilter.sharedMesh = mesh;
            //if (mesh.vertices.Distinct().Count() >= 3)
            //{
            meshCollider.sharedMesh = mesh;
            //meshCollider.convex = true;
            //}
            //        if (mesh.vertices.Distinct().Count() >= 3)
            //            meshCollider.convex = true;
            verticesNative.Dispose();
            normalsNative.Dispose();
            trianglesNative.Dispose();
            colorsNative.Dispose();
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = new Color(0, 0.5f, 0.5f, 0.5f);
            Vector3 pos = localCenter;
            Gizmos.DrawCube(pos, cubeSize);
        }

        // Update is called once per frame
        //void Update()
        //{

        //}

        private void OnDestroy()
        {
            Destroy(mesh);
        }
    }
}
