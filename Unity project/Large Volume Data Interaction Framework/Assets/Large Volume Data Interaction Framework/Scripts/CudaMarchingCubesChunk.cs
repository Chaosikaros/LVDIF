using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Collections;
using ChaosIkaros;
using UnityEngine.Rendering;
//using Unity.Burst;
//using Unity.Jobs;

namespace ChaosIkaros.LVDIF
{
    public class CudaMarchingCubesChunk : MonoBehaviour
    {
        Mesh inputMesh;
        Transform meshTransfrom;
        Mesh mesh;
        MeshFilter meshFilter;
        MeshCollider meshCollider;
        public Material mat;
        public Vector3 cubeSize;
        public Vector3 unitSize;
        public Vector3 voxelOriginal;
        public Vector3 voxelCenter;
        public float meshOffset;
        public float inputOffset = 1;
        [Tooltip("Position offset")]
        public Vector3 CenterPos;
        public int chunkID = 0;
        public int triCount = 0;
        public bool hasInput = false;
        public bool loadModel = false;
        public bool stop = false;
        public bool enableTimer = false;
        public bool logTimer = false;
        //HideInInspector]
        public bool nextFrame = true;
        public bool updatedFrame = false;
        NativeArray<Vector3> verticesNative;
        NativeArray<Vector3> normalsNative;
        NativeArray<int> trianglesNative;
        private System.Diagnostics.Stopwatch timerAll;
        public Vector3 lastPos;
        void Start()
        {
            BrushManager.currentChunk = this;
            lastPos = transform.position;
            meshCollider = gameObject.AddComponent<MeshCollider>();
            meshFilter = gameObject.AddComponent<MeshFilter>();
            gameObject.AddComponent<MeshRenderer>().material = mat;
            //mesh = GetComponent<MeshFilter>().mesh;
            //mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        }

        public void InitChunk(int id, MeshFilter inputMeshFilter)
        {
            loadModel = CudaMarchingCubesChunks.cudaMCManager.loadModel;
            hasInput = false;
            updatedFrame = false;
            inputMesh = inputMeshFilter.mesh;
            meshTransfrom = inputMeshFilter.transform;
            chunkID = id;
            mesh = new Mesh();
            enableTimer = CudaMarchingCubesChunks.cudaMCManager.enableTimer;
            logTimer = CudaMarchingCubesChunks.cudaMCManager.logTimer;
            //CudaBridge.FreeMemoryForMC(chunkID);
            CudaBridge.MallocMemoryForMC(chunkID, CudaMarchingCubesChunks.cudaMCManager.GridResLog2,
                CudaMarchingCubesChunks.cudaMCManager.volumeWidth);
            timerAll = System.Diagnostics.Stopwatch.StartNew();
            cubeSize = CudaMarchingCubesChunks.cudaMCManager.ScaleRatio
                * CudaMarchingCubesChunks.cudaMCManager.gridSize
                * CudaMarchingCubesChunks.cudaMCManager.parentScale;
            unitSize = new Vector3(cubeSize.x / CudaMarchingCubesChunks.cudaMCManager.gridSize.x,
                cubeSize.y / CudaMarchingCubesChunks.cudaMCManager.gridSize.y,
                cubeSize.z / CudaMarchingCubesChunks.cudaMCManager.gridSize.z);
            meshOffset = 0.5f;
            StopAllCoroutines();
            StartCoroutine(UpdateMeshLoop());
        }

        IEnumerator UpdateMeshLoop()
        {
            if (enableTimer)
                Debug.Log("ChunkID: " + chunkID + ": Start Loop");
            while (true)
            {
                if (!stop)
                {
                    //stop = true;
                    yield return new WaitUntil(() => nextFrame);
                    if (enableTimer)
                        timerAll.Restart();
                    if (loadModel)
                    {
                        inputMesh.RecalculateBounds();
                        Bounds bounds = inputMesh.bounds;
                        Vector3[] verticesT = inputMesh.vertices;
                        int[] trianglesT = inputMesh.triangles;
                        Vector3[] normalsT = inputMesh.normals;
                        //for (int i = 0; i < verticesT.Length; i++)
                        //    verticesT[i] = meshTransfrom.localToWorldMatrix.MultiplyPoint3x4(verticesT[i]);
                        //for (int i = 0; i < verticesT.Length; i++)
                        //    Debug.Log("vertices[" + i + "]: " + verticesT[i].ToString());
                        //for (int i = 0; i < trianglesT.Length; i++)
                        //    Debug.Log("triangles[" + i + "]: " + trianglesT[i]);
                        //for (int i = 0; i < normalsT.Length; i++)
                        //    Debug.Log("normals[" + i + "]: " + normalsT[i]);
                        //for (int i = 0; i < trianglesT.Length; i++)
                        //    Debug.Log("vertices[triangles[" + i + "]]: " + verticesT[trianglesT[i]]);
                        //for (int i = 0; i < trianglesT.Length; i++)
                        //    Debug.Log("normals[triangles[" + i + "]]: " + normalsT[trianglesT[i]]);
                        //for (int i = 0; i < verticesT.Length; i++)
                        //{
                        //    Vector3 a = verticesT[i];
                        //    Vector3 b = CudaMarchingCubesChunks.cudaMCManager.inputMesh.gameObject.transform.localScale;
                        //    verticesT[i] = new Vector3(a.x / b.x, a.y / b.y, a.z / b.z); ;
                        //}
                        CudaBridge.SetMeshForVoxel(CudaMarchingCubesChunks.cudaMCManager.gridSize, bounds.max, bounds.min,
                            verticesT, trianglesT, normalsT, verticesT.Length, trianglesT.Length, normalsT.Length);
                        Debug.Log("ChunkID: " + chunkID + ": load mesh");
                    }
                    if (hasInput)
                    {
                        updatedFrame = true;
                        //hasInput = false;
                        //BrushManager.holder.inputRadius, 
                        //    BrushManager.holder.cursorVoxelPos, CudaMarchingCubesChunks.cudaMCManager.mouseBrush.eraserMode);
                        CudaBridge.SetInputBrushForVoxel(chunkID, (int)BrushManager.holder.inputRadius,
                            BrushManager.holder.finalCursorPosInBrush.Count,
                        BrushManager.holder.finalCursorPosInBrush.ToArray(), BrushManager.holder.colorType,
                        BrushManager.holder.colorBrushOffset, BrushManager.holder.eraserMode,
                        (int)BrushManager.holder.brushShape);
                    }
                    //CenterPos = - 0.5f * unitSize;
                    CenterPos = Vector3.zero;
                    lastPos = transform.localPosition;
                    Vector3 pos = 2 * lastPos;
                    voxelCenter = pos;
                    //voxelCenter = pos+ 0.5f * unitSize;
                    voxelOriginal = voxelCenter - 0.5f * cubeSize;
                    if (enableTimer)
                    {
                        timerAll.Stop();
                        if (logTimer)
                            Debug.Log("ChunkID: " + chunkID + ": Set Marching Cubes Kernel Thread Pre-----------" + timerAll.ElapsedMilliseconds.ToString());
                        timerAll.Restart();
                    }
                    for (int i = 0; i < CudaMarchingCubesChunks.meshChunkUpdateStates.Count; i++)
                    {
                        CudaMarchingCubesChunks.meshChunkUpdateStates[i] = false;
                    }
                    //lastPos + CenterPos + meshOffset * unitSize
                    CudaBridge.SetMarchingCubesKernelInThread(chunkID, CudaMarchingCubesChunks.cudaMCManager.voxelThreads,
                        CudaMarchingCubesChunks.cudaMCManager.triangleThreads, lastPos + CenterPos + meshOffset * unitSize, CudaMarchingCubesChunks.cudaMCManager.volumeWidth,
                        CudaMarchingCubesChunks.cudaMCManager.octreeBboxSizeOffset, CudaMarchingCubesChunks.cudaMCManager.IsoLevel,
                        CudaMarchingCubesChunks.cudaMCManager.EnableSmooth, loadModel,
                        CudaMarchingCubesChunks.cudaMCManager.loadSDF, CudaMarchingCubesChunks.cudaMCManager.ImportSDF(),
                        CudaMarchingCubesChunks.cudaMCManager.GridSizeLog2OctreeBox,
                        CudaMarchingCubesChunks.cudaMCManager.exportTexture3D, CudaMarchingCubesChunks.cudaMCManager.GetImportSDFPath()
                        , CudaMarchingCubesChunks.cudaMCManager.filterValue, CudaMarchingCubesChunks.cudaMCManager.sleepTime, CudaMarchingCubesChunks.cudaMCManager.maxUpdatedChunkPerFrame,
                        (int)CudaMarchingCubesChunks.cudaMCManager.SVFSetting);
                    if (loadModel)
                        loadModel = false;
                    if (enableTimer)
                    {
                        timerAll.Stop();
                        if (logTimer)
                            Debug.Log("ChunkID: " + chunkID + ": Set Marching Cubes Kernel Thread After-----------" + timerAll.ElapsedMilliseconds.ToString());
                        timerAll.Restart();
                    }
                    yield return new WaitUntil(() => CudaMarchingCubesChunks.chunkStates[chunkID]);
                    CudaMarchingCubesChunks.chunkStates[chunkID] = false;
                    triCount = CudaBridge.GetExtractMarchingTriCount(chunkID);
                    CudaMarchingCubesChunks.cudaMCManager.triangleNum = triCount;
                    //Debug.Log("TriCount: " + triCount);
                    if (enableTimer)
                    {
                        timerAll.Stop();
                        if (logTimer)
                            Debug.Log("ChunkID: " + chunkID + ": Set Marching Cubes Kernel Thread CallBack-----------" + timerAll.ElapsedMilliseconds.ToString());
                        timerAll.Restart();
                    }

                    //if (triCount == 0)
                    //    Debug.Log("ChunkID: " + chunkID + ": Marching Cubes Kernel error !");
                    //else
                    //{
                    //Vector3[] vertices = new Vector3[triCount * 3];
                    //Vector3[] normals = new Vector3[triCount * 3];
                    //int[] triangles = new int[triCount * 3];
                    //verticesNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    //normalsNative = new NativeArray<Vector3>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    //trianglesNative = new NativeArray<int>(triCount * 3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                    //UpdateArray();
                    //CudaBridge.GetExtractMarchingCubesData(chunkID, triCount * 3, vertices, normals, triangles);
                    int updatedChunkCounter = 0;
                    for (int i = 0; i < CudaMarchingCubesChunks.meshChunkUpdateStates.Count; i++)
                    {
                        CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().needUpate =
                            CudaMarchingCubesChunks.meshChunkUpdateStates[i];
                        if (CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().needUpate)
                        {
                            updatedChunkCounter++;
                            if (CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().UpdateMeshBuffer(
                                    CudaMarchingCubesChunks.meshChunkTriCounters[i]))
                                //yield return new WaitUntil(()=> CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().jobHandle.IsCompleted);
                                yield return new WaitUntil(() => CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().doneUpdate);
                            CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().UpdateMesh();
                            //CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().UpdateMeshTask(
                            //        CudaMarchingCubesChunks.meshChunkTriCounters[i]);
                            if (updatedChunkCounter > CudaMarchingCubesChunks.cudaMCManager.maxUpdatedChunkPerFrame
                                        && updatedChunkCounter % CudaMarchingCubesChunks.cudaMCManager.maxUpdatedChunkPerFrame == 0)
                                yield return new WaitForSeconds((float)(CudaMarchingCubesChunks.cudaMCManager.sleepTime / 1000.0f));
                        }
                    }
                    //updatedChunkCounter = 0;
                    if (enableTimer)
                    {
                        timerAll.Stop();
                        if (logTimer)
                            Debug.Log("ChunkID: " + chunkID + ": Get Extract Marching Cubes Data -----" + timerAll.ElapsedMilliseconds.ToString());
                        timerAll.Restart();
                    }

                    //for (int i = 0; i < CudaMarchingCubesChunks.cudaMCManager.meshChunks.Count; i++)
                    //{
                    //    if (CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().needUpate)
                    //    {
                    //        updatedChunkCounter++;
                    //        CudaMarchingCubesChunks.cudaMCManager.meshChunks[i].GetComponent<MeshChunk>().UpdateMesh();
                    //        if (updatedChunkCounter > CudaMarchingCubesChunks.cudaMCManager.maxUpdatedChunkPerFrame
                    //            && updatedChunkCounter % CudaMarchingCubesChunks.cudaMCManager.maxUpdatedChunkPerFrame == 0)
                    //            yield return new WaitForSeconds((float)(CudaMarchingCubesChunks.cudaMCManager.sleepTime / 1000.0f));
                    //    }
                    //}

                    //MeshUpdateFlags flags =
                    //    //MeshUpdateFlags.DontNotifyMeshUsers
                    //    //|
                    //    MeshUpdateFlags.DontRecalculateBounds
                    //    | MeshUpdateFlags.DontResetBoneBounds
                    //    | MeshUpdateFlags.DontValidateIndices;
                    //mesh.SetVertexBufferParams(triCount * 3,
                    //    new VertexAttributeDescriptor(VertexAttribute.Position, stream: 0),
                    //    new VertexAttributeDescriptor(VertexAttribute.Normal, stream: 1));
                    //mesh.SetIndexBufferParams(triCount * 3, IndexFormat.UInt32);
                    //mesh.SetIndexBufferData(trianglesNative, 0, 0, trianglesNative.Length, flags);
                    //var submesh = new SubMeshDescriptor(0, triCount * 3, MeshTopology.Triangles);
                    //submesh.bounds = new Bounds(Vector3.zero, new Vector3(10, 10, 10));
                    //mesh.SetSubMesh(0, submesh);
                    //mesh.bounds = submesh.bounds;
                    //mesh.SetVertexBufferData(verticesNative, 0, 0, verticesNative.Length, 0, flags);
                    //mesh.SetVertexBufferData(normalsNative, 0, 0, normalsNative.Length, 1, flags);
                    ////mesh.Clear();
                    ////mesh.SetVertices(verticesNative);
                    ////mesh.SetNormals(normalsNative);
                    ////mesh.triangles = triangles;
                    ////mesh.vertices = vertices;
                    ////mesh.normals = normals;
                    ////mesh.triangles = triangles;
                    //meshFilter.sharedMesh = mesh;
                    //meshCollider.sharedMesh = mesh;

                    if (enableTimer)
                    {
                        timerAll.Stop();
                        if (logTimer)
                            Debug.Log("ChunkID: " + chunkID + ": Update Mesh -----------------------" + timerAll.ElapsedMilliseconds.ToString());
                    }

                    //}
                    //DrawCubeVoxels();
                    nextFrame = false;
                    if (updatedFrame && hasInput)
                    {
                        hasInput = false;
                        updatedFrame = false;
                    }
                    CudaMarchingCubesChunks.cudaMCManager.chunkUpdateCounter++;
                }
                else
                {
                    yield return new WaitForSeconds(1.0f);
                }
            }
        }

        public void UpdateArray()
        {
            unsafe
            {
                void* ptrV = NativeArrayUnsafeUtility.GetUnsafePtr(verticesNative);
                void* ptrN = NativeArrayUnsafeUtility.GetUnsafePtr(normalsNative);
                void* ptrT = NativeArrayUnsafeUtility.GetUnsafePtr(trianglesNative);
                CudaBridge.GetExtractMarchingCubesData(chunkID, triCount * 3, ptrV, ptrN, ptrT);
            }
        }
        void Update()
        {

        }

        public bool IsVaildInput(Vector3 cursorPos)
        {
            Vector3 inputPos = cursorPos - voxelOriginal;
            Vector3 output = new Vector3((float)Math.Abs(inputPos.z / unitSize.z),
                (float)Math.Abs(inputPos.y / unitSize.y),
                (float)Math.Abs(inputPos.x / unitSize.x));
            //output += inputOffset * meshOffset * Vector3.one;
            if (output.x > 0 && output.y > 0 && output.z > 0 &&
                output.x < CudaMarchingCubesChunks.cudaMCManager.gridSize.x - 1 &&
                output.y < CudaMarchingCubesChunks.cudaMCManager.gridSize.y - 1 &&
                output.z < CudaMarchingCubesChunks.cudaMCManager.gridSize.z - 1)
                return true;
            return false;
        }

        public Vector3 GetVoxelPos(Vector3 cursorPos)
        {
            Vector3 inputPos = cursorPos - voxelOriginal;
            Vector3 output = new Vector3((float)Math.Abs(inputPos.z / unitSize.z),
                (float)Math.Abs(inputPos.y / unitSize.y),
                (float)Math.Abs(inputPos.x / unitSize.x));
            //output += inputOffset * meshOffset * Vector3.one;
            output.x = Mathf.Clamp(output.x, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.x - 1);
            output.y = Mathf.Clamp(output.y, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.y - 1);
            output.z = Mathf.Clamp(output.z, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.z - 1);
            return output;
        }

        public Vector3 GetVoxelPosInVolume(Vector3 cursorPos)
        {
            //Vector3 inputPos = cursorPos - 0.25f * meshOffset * Vector3.one / CudaMarchingCubesChunks.cudaMCManager.gridSize.x;
            Vector3 inputPos = cursorPos;
            Vector3 output = new Vector3(CudaMarchingCubesChunks.cudaMCManager.gridSize.x * inputPos.x,
                CudaMarchingCubesChunks.cudaMCManager.gridSize.y * inputPos.y,
                CudaMarchingCubesChunks.cudaMCManager.gridSize.z * inputPos.z);
            output.x = Mathf.Clamp(output.x, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.x - 1);
            output.y = Mathf.Clamp(output.y, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.y - 1);
            output.z = Mathf.Clamp(output.z, 0, CudaMarchingCubesChunks.cudaMCManager.gridSize.z - 1);
            return output;
        }

        public void DrawCubeVoxels()
        {
            int cubeVoxelsCount = (int)CudaMarchingCubesChunks.cudaMCManager.gridSize.x * (int)CudaMarchingCubesChunks.cudaMCManager.gridSize.y * (int)CudaMarchingCubesChunks.cudaMCManager.gridSize.z;
            float[] cubeVoxels = new float[cubeVoxelsCount];
            Debug.Log("GetExtractCubeVoxels: " + CudaBridge.GetExtractCubeVoxels(chunkID, cubeVoxelsCount, cubeVoxels));
            Vector3 pos = 2 * lastPos;
            Vector3 cubeSizeT = cubeSize + unitSize;
            Vector3 unitSizeT = new Vector3(cubeSizeT.x / CudaMarchingCubesChunks.cudaMCManager.gridSize.x,
                cubeSizeT.y / CudaMarchingCubesChunks.cudaMCManager.gridSize.y,
                cubeSizeT.z / CudaMarchingCubesChunks.cudaMCManager.gridSize.z);
            Vector3 origianlPos = pos + 0.5f * unitSize - 0.5f * cubeSizeT + 0.5f * unitSizeT;
            //origianlPos = pos + unitSize - 0.5f * cubeSizeT + 0.5f * unitSizeT;
            Gizmos.color = new Color(0, 0.5f, 0.5f, 0.5f);
            for (int x = 0; x < CudaMarchingCubesChunks.cudaMCManager.gridSize.x; x++)
            {
                for (int y = 0; y < CudaMarchingCubesChunks.cudaMCManager.gridSize.y; y++)
                {
                    for (int z = 0; z < CudaMarchingCubesChunks.cudaMCManager.gridSize.z; z++)
                    {
                        //Debug.Log(cubeVoxels[(int)((x * CudaMarchingCubesChunks.cudaMCManager.gridSize.y + y) * CudaMarchingCubesChunks.cudaMCManager.gridSize.z + z)]);
                        if (cubeVoxels[(int)((x * CudaMarchingCubesChunks.cudaMCManager.gridSize.y + y) * CudaMarchingCubesChunks.cudaMCManager.gridSize.z + z)] > CudaMarchingCubesChunks.cudaMCManager.IsoLevel)
                        {
                            GameObject go = GameObject.Instantiate(CudaMarchingCubesChunks.cudaMCManager.voxelCube, CudaMarchingCubesChunks.cudaMCManager.voxelParent);
                            go.transform.localScale = 0.9f * unitSizeT;
                            go.transform.position = origianlPos + new Vector3(unitSizeT.x * x, unitSizeT.y * y, unitSizeT.z * z);
                        }
                    }
                }
            }
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = new Color(0, 0.5f, 0.5f, 0.2f);
            Vector3 pos = transform.TransformVector(2 * lastPos + CenterPos);
            Gizmos.DrawCube(pos, cubeSize);
        }

        private void OnDestroy()
        {

        }
    }
}
