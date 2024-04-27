using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using TMPro;

namespace ChaosIkaros.LVDIF
{

    public enum SVFSetting
    {
        Off,
        SVF1,
        SVF2,
        SVF3,
        SVF4
    }

    public class CudaMarchingCubesChunks : MonoBehaviour
    {
        public SVFSetting SVFSetting = SVFSetting.SVF4;

        const int MaxArraySize = 1024 * 1024 * 1024;
        public static bool exist = false;
        public static CudaMarchingCubesChunks cudaMCManagerHolder;
        public static CudaMarchingCubesChunks cudaMCManager
        {
            get
            {
                InitManager();
                return cudaMCManagerHolder;
            }
        }

        public static void InitManager()
        {
            if (!exist)
            {
                exist = true;
                GameObject managerHolder = GameObject.Find("MarchingCubeCudaChunks");
                if (managerHolder == null)
                    managerHolder = new GameObject("MarchingCubeCudaChunks");
                DontDestroyOnLoad(managerHolder);
                if (!managerHolder.GetComponent<CudaMarchingCubesChunks>())
                    cudaMCManagerHolder = managerHolder.AddComponent<CudaMarchingCubesChunks>();
                else
                    cudaMCManagerHolder = managerHolder.GetComponent<CudaMarchingCubesChunks>();
                CudaUtility.InitializedCudaDebugCallBack();
            }
        }

        void Awake()
        {
            InitManager();
            DontDestroyOnLoad(gameObject);
        }
        [DllImport("CudaUnity")]
        public static extern void SetCallbackForMeshChunk(CallbackForMeshChunk callback);

        [DllImport("CudaUnity")]
        public static extern void SetCallbackForMC(CallbackForMC callback);

        [DllImport("CudaUnity")]
        public static extern void SetCallbackForSdfExport(SdfCallback callback);

        public delegate void CallbackForMeshChunk(int chunkID, bool state, int triCountInput);

        public delegate void CallbackForMC(int chunkID);

        public delegate void SdfCallback([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] float[] sdfData, int size);
        public void MeshChunkCallbackFuction(int chunkID, bool state, int triCountInput)
        {
            meshChunkUpdateStates[chunkID] = state;
            meshChunkTriCounters[chunkID] = triCountInput;
            //Debug.Log("Mesh Chunk ID: " + chunkID + "; Trinum: " + triCountInput);
        }
        static CallbackForMeshChunk meshChunkCallback;

        public void McCallbackFuction(int chunkID)
        {
            chunkStates[chunkID] = true;
        }
        static CallbackForMC callback;

        public void SdfCallbackFuction(float[] sdfData, int size)
        {
            CudaMarchingCubesChunks.createNewSDF = true;
            if (CudaMarchingCubesChunks.cudaMCManager.exportTexture3D)
            {
                CudaMarchingCubesChunks.lastSDF = new float[size];
                CudaMarchingCubesChunks.lastSDF = sdfData;
            }
        }

        static SdfCallback sdfCallback;

        public static List<bool> chunkStates = new List<bool>();
        public static List<bool> meshChunkUpdateStates = new List<bool>();
        public static List<int> meshChunkTriCounters = new List<int>();
        public Material wallMaterial;
        public GameObject allUI;
        public GameObject uiGroup;
        public TMP_Dropdown inputFileDropdown;
        public TMP_Dropdown switchModeDropdown;
        public Slider volumeSizeSlider;
        public Slider volumeChunkSizeSlider;
        public Text volumeSizeText;
        public Text volumeChunkSizeText;
        public Button initButton;
        public Button resetButton;
        public Text statusDebug;
        public Text fpsDisplay;
        public static float[] lastSDF;
        public static ushort2[] ushort2SDF;
        public Texture3D exportSDF;
        public bool exportTexture3D = false;
        public bool exportNewSdf = false;
        public bool loadSDF = false;
        public bool loadSDFFromUnity = false;
        public bool loadModel = true;
        public string importSDF;
        public List<Color> volumeColors = new List<Color>();
        private float[] importLargeSDFArray;
        public static bool createNewSDF = false;
        public Transform voxelParent;
        public GameObject voxelCube;
        public MeshFilter inputMesh;
        Mesh mesh;
        public GameObject chunkPrefab;
        public GameObject meshChunkPrefab;
        [Tooltip("Size of All Chunks")]
        public Vector3Int ChunksSize;
        [Tooltip("Smooth normals are slower than flat normals")]
        public bool EnableSmooth;
        public Material mat;
        [Tooltip("Max voxel number")]
        public int maxVoxelNum;
        [Tooltip("Actual voxel number")]
        public int actualVoxelNum;
        [Tooltip("Grid size")]
        public Vector3 gridSize;
        public Vector3 GridResLog2;
        public Vector3 meshChunkGroupSize;
        public Vector3 meshChunkSize;
        [Tooltip("VoxelNum = 2^GridResLog2 * 2^GridResLog2 * 2^GridResLog2")]
        [Range(6, 10)]
        public int GridResLog2_X = 6;
        public int octreeBoxSize = 4;
        [Range(3, 7)]
        public int GridSizeLog2OctreeBox = 3;
        [Range(0.5f, 5)]
        public float octreeBboxSizeOffset = 2;
        public int minChunkSize = 16;
        [Range(4, 10)]
        public int minChunkSizeLog2 = 5;
        public int chunkThreads = 2;
        [Range(0, 8)]
        public int chunkThreadsLog2 = 5;
        [Range(1, 3)]
        private int ChunkSize_X = 1;
        [Range(1, 3)]
        private int ChunkSize_Y = 1;
        [Range(1, 3)]
        private int ChunkSize_Z = 1;
        [Range(1, 100)]
        public int maxUpdatedChunkPerFrame = 6;
        [Range(1, 1000)]
        public int sleepTime = 10;
        public float parentScale = 1.0f;
        public float ScaleRatio = 1.0f;
        [Tooltip("Edge width of the volume")]
        public float volumeWidth = 2f;
        public float NoiseInterval = 0.05f;
        public float IsoLevel = 0.2f;
        public float filterValue = 1.5f;
        //Thread number of voxel kernel
        [HideInInspector]
        public int voxelThreads = 128;
        //Thread number of triangle kernel
        [HideInInspector]
        public int triangleThreads = 32;//
        public bool enableTimer = false;
        public bool logTimer = false;
        public bool stop = false;
        public List<GameObject> chunks = new List<GameObject>();
        public List<GameObject> meshChunks = new List<GameObject>();
        [HideInInspector]
        public int chunkUpdateCounter = 0;
        public int triangleNum = 0;
        public int chunkNum = 0;
        public static int voxelSize = 0;
        private System.Diagnostics.Stopwatch timerAll;
        private string dirpath = "";
        private Queue<float> frameTime = new Queue<float> { };
        public List<string> fileNames = new List<string>();
        public static float globalFPS;
        void Start()
        {
            maxUpdatedChunkPerFrame = 100;
            sleepTime = 1;
            //return;
            dirpath = Application.dataPath + "/Large Volume Data Interaction Framework/Textures/";
            if (!Directory.Exists(dirpath))
            {
                Directory.CreateDirectory(dirpath);
            }
            for (int i = 0; i < 10; i++)
            {
                frameTime.Enqueue(0);
            }
            timerAll = System.Diagnostics.Stopwatch.StartNew();
            inputFileDropdown.ClearOptions();
            UpdateFileList();
            if (enableTimer)
                StartCoroutine(TimerLoop());
        }

        public void UpdateFileList()
        {
            fileNames.Clear();
            DirectoryInfo folder = new DirectoryInfo(dirpath);
            FileInfo[] files = folder.GetFiles();
            List<FileInfo> fileList = new List<FileInfo>();
            for (int i = 0; i < files.Length; i++)
            {
                if (files[i].Name.Contains(".bv") && !files[i].Name.Contains(".meta"))
                {
                    fileNames.Add(files[i].Name);
                }
            }
            if (fileNames.Count == 0)
                SwitchMode(0);
            else
            {
                inputFileDropdown.ClearOptions();
                inputFileDropdown.AddOptions(fileNames);
            }
        }

        public void ToggleUIGroup()
        {
            uiGroup.SetActive(!uiGroup.activeSelf);
        }
        public void OnFileSelection(int input)
        {
            importSDF = fileNames[input];
            ClampInputParameters();
        }

        public void SwitchMode(int input)
        {
            if (input == 0 || fileNames.Count == 0)
            {
                exportNewSdf = true;
                loadSDF = false;
                loadModel = true;
            }
            else
            {
                exportNewSdf = false;
                loadSDF = true;
                loadModel = false;
                OnFileSelection(inputFileDropdown.value);
            }
        }

        public void SetVolumeSize()
        {
            GridResLog2_X = (int)volumeSizeSlider.value;
            ClampInputParameters();
        }
        public void SetVolumeChunkSize()
        {
            minChunkSizeLog2 = (int)volumeChunkSizeSlider.value;
            ClampInputParameters();
        }
        public void ResetVolume()
        {
            initButton.gameObject.SetActive(true);
            resetButton.gameObject.SetActive(false);
            for (int i = 0; i < chunks.Count; i++)
            {
                CudaBridge.FreeMemoryForMC(i);
                GameObject.Destroy(chunks[i]);
            }
            for (int i = 0; i < meshChunks.Count; i++)
            {
                GameObject.Destroy(meshChunks[i]);
            }
            StopAllCoroutines();
            if (enableTimer)
                StartCoroutine(TimerLoop());
            SwitchMode(switchModeDropdown.value);
            triangleNum = 0;
            System.GC.Collect();
        }
        public void InitChunk()
        {
            SwitchMode(switchModeDropdown.value);
            chunks.Clear();
            meshChunks.Clear();
            chunkStates.Clear();
            meshChunkUpdateStates.Clear();
            meshChunkTriCounters.Clear();
            initButton.gameObject.SetActive(false);
            resetButton.gameObject.SetActive(true);
            ClampInputParameters();
            voxelSize = (int)gridSize.x;
            if (voxelSize > 512)
                exportTexture3D = false;
            if (voxelThreads > 0)
                while (maxVoxelNum / voxelThreads < 1 || voxelThreads > 1024)
                    voxelThreads /= 2;
            if (triangleThreads > 0)
                while (maxVoxelNum / triangleThreads < 1 || triangleThreads > 1024)
                    triangleThreads /= 2;

            meshChunkCallback = MeshChunkCallbackFuction;
            SetCallbackForMeshChunk(meshChunkCallback);
            callback = McCallbackFuction;
            SetCallbackForMC(callback);
            sdfCallback = SdfCallbackFuction;
            SetCallbackForSdfExport(sdfCallback);

            //gameObject.AddComponent<MeshFilter>();
            //gameObject.AddComponent<MeshRenderer>().material = mat;
            //mesh = GetComponent<MeshFilter>().mesh;
            //mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            //ClampInputParameters();
            if (CudaBridge.SetMarchingCubesChunks(chunkNum, minChunkSizeLog2, chunkThreads))
            {
                Debug.Log("Safe Chunk");
                for (int x = 0; x < ChunksSize.x; x++)
                {
                    for (int y = 0; y < ChunksSize.y; y++)
                    {
                        for (int z = 0; z < ChunksSize.z; z++)
                        {
                            chunkStates.Add(false);
                            int id = (x * ChunksSize.y + y) * ChunksSize.z + z;
                            GameObject newChunk = GameObject.Instantiate(chunkPrefab, transform);
                            chunks.Add(newChunk);
                            newChunk.transform.localPosition = 0.5f * ScaleRatio * new Vector3(gridSize.x * x, gridSize.y * y, gridSize.z * z);
                            newChunk.GetComponent<CudaMarchingCubesChunk>().mat = mat;
                            newChunk.GetComponent<CudaMarchingCubesChunk>().InitChunk(id, inputMesh);
                            newChunk.name = "Chunk " + id;
                            Vector3 meshChunkGroupCenter = newChunk.transform.localPosition;
                            Vector3 scaledMeshChunkSize = ScaleRatio * meshChunkSize;
                            Vector3 meshChunkGroupOriginal = meshChunkGroupCenter - 0.5f * ScaleRatio * gridSize
                                + 0.5f * scaledMeshChunkSize;
                            for (int xM = 0; xM < meshChunkGroupSize.x; xM++)
                            {
                                for (int yM = 0; yM < meshChunkGroupSize.y; yM++)
                                {
                                    for (int zM = 0; zM < meshChunkGroupSize.z; zM++)
                                    {
                                        meshChunkUpdateStates.Add(false);
                                        meshChunkTriCounters.Add(0);
                                        int idM = (int)((xM * meshChunkGroupSize.y + yM) * meshChunkGroupSize.z + zM);
                                        GameObject newMeshChunk = GameObject.Instantiate(meshChunkPrefab, newChunk.transform);
                                        meshChunks.Add(newMeshChunk);
                                        //newMeshChunk.transform.localPosition = meshChunkGroupOriginal + new Vector3(scaledMeshChunkSize.x * xM,
                                        //    scaledMeshChunkSize.y * yM, scaledMeshChunkSize.z * zM);
                                        //newMeshChunk.transform.localPosition += new Vector3(-0.5f * meshChunkOffset, -0.5f * meshChunkOffset, -0.5f * meshChunkOffset);
                                        newMeshChunk.GetComponent<MeshChunk>().InitMeshChunk(idM, newChunk.GetComponent<CudaMarchingCubesChunk>(),
                                            newMeshChunk.transform.TransformVector(meshChunkGroupOriginal + new Vector3(scaledMeshChunkSize.x * xM,
                                            scaledMeshChunkSize.y * yM, scaledMeshChunkSize.z * zM)));
                                        newMeshChunk.transform.localPosition = newChunk.transform.localPosition;
                                        newMeshChunk.name = "MeshChunk " + idM;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StartCoroutine(ChunksManager());
        }

        public string GetImportSDFPath()
        {
            return dirpath + importSDF;
        }

        public ushort2[] ImportSDF()
        {
            if (loadSDFFromUnity)
            {
                loadSDF = false;
                loadSDFFromUnity = false;
                return ushort2SDF;
            }
            else if (loadSDF)
            {
                loadSDF = false;
                if (!File.Exists(GetImportSDFPath()))
                {
                    Debug.Log("Cannot find the file:" + GetImportSDFPath());
                }
                return null;
            }
            return null;
        }
        public IEnumerator TimerLoop()
        {
            while (true)
            {
                //timerAll.Reset();
                //timerAll.Start();
                frameTime.Dequeue();
                frameTime.Enqueue(1f / Time.unscaledDeltaTime);
                //yield return new WaitForEndOfFrame();
                yield return new WaitForSeconds(0.1f);
                //frameTime.Dequeue();
                //frameTime.Enqueue((float) 1000 / (float)timerAll.ElapsedMilliseconds);
                globalFPS = frameTime.ToArray().Average();
                fpsDisplay.text = "FPS: " + globalFPS.ToString("F2");
            }
        }
        IEnumerator ChunksManager()
        {
            int counter = 0;
            float timeAll = 0;
            while (true)
            {
                if (!stop)
                {
                    //stop = true;
                    //if (enableTimer)
                    //{
                    //    timerAll.Reset();
                    //    timerAll.Start();
                    //}
                    for (int i = 0; i < chunkNum; i++)
                        chunks[i].GetComponent<CudaMarchingCubesChunk>().nextFrame = true;
                    yield return new WaitUntil(() => chunkUpdateCounter == chunkNum);
                    chunkUpdateCounter = 0;
                    //if (enableTimer)
                    //{
                    //    timerAll.Stop();
                    //    //counter++;
                    //    frameTime.Dequeue();
                    //    frameTime.Enqueue(timerAll.ElapsedMilliseconds);
                    //    timeAll = frameTime.ToArray().Average();
                    //    fpsDisplay.text = "FPS: " + (1000 / timeAll).ToString("F2");
                    //    //Debug.Log("Average FPS: " + 1000 / (timeAll / counter) + "; Last / Average Frame Gap: " + 
                    //    //    timerAll.ElapsedMilliseconds.ToString() + "/" + timeAll / counter);
                    //}
                    if (loadSDF)
                        loadSDF = false;
                }
                else
                {
                    yield return new WaitForSeconds(1.0f);
                }
            }
        }

        private void OnValidate()
        {
            ClampInputParameters();
        }

        public void ClampInputParameters()
        {
            transform.localScale = parentScale * Vector3.one;
            if (!loadSDFFromUnity && loadSDF && importSDF != "")
            {
                GridResLog2_X = (int)Mathf.Log(float.Parse(importSDF.Split('_')[1]), 2);
            }
            octreeBoxSize = (int)Mathf.Pow(2, GridSizeLog2OctreeBox);
            if (GridResLog2_X >= 10)
                minChunkSizeLog2 = Math.Clamp(minChunkSizeLog2, 5, GridResLog2_X);
            else
                minChunkSizeLog2 = Math.Clamp(minChunkSizeLog2, 4, GridResLog2_X);
            GridResLog2_X = Math.Clamp(GridResLog2_X, 6, 10);
            minChunkSizeLog2 = Math.Min(minChunkSizeLog2, GridResLog2_X);
            minChunkSize = (int)Mathf.Pow(2, minChunkSizeLog2);
            GridResLog2 = new Vector3(GridResLog2_X, GridResLog2_X, GridResLog2_X);
            gridSize = new Vector3(Mathf.Pow(2, GridResLog2_X), Mathf.Pow(2, GridResLog2_X), Mathf.Pow(2, GridResLog2_X));
            chunkThreads = Math.Min((int)Mathf.Pow(2, chunkThreadsLog2), (int)((gridSize.x / minChunkSize)
                * (gridSize.y / minChunkSize) * (gridSize.z / minChunkSize)));
            chunkThreadsLog2 = (int)Mathf.Log(chunkThreads, 2);
            maxVoxelNum = (int)(gridSize.x * gridSize.y * gridSize.z);
            actualVoxelNum = (int)((gridSize.x - 1) * (gridSize.y - 1) * (gridSize.z - 1));
            ChunksSize = new Vector3Int(ChunkSize_X, ChunkSize_Y, ChunkSize_Z);
            ScaleRatio = volumeWidth / gridSize.x;
            chunkNum = ChunksSize.x * ChunksSize.y * ChunksSize.z;
            if (volumeSizeSlider == null)
                return;
            volumeSizeSlider.value = GridResLog2_X;
            volumeChunkSizeSlider.value = minChunkSizeLog2;
            volumeSizeText.text = "Volume Size: " + (int)gridSize.x;
            volumeChunkSizeText.text = "Chunk Size: " + (int)minChunkSize;
            wallMaterial.mainTextureScale = 0.5f * new Vector2(gridSize.x, gridSize.y);
            meshChunkSize = new Vector3(minChunkSize, minChunkSize, minChunkSize);
            meshChunkGroupSize = new Vector3(gridSize.x / minChunkSize, gridSize.y / minChunkSize, gridSize.z / minChunkSize);
            //if (CudaBridge.SetMarchingCubesChunks(chunkNum, minChunkSizeLog2, chunkThreads))
            //    for (int x = 0; x < ChunksSize.x; x++)
            //    {
            //        for (int y = 0; y < ChunksSize.y; y++)
            //        {
            //            for (int z = 0; z < ChunksSize.z; z++)
            //            {
            //                int id = (x * ChunksSize.y + y) * ChunksSize.z + z;
            //                if(id < chunks.Count)
            //                    chunks[id].GetComponent<CudaMarchingCubesChunk>().InitChunk(id, inputMesh);
            //            }
            //        }
            //    }
        }

        void Update()
        {
            //if(Input.GetKeyUp(KeyCode.A))
            //{
            //    CudaBridge.SetMarchingCubesChunks(1, minChunkSizeLog2, chunkThreads);
            //    Debug.Log("Set");
            //}
            //if (Input.GetKeyUp(KeyCode.S))
            //{
            //    CudaBridge.MallocMemoryForMC(0, CudaMarchingCubesChunks.cudaMCManager.GridResLog2,
            //        CudaMarchingCubesChunks.cudaMCManager.volumeWidth);
            //    Debug.Log("Malloc");
            //}
            //if (Input.GetKeyUp(KeyCode.D))
            //{
            //    CudaBridge.FreeMemoryForMC(0);
            //    Debug.Log("Free");
            //}
            statusDebug.text = "Voxels: " + maxVoxelNum + "\r\nTriangles: " + triangleNum;
            if (Input.GetKeyUp(KeyCode.Escape))
            {
                allUI.SetActive(!allUI.activeSelf);
            }

            if (exportTexture3D && createNewSDF)
            {
                exportTexture3D = false;
                createNewSDF = false;
                exportSDF = new Texture3D(voxelSize, voxelSize, voxelSize, TextureFormat.RFloat, false);
                ExportSDF(exportSDF, lastSDF, CudaMarchingCubesChunks.voxelSize, inputMesh.gameObject.name);
            }

            if (createNewSDF)
            {
                createNewSDF = false;
                if (exportNewSdf)
                {
                    ExportBinarySDF();
                }
            }

        }

        public void ExportBinarySDF()
        {
            string path = dirpath + "SDF_" + voxelSize + "_" + inputMesh.gameObject.name + ".bv";
            CudaBridge.SaveSdfData(chunks[0].GetComponent<CudaMarchingCubesChunk>().chunkID, path);
        }

        public void ExportSDF(Texture3D exportSDF, float[] sdfData, int voxelSize, string name)
        {
#if UNITY_EDITOR
            exportSDF.SetPixelData(sdfData, 0);
            AssetDatabase.CreateAsset(exportSDF, "Assets/Large Volume Data Interaction Framework/Textures/SDF_" + voxelSize + "_" + name + ".asset");
#endif
        }
    }
}
