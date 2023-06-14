using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ChaosIkaros.LVDIF
{
    public class CudaManager : MonoBehaviour
    {
        public Text debugPanel;
        public int deviceCount = 0;
        public int deviceID = 0;
        public bool displayErrorInMessageBox = false;
        public bool displayErrorInExtraConsole = true;
        public static bool exist = false;
        public static CudaManager cudaManagerHolder;
        [Tooltip("0 means no debug output; the bigger level, means the more debug content.")]
        [Range(0,10)]
        public int debugLevel = 0;
        public static CudaManager cudaManager
        {
            get
            {
                InitCudaManager();
                return cudaManagerHolder;
            }
        }

        public static void InitCudaManager()
        {
            if (!exist)
            {
                exist = true;
                GameObject managerHolder = GameObject.Find("CudaManager");
                if (managerHolder == null)
                    managerHolder = new GameObject("CudaManager");
                DontDestroyOnLoad(managerHolder);
                if (!managerHolder.GetComponent<CudaManager>())
                    cudaManagerHolder = managerHolder.AddComponent<CudaManager>();
                else
                    cudaManagerHolder = managerHolder.GetComponent<CudaManager>();
                CudaUtility.InitializedCudaDebugCallBack();
            }
        }

        void Awake()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 3000;
            InitCudaManager();
            DontDestroyOnLoad(gameObject);
            CudaUtility.DisplayErrorInMessageBox(displayErrorInMessageBox);
            CudaUtility.DisplayErrorInExtraConsole(displayErrorInExtraConsole);
            //CudaUtility.GetDeviceInfo();
            deviceCount = CudaUtility.GetDeviceNumber();
            if (deviceCount >= 2)
            {
                CudaUtility.SetDevice(1);// set individual GPU for marching cubes pipeline if there are multiple GPUs.
            }

        }

        private void OnValidate()
        {
            CudaUtility.SetDebugLevel(debugLevel);
        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            //if (debugPanel != null)
            //{
            //    if (Input.GetKeyUp(KeyCode.Tab))
            //    {
            //        deviceID++;
            //        if (deviceID >= deviceCount)
            //            deviceID = 0;
            //        CudaUtility.SetDevice(deviceID);
            //        debugPanel.text = "Set cuda device: " + deviceID;
            //    }
            //}
        }
    }
}
