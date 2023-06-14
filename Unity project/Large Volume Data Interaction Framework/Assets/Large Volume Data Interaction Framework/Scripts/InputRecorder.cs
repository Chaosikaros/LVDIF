using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Text;
using TMPro;

namespace ChaosIkaros.LVDIF
{
    public class InputRecorder : MonoBehaviour
    {
        public static Vector4 StringArrayToVector4(string[] v)
        {
            return new Vector4(float.Parse(v[0]), float.Parse(v[1]), float.Parse(v[2]), float.Parse(v[3]));
        }
        public static string Vector4ToString(Vector4 v)
        {
            return v.x.ToString("F3") + "-" + v.y.ToString("F3") + "-" + v.z.ToString("F3") + "-" + v.w.ToString("F3");
        }

        public static string GetTime()
        {
            return "\"" + System.DateTime.Now.ToString("yyyy_MM_dd-hh:mm:ss.ffffff") + "\"";
        }
        public static string GetTimeForFileName()
        {
            return System.DateTime.Now.ToString("yyyy_MM_dd_hh_mm_ss_ffffff");
        }
        public class InputFrame
        {
            public string sourceData = "";
            public string timeStamp = "";
            public int frameID = 0;
            public bool isValid = false;
            public string rawStartTime = "none";
            public string rawEndTime = "none";
            public DateTime startTime;
            public DateTime endTime;
            public int inputMethod = 0;
            public int brushShape = 0;
            public int inputRadius = 0;
            public bool eraserMode = false;
            public int colorType = 0;
            public List<Vector4> cursorVoxelPosList = new List<Vector4>();
            public InputFrame(string source)
            {
                sourceData = source;
                string[] temp = source.Split(',');
                if (temp.Length != 10)
                {
                    Debug.Log("Broken frame: " + source);
                    return;
                }
                timeStamp = temp[0];
                int.TryParse(temp[1], out frameID);
                int.TryParse(temp[2], out inputMethod);
                int.TryParse(temp[3], out brushShape);
                int.TryParse(temp[4], out inputRadius);
                int eraserModeTemp = 0;
                int.TryParse(temp[5], out eraserModeTemp);
                eraserMode = eraserModeTemp == 0 ? false : true;
                int.TryParse(temp[6], out colorType);
                rawStartTime = temp[7].Trim('"');
                if (rawStartTime != "none")
                    startTime = DateTime.ParseExact(rawStartTime.Split('-')[1], "hh:mm:ss.ffffff", System.Globalization.CultureInfo.CurrentCulture);
                rawEndTime = temp[8].Trim('"');
                if (rawEndTime != "none")
                    endTime = DateTime.ParseExact(rawEndTime.Split('-')[1], "hh:mm:ss.ffffff", System.Globalization.CultureInfo.CurrentCulture);
                string[] tempPos = temp[9].Split('_');
                if (tempPos.Length == 0)
                {
                    Debug.Log("Empty input: " + source);
                    return;
                }
                cursorVoxelPosList = new List<Vector4>();
                for (int i = 0; i < tempPos.Length; i++)
                    cursorVoxelPosList.Add(StringArrayToVector4(tempPos[i].Split("-")));

                isValid = true;
            }
        }
        private string recordingTypes = "time stamp,frame ID, input method, brush shape, input radius, eraser mode, color type, start time, end time,cursorVoxelPos\r\n";
        private string analysisTypes = "file ID, total input time, total brush distance, average input time, average brush distance\r\n";
        private List<string> outputData = new List<string>();
        private List<InputFrame> inputFrames = new List<InputFrame>();
        private string dirpath = "";
        private List<FileInfo> fileList;
        private List<string> fileNameList;
        public static InputRecorder holder;
        public GameObject allUI;
        public TMP_Dropdown recordingDropdown;
        public TMP_Text recordingButtonText;
        public Text debugLogger;
        public int maxLength = 1000;
        public int dataCounter = 0;
        public bool isRecording = false;
        public int frameIDCounter = 0;
        // Start is called before the first frame update
        void Start()
        {
            holder = this;
            dirpath = Application.dataPath + "/Large Volume Data Interaction Framework/Recordings/";
            if (!Directory.Exists(dirpath))
            {
                Directory.CreateDirectory(dirpath);
            }
            //outputData.Add(GetTime() + ",1,0,0,0,none,none,32-32-32-10_64-64-64-10\r\n");
            //OutputCSV(dirpath + "Recording_testing.csv", recordingTypes, outputData);
            FileProcessor();
        }

        public void RecordOneFrame()
        {
            if (isRecording)
            {
                string inputFrameTemp = GetTime() + "," + frameIDCounter + ",";
                frameIDCounter++;
                inputFrameTemp += (int)BrushManager.holder.inputMethod + ",";
                inputFrameTemp += (int)BrushManager.holder.brushShape + ",";
                inputFrameTemp += (int)Mathf.Clamp(BrushManager.holder.inputRadius,
                    BrushManager.holder.minInputRadius, BrushManager.holder.maxInputRadius) + ",";
                inputFrameTemp += (BrushManager.holder.eraserMode ? 1 : 0) + ",";
                inputFrameTemp += (int)BrushManager.holder.colorType + ",";
                inputFrameTemp += BrushManager.holder.brushStartTime + ",";
                inputFrameTemp += BrushManager.holder.brushEndTime + ",";
                string inputFramePosList = "";
                for (int i = 0; i < BrushManager.holder.finalCursorPosInBrush.Count; i++)
                {
                    inputFramePosList += Vector4ToString(BrushManager.holder.finalCursorPosInBrush[i]);
                    if (i < BrushManager.holder.finalCursorPosInBrush.Count - 1)
                        inputFramePosList += "_";
                }
                inputFrameTemp += inputFramePosList;
                inputFrameTemp += "\r\n";
                outputData.Add(inputFrameTemp);
            }
        }

        public void StartRecording()
        {
            if (!isRecording)
            {
                isRecording = true;
                debugLogger.text = "Recording";
                recordingButtonText.text = "Stop Recording";
                frameIDCounter = 0;
                outputData.Clear();
            }
            else
                StopRecording();
        }
        public void StopRecording()
        {
            isRecording = false;
            debugLogger.text = "Stop Recording";
            recordingButtonText.text = "Start Recording";
            if (outputData.Count != 0)
            {
                string currentTime = GetTimeForFileName();
                OutputCSV(dirpath + "Recording_" + CudaMarchingCubesChunks.cudaMCManager.gridSize.x + "_" + currentTime + ".csv", recordingTypes, outputData);

                //string path = dirpath + "SDF_" + CudaMarchingCubesChunks.cudaMCManager.gridSize.x + "_" + currentTime + ".bv";
                //CudaBridge.SaveSdfData(CudaMarchingCubesChunks.cudaMCManager.chunks[0].GetComponent<CudaMarchingCubesChunk>().chunkID, path);
            }
            FileProcessor();
        }
        public void FileProcessor()
        {
            DirectoryInfo folder = new DirectoryInfo(dirpath);
            FileInfo[] files = folder.GetFiles();
            fileList = new List<FileInfo>();
            fileNameList = new List<string>();
            for (int i = 0; i < files.Length; i++)
            {
                if (files[i].Name.StartsWith("R") && !files[i].Name.Contains(".meta"))
                {
                    //if (files[i].Name.Split('_').Length == 2)
                    //{
                    //Debug.Log(files[i].Name);
                    fileList.Add(files[i]);
                    fileNameList.Add(files[i].Name);
                    //}
                }
            }
            recordingDropdown.ClearOptions();
            recordingDropdown.AddOptions(fileNameList);
        }
        public void LoadFileByDropdown()
        {
            LoadFile(recordingDropdown.value);
        }

        private void LoadFile(int fileID)
        {
            string fileName = fileList[fileID].Name;
            if (!File.Exists(dirpath + fileName))
            {
                debugLogger.text = "Cannot find the file:" + dirpath + fileName;
                return;
            }
            string[] rawdata = File.ReadAllLines(dirpath + fileName);
            dataCounter = 0;
            maxLength = rawdata.Length;
            inputFrames.Clear();
            List<InputFrame> rawinputFrames = new List<InputFrame>();
            for (int i = 1; i < maxLength; i++)
            {
                //Debug.Log(rawdata[i]);
                rawinputFrames.Add(new InputFrame(rawdata[i]));
            }

            for (int i = 0; i < rawinputFrames.Count; i++)
            {
                if (rawinputFrames[i].isValid)
                {
                    inputFrames.Add(rawinputFrames[i]);
                    dataCounter++;
                }
            }

            //ExportAnalysisReport(fileName);
            StartCoroutine(RecoridngPlayer());
        }

        public void ExportAnalysisReport(string fileName)
        {
            outputData.Clear();
            float averageInputTime = 0;
            float totalInputTime = 0;
            int inputTimeCounter = 0;
            float averageBrushDistance = 0;
            float totalBrushDistance = 0;
            for (int i = 0; i < inputFrames.Count; i++)
            {
                if (inputFrames[i].inputMethod > 0)
                {
                    inputTimeCounter++;
                    totalInputTime += (float)(inputFrames[i].endTime - inputFrames[i].startTime).TotalSeconds;
                    for (int j = 1; j < inputFrames[i].cursorVoxelPosList.Count; j++)
                    {
                        Vector4 tempVector = inputFrames[i].cursorVoxelPosList[j] - inputFrames[i].cursorVoxelPosList[j - 1];
                        totalBrushDistance += (new Vector3(tempVector.x, tempVector.y, tempVector.z)).magnitude;
                    }
                }
            }
            averageInputTime = totalInputTime / inputTimeCounter;
            averageBrushDistance = totalBrushDistance / inputTimeCounter;

            outputData.Add("0" + "," + totalInputTime + "," + totalBrushDistance + "," +
                averageInputTime + "," + averageBrushDistance + "\r\n");
            OutputCSV(dirpath + "Analysis_" + fileName, analysisTypes, outputData);
        }

        public IEnumerator RecoridngPlayer()
        {
            yield return new WaitForSeconds(1);
            debugLogger.text = "Playing recording frame: count " + inputFrames.Count;
            if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            {
                debugLogger.text = "Playing recording frame: has not active chunk";
                yield break;
            }
            if (inputFrames.Count == 0)
            {
                debugLogger.text = "Playing recording frame: has not input frame";
                yield break;
            }
            BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();
            BrushManager.holder.cursor.GetComponent<MeshRenderer>().enabled = false;
            for (int i = 0; i < inputFrames.Count; i++)
            {
                debugLogger.text = "Playing recording frame: " + i;
                BrushManager.holder.eraserMode = inputFrames[i].eraserMode;
                BrushManager.holder.SetInputMethod(inputFrames[i].inputMethod);
                BrushManager.holder.SetBrushShape(inputFrames[i].brushShape);
                BrushManager.holder.inputRadius = Mathf.Clamp(inputFrames[i].inputRadius,
                    BrushManager.holder.minInputRadius, BrushManager.holder.maxInputRadius);
                BrushManager.holder.colorType = inputFrames[i].colorType;
                BrushManager.holder.UpdateColor();
                BrushManager.holder.finalCursorPosInBrush.Clear();
                BrushManager.holder.finalCursorPosInBrush.AddRange(inputFrames[i].cursorVoxelPosList);
                BrushManager.currentChunk.hasInput = true;
                CudaMarchingCubesChunks.cudaMCManager.stop = false;
                yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
            }

            BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
            BrushManager.holder.SetInputMethod((int)InputMethod.SingleBrush);
            BrushManager.holder.eraserMode = false;
            BrushManager.holder.colorType = 0;
            BrushManager.holder.cursor.GetComponent<MeshRenderer>().enabled = true;
        }

        public void OutputCSV(string fileName, string types, List<string> allData)
        {
            if (!File.Exists(fileName))
                File.Create(fileName).Dispose();
            Stream stream = File.OpenWrite(fileName);
            BufferedStream bfs = new BufferedStream(stream);
            bfs.Seek(0, SeekOrigin.Begin);
            bfs.SetLength(0);//clear file
            byte[] buffType = new UTF8Encoding().GetBytes(types);
            bfs.Write(buffType, 0, buffType.Length);
            for (int i = 0; i < allData.Count; i++)
            {
                byte[] buffData = new UTF8Encoding().GetBytes(allData[i]);
                bfs.Write(buffData, 0, buffData.Length);
            }
            bfs.Flush();
            bfs.Close();
            stream.Close();
            Debug.Log("Saved file: " + fileName);
        }
        // Update is called once per frame
        void Update()
        {
            if (Input.GetKeyUp(KeyCode.Escape))
            {
                allUI.SetActive(!allUI.activeSelf);
            }
            //if (Input.GetKeyUp(KeyCode.Space))
            //{
            //    StartRecording();
            //}
        }
    }
}
