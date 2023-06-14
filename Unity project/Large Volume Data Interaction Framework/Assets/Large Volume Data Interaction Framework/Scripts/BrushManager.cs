using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace ChaosIkaros.LVDIF
{
    public enum InputMethod
    {
        SingleBrush,
        ConstantBrush,
        DynamicBrush,
        DrillBrush
    }

    public enum BrushShape
    {
        Sphere,
        Cube
    }

    public enum BrushState
    {
        Start,
        End
    }

    public class BrushManager : MonoBehaviour
    {
        public AbstractInputDevice inputDevice;
        public static BrushManager holder;
        public bool enableTimer = false;
        public Text sizeText;
        public Slider colorAlphaSlider;
        public RawImage currentColorHint;
        public BrushShape brushShape = BrushShape.Sphere;
        public BrushState brushState = BrushState.End;
        public InputMethod inputMethod = InputMethod.SingleBrush;
        public TMP_Dropdown inputMethodDropdown;
        public GameObject cursor;
        public GameObject cursorSphere;
        public GameObject cursorCube;
        public int realTimeBrushSize = 5;
        public bool realTimeBrush = false;
        public bool fixedUpdateBrush = false;
        public bool adjustableBrushSize = true;
        public bool eraserMode = false;
        [Range(0, 1.0f)]
        public float brushAlpha = 1;
        public float colorBrushOffset = 1.5f;
        public int colorType = 0;
        public List<Color> volumeColors = new List<Color>();
        public float minBrushInterval = 1.6f;
        public float mouseWheelSpeed = 0.5f;
        public float maxRayLength = 1000.0f;
        public float inputRadius = 3;
        public int maxInputRadius = 100;
        public int minInputRadius = 3;
        public Camera brushCamera;
        public static CudaMarchingCubesChunk currentChunk;
        public Vector3 cursorVoxelPos;
        public List<Vector3> cursorPosInBrush = new List<Vector3>();
        public List<Vector4> rawCursorPosInBrush = new List<Vector4>();
        public List<Vector4> finalCursorPosInBrush = new List<Vector4>();
        public string brushStartTime = "none";
        public string brushEndTime = "none";
        int lastInputPosID = 0;
        Color newColor;
        private System.Diagnostics.Stopwatch timerAll;
        private void Awake()
        {
            timerAll = System.Diagnostics.Stopwatch.StartNew();
            holder = this;
            CudaMarchingCubesChunks.cudaMCManager.volumeColors = new List<Color>();
            CudaMarchingCubesChunks.cudaMCManager.volumeColors.AddRange(volumeColors);
            brushState = BrushState.End;
            currentColorHint.color = volumeColors[colorType];
            inputMethodDropdown.ClearOptions();
            List<string> inputMethods = new List<string>();
            inputMethods.AddRange(System.Enum.GetNames(typeof(InputMethod)));
            inputMethodDropdown.AddOptions(inputMethods);
            if (realTimeBrush)
            {
                inputMethod = InputMethod.ConstantBrush;
                inputMethodDropdown.value = 1;
            }
        }
        // Start is called before the first frame update
        void Start()
        {

        }
        private void OnValidate()
        {
            ClampColorType();
            SwitchBrushShape();
        }
        public void SetInputMethod(int input)
        {
            inputMethod = (InputMethod)input;
            cursor.GetComponent<TrailRenderer>().enabled = false;
            cursor.GetComponent<TrailRenderer>().Clear();
        }

        public void SetBrushShape(int input)
        {
            brushShape = (BrushShape)input;
            SwitchBrushShape();
        }

        public void SwitchBrushShape()
        {
            if (brushShape == BrushShape.Sphere)
            {
                cursorSphere.SetActive(true);
                cursor = cursorSphere;
                cursorCube.SetActive(false);
            }
            else if (brushShape == BrushShape.Cube)
            {
                cursorCube.SetActive(true);
                cursor = cursorCube;
                cursorSphere.SetActive(false);
            }
            cursor.GetComponent<TrailRenderer>().enabled = false;
            cursor.GetComponent<TrailRenderer>().Clear();
        }
        public void OnAlphaChange(float alpha)
        {
            newColor = new Color(volumeColors[colorType].r, volumeColors[colorType].g, volumeColors[colorType].b, alpha);
            currentColorHint.color = newColor;
        }
        public void AddNewColor()
        {
            if (!volumeColors.Contains(newColor))
            {
                volumeColors.Add(newColor);
                colorType = volumeColors.Count - 1;
            }
        }
        public void ClampColorType()
        {
            colorType = Mathf.Clamp(colorType, 0, volumeColors.Count - 1);
        }
        public void UpdateColor()
        {
            if (colorType < 0)
                colorType = volumeColors.Count - 1;
            if (colorType > volumeColors.Count - 1)
                colorType = 0;
            ClampColorType();
            currentColorHint.color = volumeColors[colorType];
            colorAlphaSlider.value = currentColorHint.color.a;
        }
        public void ResizeTrial()
        {
            cursor.GetComponent<TrailRenderer>().startWidth = cursor.transform.localScale.x;
            cursor.GetComponent<TrailRenderer>().endWidth = cursor.transform.localScale.x;
        }
        public void UpdateColorHint()
        {
            if (Input.GetKeyDown(KeyCode.Q))
                colorType--;
            if (Input.GetKeyDown(KeyCode.E))
                colorType++;
            if (Input.GetKeyDown(KeyCode.Q) || Input.GetKeyDown(KeyCode.E))
            {
                UpdateColor();
            }
        }
        public int SetMultipleBrush()
        {
            if (cursorPosInBrush.Count == 0)
                return 0;
            int inputStart = 0;
            if (realTimeBrush)
                inputStart = lastInputPosID;
            rawCursorPosInBrush.Clear();
            rawCursorPosInBrush.Add(cursorPosInBrush[inputStart]);
            for (int i = inputStart; i < cursorPosInBrush.Count - 1; i++)
            {
                Vector3 currentCursorPos = cursorPosInBrush[i];
                Vector3 nextCursorPos = cursorPosInBrush[i + 1];
                if (currentCursorPos == nextCursorPos)
                    //|| !currentChunk.IsVaildInput(currentCursorPos) || !currentChunk.IsVaildInput(nextCursorPos))
                    continue;
                float currentInterval = Vector3.Distance(currentCursorPos, nextCursorPos);
                if (currentInterval < minBrushInterval)
                {
                    rawCursorPosInBrush.Add(currentCursorPos);
                }
                else
                {
                    float interpolationRatio = minBrushInterval / currentInterval;
                    float currentRatio = 0;
                    while (currentRatio < 1)
                    {
                        rawCursorPosInBrush.Add(Vector3.Lerp(currentCursorPos, nextCursorPos, currentRatio));
                        currentRatio += interpolationRatio;
                    }
                    rawCursorPosInBrush.Add(nextCursorPos);
                }
            }
            finalCursorPosInBrush.Clear();
            for (int i = 0; i < rawCursorPosInBrush.Count; i++)
            {
                if (i > 0)
                {
                    if (rawCursorPosInBrush[i - 1] == rawCursorPosInBrush[i])
                        continue;
                }
                cursorVoxelPos = rawCursorPosInBrush[i];
                Vector4 temp = cursorVoxelPos;
                temp.w = inputRadius;
                finalCursorPosInBrush.Add(temp);
            }
            if (inputMethod == InputMethod.DynamicBrush)
            {
                //float currentIntervalD = Vector3.Distance(finalCursorPosInBrush[0],
                //    finalCursorPosInBrush[finalCursorPosInBrush.Count - 1]);
                float interpolationRatioD = 1 / (float)finalCursorPosInBrush.Count;
                float currentRatioD = 0;
                for (int i = finalCursorPosInBrush.Count - 1; i > 0; i--)
                {
                    Vector4 temp = finalCursorPosInBrush[i];
                    //temp.w = Mathf.Lerp(minInputRadius, inputRadius, currentRatioD);
                    temp.w = Mathf.Lerp(1, inputRadius, currentRatioD);
                    finalCursorPosInBrush[i] = temp;
                    currentRatioD += interpolationRatioD;
                }
            }
            if (realTimeBrush)
                lastInputPosID = cursorPosInBrush.Count - 1;
            return finalCursorPosInBrush.Count;
        }

        void FixedUpdate()
        {
            if (fixedUpdateBrush && currentChunk != null && brushState == BrushState.Start)
            {
                cursorPosInBrush.Add(currentChunk.GetVoxelPos(cursor.transform.position));
                if (enableTimer)
                {
                    timerAll.Stop();
                    Debug.Log("Fixed Brush: " + timerAll.ElapsedMilliseconds.ToString());
                    timerAll.Restart();
                }
                if (realTimeBrush)
                {
                    cursor.GetComponent<TrailRenderer>().enabled = false;
                    if (cursorPosInBrush.Count != 0 && cursorPosInBrush.Count % realTimeBrushSize == 0)
                    {
                        if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                            brushEndTime = InputRecorder.GetTime();
                        brushState = BrushState.Start;
                        if (!currentChunk.hasInput && SetMultipleBrush() != 0)
                        {
                            if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                                InputRecorder.holder.RecordOneFrame();
                            currentChunk.hasInput = true;
                            CudaMarchingCubesChunks.cudaMCManager.stop = false;
                        }
                    }
                }
            }
        }
        // Update is called once per frame
        void Update()
        {
            UpdateColorHint();
            if (inputDevice.GetMiddleButtonUp() || Input.GetKeyUp(KeyCode.Space))
            {
                eraserMode = !eraserMode;
            }
            if (eraserMode)
            {
                cursor.GetComponent<Renderer>().material.color = new Color(1.0f, 0, 0, 0.5f);
                cursorSphere.GetComponent<Renderer>().material.color = new Color(1.0f, 0, 0, 0.5f);
                cursorCube.GetComponent<Renderer>().material.color = new Color(1.0f, 0, 0, 0.5f);
            }
            else
            {
                cursorSphere.GetComponent<Renderer>().material.color = new Color(0, 1.0f, 0, 0.5f);
                cursorCube.GetComponent<Renderer>().material.color = new Color(0, 1.0f, 0, 0.5f);
                ClampColorType();
                cursor.GetComponent<Renderer>().material.color = volumeColors[colorType];
            }
            if (inputMethod == InputMethod.SingleBrush)
            {
                if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                {
                    brushStartTime = InputRecorder.GetTime();
                    brushEndTime = InputRecorder.GetTime();
                }
                cursor.GetComponent<TrailRenderer>().enabled = false;
                Ray ray = brushCamera.ScreenPointToRay(Input.mousePosition);
                bool isOverUI = UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject();
                if (Physics.Raycast(ray, out RaycastHit hit, maxRayLength) && !isOverUI)
                {
                    if (hit.transform.parent.GetComponent<CudaMarchingCubesChunk>())
                    {
                        currentChunk = hit.transform.parent.GetComponent<CudaMarchingCubesChunk>();
                        cursor.SetActive(true);
                        if (adjustableBrushSize)
                        {
                            if (inputDevice.ScrollWheelValue() > 0)
                                inputRadius += mouseWheelSpeed;
                            if (inputDevice.ScrollWheelValue() < 0)
                                inputRadius -= mouseWheelSpeed;
                        }
                        inputRadius = Mathf.Clamp(inputRadius, minInputRadius, maxInputRadius);
                        cursor.transform.localScale = 2 * inputRadius * currentChunk.unitSize;
                        cursor.transform.position = hit.point;
                        if (inputDevice.GetLeftButtonDown())
                        {
                            cursorVoxelPos = currentChunk.GetVoxelPos(cursor.transform.position);
                            Vector4 temp = cursorVoxelPos;
                            temp.w = inputRadius;
                            finalCursorPosInBrush.Clear();
                            finalCursorPosInBrush.Add(temp);
                            //cursorVoxelPos.x = CudaMarchingCubesChunks.cudaMCManagerHolder.gridSize.x - cursorVoxelPos.x;
                            //cursorVoxelPos.z = CudaMarchingCubesChunks.cudaMCManagerHolder.gridSize.z - cursorVoxelPos.z;
                            if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                                InputRecorder.holder.RecordOneFrame();
                            currentChunk.hasInput = true;
                            CudaMarchingCubesChunks.cudaMCManager.stop = false;
                        }
                    }
                }
                else
                {
                    cursor.SetActive(false);
                }
            }
            else if (inputMethod == InputMethod.DrillBrush || inputMethod == InputMethod.ConstantBrush
                || inputMethod == InputMethod.DynamicBrush)
            {
                if (adjustableBrushSize)
                {
                    if (inputDevice.ScrollWheelValue() > 0)
                        inputRadius += mouseWheelSpeed;
                    if (inputDevice.ScrollWheelValue() < 0)
                        inputRadius -= mouseWheelSpeed;
                }
                if (currentChunk == null)
                {
                    if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count > 0 && CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0] != null)
                        currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();
                    else
                        return;
                }
                cursor.transform.localScale = 2 * inputRadius * currentChunk.unitSize;
                inputRadius = Mathf.Clamp(inputRadius, minInputRadius, maxInputRadius);
                cursor.transform.localScale = 2 * inputRadius * currentChunk.unitSize;
                if (inputMethod == InputMethod.DrillBrush)
                {
                    cursor.SetActive(true);
                    if (inputDevice.InputDeviceType() == InputDevice.Mouse)
                        cursor.transform.position = brushCamera.transform.position + brushCamera.transform.forward;
                    if (inputDevice.InputDeviceType() == InputDevice.VR || inputDevice.InputDeviceType() == InputDevice.Haptic)
                        cursor.transform.localPosition = Vector3.zero;
                }
                else
                {
                    if (inputDevice.InputDeviceType() == InputDevice.Mouse)
                    {
                        Ray ray = brushCamera.ScreenPointToRay(Input.mousePosition);
                        bool isOverUI = UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject();
                        if (Physics.Raycast(ray, out RaycastHit hit, maxRayLength) && !isOverUI)
                        {
                            if (hit.transform.parent.GetComponent<CudaMarchingCubesChunk>())
                            {
                                currentChunk = hit.transform.parent.GetComponent<CudaMarchingCubesChunk>();
                                cursor.SetActive(true);
                                inputRadius = Mathf.Clamp(inputRadius, minInputRadius, maxInputRadius);
                                cursor.transform.localScale = 2 * inputRadius * currentChunk.unitSize;
                                cursor.transform.position = hit.point;
                            }
                        }
                        else
                        {
                            cursor.SetActive(false);
                            return;
                        }
                    }
                    else
                        cursor.transform.localPosition = Vector3.zero;
                }
                if (inputDevice.GetLeftButtonDown() && brushState == BrushState.End)
                {
                    if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                        brushStartTime = InputRecorder.GetTime();
                    inputDevice.AftertLeftButtonDown();
                    if (realTimeBrush)
                        lastInputPosID = 0;
                    cursorPosInBrush.Clear();
                    brushState = BrushState.Start;
                    cursor.GetComponent<TrailRenderer>().enabled = true;
                    cursor.GetComponent<TrailRenderer>().material = cursor.GetComponent<Renderer>().material;
                }
                if (inputDevice.GetLeftButton() && brushState == BrushState.Start)
                {
                    fixedUpdateBrush = true;
                    //cursorPosInBrush.Add(currentChunk.GetVoxelPos(cursor.transform.position));
                    //if (realTimeBrush)
                    //{
                    //    cursor.GetComponent<TrailRenderer>().enabled = false;
                    //    if (cursorPosInBrush.Count != 0 && cursorPosInBrush.Count % realTimeBrushSize == 0)
                    //    {
                    //        if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                    //            brushEndTime = InputRecorder.GetTime();
                    //        brushState = BrushState.Start;
                    //        if (SetMultipleBrush() != 0)
                    //        {
                    //            if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                    //                InputRecorder.holder.RecordOneFrame();
                    //            currentChunk.hasInput = true;
                    //            CudaMarchingCubesChunks.cudaMCManager.stop = false;
                    //        }
                    //    }
                    //}
                }
                if (inputDevice.GetLeftButtonUp() && brushState == BrushState.Start)
                {
                    fixedUpdateBrush = false;
                    if (realTimeBrush)
                    {
                        lastInputPosID = 0;
                        cursorPosInBrush.Clear();
                    }
                    if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                        brushEndTime = InputRecorder.GetTime();
                    cursor.GetComponent<TrailRenderer>().enabled = false;
                    cursor.GetComponent<TrailRenderer>().Clear();
                    brushState = BrushState.End;
                    SetMultipleBrush();
                    if (SetMultipleBrush() != 0)
                    {
                        if (InputRecorder.holder != null && InputRecorder.holder.isRecording)
                            InputRecorder.holder.RecordOneFrame();
                        currentChunk.hasInput = true;
                        CudaMarchingCubesChunks.cudaMCManager.stop = false;
                    }
                }
            }
            if (inputDevice.ScrollWheelValue() != 0)
            {
                sizeText.text = "Size: " + (int)inputRadius;
                ResizeTrial();
                //cursor.GetComponent<TrailRenderer>().widthMultiplier = 
            }
        }
    }
}
