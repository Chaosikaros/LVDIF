using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using ChaosIkaros.LVDIF;

public class ProceduralModeling : MonoBehaviour
{
    public Text debugText;
    public bool stopTimer = true;
    // Start is called before the first frame update
    void Start()
    {

    }

    public static void SetInputRadius(float input)
    {
        BrushManager.holder.UpdateColor();
        BrushManager.holder.inputRadius = CudaMarchingCubesChunks.cudaMCManager.gridSize.x * input;
        BrushManager.holder.inputRadius = Mathf.Clamp(BrushManager.holder.inputRadius,
            BrushManager.holder.minInputRadius, BrushManager.holder.maxInputRadius);
        BrushManager.holder.cursor.transform.localScale = 2 * BrushManager.holder.inputRadius * BrushManager.currentChunk.unitSize;
        BrushManager.holder.ResizeTrial();
    }

    public static void SetSingleBrush(Vector3 input)
    {
        BrushManager.holder.cursorVoxelPos = BrushManager.currentChunk.GetVoxelPosInVolume(input);
        Vector4 defaultInputBrush = BrushManager.holder.cursorVoxelPos;
        defaultInputBrush.w = BrushManager.holder.inputRadius;
        BrushManager.holder.finalCursorPosInBrush.Clear();
        BrushManager.holder.finalCursorPosInBrush.Add(defaultInputBrush);
        BrushManager.currentChunk.hasInput = true;
        CudaMarchingCubesChunks.cudaMCManager.stop = false;
    }

    public static void SetMultipleBrush(List<Vector3> inputList)
    {
        if (inputList.Count == 0)
            return;
        if (inputList.Count == 1)
        {
            SetSingleBrush(inputList[0]);
            return;
        }
        for (int i = 0; i < inputList.Count; i++)
            inputList[i] *= CudaMarchingCubesChunks.cudaMCManager.gridSize.x;
        BrushManager.holder.cursorPosInBrush.Clear();
        BrushManager.holder.cursorPosInBrush.AddRange(inputList);
        BrushManager.holder.SetMultipleBrush();
        BrushManager.currentChunk.hasInput = true;
        CudaMarchingCubesChunks.cudaMCManager.stop = false;
    }

    public void ResetCanvas()
    {
        StartCoroutine(ResetCanvasAndInput());
    }

    public IEnumerator ResetCanvasAndInput()
    {
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();
        BrushManager.holder.realTimeBrush = false;
        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 0;
        SetInputRadius(0.5f);
        SetSingleBrush(0.5f * Vector3.one);

        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        float minRadius = BrushManager.holder.minInputRadius / CudaMarchingCubesChunks.cudaMCManager.gridSize.x;
        SetInputRadius(minRadius);

    }

    public void FillCanvas()
    {
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();
        BrushManager.holder.realTimeBrush = false;
        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        SetInputRadius(0.5f);
        SetSingleBrush(0.5f * Vector3.one);
    }

    public List<Vector2> GetCirclePoints(float thetaScale, float radius)
    {
        List<Vector2> outputPoints = new List<Vector2>();
        for (float theta = 0; theta < 2 * Mathf.PI; theta += thetaScale)
            outputPoints.Add(radius * new Vector2(Mathf.Cos(theta), Mathf.Sin(theta)));
        return outputPoints;
    }

    public List<Vector2> GetRectanglePoints(float x, float y, float width, float height, float minSegment = 0)
    {
        List<Vector2> outputPoints = new List<Vector2>();
        outputPoints.Add(new Vector2(x, y));
        if (minSegment != 0 && minSegment > 0)
            for (float i = minSegment; i < width; i += minSegment)
               outputPoints.Add(new Vector2(x + i, y));
        outputPoints.Add(new Vector2(x + width, y));
        if (minSegment != 0 && minSegment > 0)
            for (float i = minSegment; i < height; i += minSegment)
                outputPoints.Add(new Vector2(x + width, y + i));
        outputPoints.Add(new Vector2(x + width, y + height));
        if (minSegment != 0 && minSegment > 0)
            for (float i = minSegment; i < width; i += minSegment)
                outputPoints.Add(new Vector2(x + width - i, y + height));
        outputPoints.Add(new Vector2(x, y + height));
        if (minSegment != 0 && minSegment > 0)
            for (float i = minSegment; i < height; i += minSegment)
                outputPoints.Add(new Vector2(x, y + height - i));

        for (int i = 0; i < outputPoints.Count; i++)
            outputPoints[i] -= 0.5f * new Vector2(width,height);
        return outputPoints;
    }

    public IEnumerator GenerateTaskMapA1()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 3;
        SetInputRadius(0.4f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.3f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.15f);
        SetSingleBrush(new Vector3(0.2f, 0.8f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 1;
        SetInputRadius(0.5f);
        SetSingleBrush(new Vector3(0.5f, 0, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 1;
        SetInputRadius(0.5f);
        SetSingleBrush(new Vector3(1f, 0.5f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateTaskMapA2()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 3;
        SetInputRadius(0.4f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.3f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.15f);
        SetSingleBrush(new Vector3(0.2f, 0.8f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 1;
        SetInputRadius(0.5f);
        SetSingleBrush(new Vector3(0.5f, 0, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 1;
        SetInputRadius(0.5f);
        SetSingleBrush(new Vector3(1f, 0.5f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateTaskMapB1()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        List<Vector3> multipleBrushList = new List<Vector3>();
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.05f);
        multipleBrushList.Clear();
        List<Vector2> circlePoints = GetCirclePoints(0.05f, 0.25f);
        float minSegment = 0.8f * 1 / (float)circlePoints.Count;
        for (int i = 0; i < circlePoints.Count; i++)
            multipleBrushList.Add(new Vector3(Mathf.Clamp(minSegment * (float)i - 0.4f, -0.4f, 0.4f), circlePoints[i].x, 
                circlePoints[i].y) + Vector3.one * 0.5f);
        SetMultipleBrush(multipleBrushList);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.colorType = 3;
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateTaskMapB2()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        List<Vector3> multipleBrushList = new List<Vector3>();
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.05f);
        multipleBrushList.Clear();
        List<Vector2> rectanglePoints = GetRectanglePoints(0f, 0f, 0.5f, 0.5f, 0.01f);
        float minSegment = 0.8f * 1 / (float)rectanglePoints.Count;
        for (int i = 0; i < rectanglePoints.Count; i++)
            multipleBrushList.Add(new Vector3(Mathf.Clamp(minSegment * (float)i - 0.4f, -0.4f, 0.4f), rectanglePoints[i].x,
                rectanglePoints[i].y) + Vector3.one * 0.5f);
        SetMultipleBrush(multipleBrushList);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.colorType = 3;
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateDemo1()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        List<Vector3> multipleBrushList = new List<Vector3>();
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.SetInputMethod((int)InputMethod.DynamicBrush);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        SetInputRadius(0.03f);
        int trailCount = 9;
        for (int j = 0; j < trailCount; j++)
        {
            BrushManager.holder.colorType = j;
            SetInputRadius(0.03f);
            multipleBrushList.Clear();
            List<Vector2> circlePoints = GetCirclePoints(0.05f, 0.25f);
            float minSegment = 0.8f * 1 / (float)circlePoints.Count;
            for (int i = 0; i < circlePoints.Count; i++)
                multipleBrushList.Add(new Vector3(Mathf.Clamp(minSegment * (float)i - 0.4f, -0.4f, 0.4f), circlePoints[i].x,
                    circlePoints[i].y));
            for (int i = 0; i < multipleBrushList.Count; i++)
            {
                Vector3 axis = new Vector3(1, 0, 0);
                float angle = j * 360 / trailCount;
                Quaternion q = Quaternion.AngleAxis(angle, axis);
                multipleBrushList[i] = q * multipleBrushList[i];
                multipleBrushList[i] += Vector3.one * 0.5f;
            }
            SetMultipleBrush(multipleBrushList);
            yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
        }

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.colorType = 3;
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateDemo2()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 1;
        SetInputRadius(0.4f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.38f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.colorType = 4;
        SetInputRadius(0.1f);

        //left face has 1 point
        SetSingleBrush(new Vector3(0.1f, 0.5f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //top face has 5 points
        SetSingleBrush(new Vector3(0.5f, 0.9f, 0.5f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.3f, 0.9f, 0.3f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.3f, 0.9f, 0.7f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.7f, 0.9f, 0.7f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.7f, 0.9f, 0.3f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //right face has 3 points
        SetSingleBrush(new Vector3(0.5f, 0.5f, 0.9f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.3f, 0.3f, 0.9f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        SetSingleBrush(new Vector3(0.7f, 0.7f, 0.9f));
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        SetInputRadius(0.05f);
    }

    public IEnumerator GenerateDemoFigure()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        BrushManager.holder.SetBrushShape((int)BrushShape.Cube);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        SetInputRadius(0.5f * 0.7812f);
        SetSingleBrush(0.5f * Vector3.one);
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        //BrushManager.holder.realTimeBrush = true;
        BrushManager.holder.eraserMode = true;
        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.SetInputMethod((int)InputMethod.ConstantBrush);
        SetInputRadius(0.05f);
    }

    public IEnumerator TimerLoop()
    {
        Queue<float> frameTime = new Queue<float> { };
        while (!stopTimer)
        {
            yield return new WaitForSeconds(0.1f);
            frameTime.Enqueue(CudaMarchingCubesChunks.globalFPS);
            string result = "Benchmark: " + frameTime.ToArray().Average().ToString("F2");
            //Debug.Log(result);
            if (debugText != null)
                debugText.text = result;
        }
    }

    public IEnumerator BenchmarkNone()
    {
        stopTimer = false;
        StartCoroutine(TimerLoop());
        yield return new WaitForSeconds(5);
        stopTimer = true;
    }

    public IEnumerator BenchmarkInput()
    {
        yield return new WaitForSeconds(1);
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        BrushManager.holder.colorType = 0;
        float minRadius = BrushManager.holder.minInputRadius / CudaMarchingCubesChunks.cudaMCManager.gridSize.x;
        SetInputRadius(minRadius);

        stopTimer = false;
        StartCoroutine(TimerLoop());
        int testCount = (int) CudaMarchingCubesChunks.cudaMCManager.gridSize.x / BrushManager.holder.minInputRadius;
        for (int i = 0; i < testCount; i++)
        {
            SetSingleBrush(i * Vector3.one * minRadius);
            yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
        }
        stopTimer = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count != 0 
            && BrushManager.currentChunk != null && !BrushManager.currentChunk.hasInput)
        {
            if (Input.GetKeyUp(KeyCode.F1))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateTaskMapA1());
            }
            else if (Input.GetKeyUp(KeyCode.F2))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateTaskMapA2());
            }
            else if (Input.GetKeyUp(KeyCode.F3))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateTaskMapB1());
            }
            else if (Input.GetKeyUp(KeyCode.F4))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateTaskMapB2());
            }
            else if (Input.GetKeyUp(KeyCode.F5))
            {
                StopAllCoroutines();
                ResetCanvas();
            }
            else if (Input.GetKeyUp(KeyCode.F6))
            {
                StopAllCoroutines();
                FillCanvas();
            }
            else if (Input.GetKeyUp(KeyCode.F7))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateDemo1());
            }
            else if (Input.GetKeyUp(KeyCode.F8))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateDemo2());
            }
            else if (Input.GetKeyUp(KeyCode.F9))
            {
                StopAllCoroutines();
                StartCoroutine(BenchmarkNone());
            }
            else if (Input.GetKeyUp(KeyCode.F10))
            {
                StopAllCoroutines();
                StartCoroutine(BenchmarkInput());
            }
            else if (Input.GetKeyUp(KeyCode.F11))
            {
                StopAllCoroutines();
                StartCoroutine(GenerateDemoFigure());
            }
        }
    }
}
