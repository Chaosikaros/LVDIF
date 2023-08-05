using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using ChaosIkaros.LVDIF;

public class ImageToVolume : MonoBehaviour
{
    public ProceduralModeling proceduralModeling;
    public int volumeGridSize;
    public Texture2D inputImage;
    public Texture2D inputDepthImage;
    public Texture2D inputDepthImageResize;
    public Texture2D clampedImage;
    public Texture2D clampedImageResize;
    public Texture2D gridImage;
    List<ushort2> ushort2SDF;
    List<Color> imageColors;
    bool clampedColor = false;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    public void DrawImage()
    {
        CudaMarchingCubesChunks.cudaMCManagerHolder.loadSDFFromUnity = true;
        CudaMarchingCubesChunks.cudaMCManagerHolder.GridResLog2_X = (int)Mathf.Log(Mathf.Clamp(volumeGridSize,64,512), 2);
        CudaMarchingCubesChunks.cudaMCManagerHolder.ClampInputParameters();
        volumeGridSize = (int)CudaMarchingCubesChunks.cudaMCManagerHolder.gridSize.x;
        ClampImage();
        SetColorMap(gridImage);
        StartCoroutine(ImageToSDF(gridImage));
    }

    public void SetColorMap(Texture2D texture2D)
    {
        clampedColor = false;
        imageColors = new List<Color>() { };

        for (int y = 0; y < texture2D.height; y++)
        {
            for (int x = 0; x < texture2D.width; x++)
            {
                Color rawColor = texture2D.GetPixel(x, y);
                imageColors.Add(RemoveAlpha(rawColor));
            }
        }
        imageColors = imageColors.Distinct().ToList();
        //if (imageColors.Count > ushort.MaxValue)
        //{
        //    clampedColor = true;
        //    Debug.Log("Compressed color");

        //    for (int i = 0; i < imageColors.Count; i++)
        //    {
        //        imageColors[i] = ClampColor(imageColors[i]);
        //    }
        //    imageColors = imageColors.Distinct().ToList();
        //    while (imageColors.Count > ushort.MaxValue)
        //    {
        //        imageColors.RemoveAt(0);
        //    }
        //}

        //if (imageColors.Count > ushort.MaxValue)
        //{
        //    clampedColor = true;
        //    Debug.Log("Compressed color");
        //    Dictionary<Color,float> colorDistances = new Dictionary<Color,float> { }; 
        //    //float minDistance = 0.1f;
        //    Color temp = imageColors[0];
        //    for (int i = 1; i < imageColors.Count; i++)
        //    {
        //        Vector3 distance = new Vector3(temp.r - imageColors[i].r, temp.g - imageColors[i].g, temp.b - imageColors[i].b);
        //        colorDistances.Add(imageColors[i],distance.magnitude);
        //    }
        //    colorDistances.OrderBy(pair => pair.Value);
        //    imageColors = new List<Color>(colorDistances.Keys);
        //    while (imageColors.Count > ushort.MaxValue)
        //    {
        //        imageColors.RemoveAt(0);
        //    }
        //}
        Debug.Log("Colors: " + imageColors.Count);
        BrushManager.holder.ResetColors(imageColors);
    }

    public Color RemoveAlpha(Color imageColor)
    {
        return new Color(imageColor.r,
                    imageColor.g ,
                    imageColor.b , 1);
    }
    public Color ClampColor(Color imageColor)
    { 
        //int maxColor = 16;
        float maxColor = 40;
        return new Color((float)(imageColor.r * maxColor) / maxColor,
                    (float)(imageColor.g * maxColor) / maxColor,
                    (float)(imageColor.b * maxColor) / maxColor, imageColor.a);
    }



    public IEnumerator ImageToSDF(Texture2D texture2D)
    {
        //proceduralModeling.ResetCanvas();
        //yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
        //if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
        //    yield break;

        SDFDicUtil.Initialized();
        ushort2SDF = new List<ushort2> { };
        Vector3 gridSize = CudaMarchingCubesChunks.cudaMCManagerHolder.gridSize;
        for (int x = 0; x < gridSize.x; x++)
        {
            for (int y = 0; y < gridSize.y; y++)
            {
                for (int z = 0; z < gridSize.z; z++)
                {
                    ushort2SDF.Add(new ushort2(SDFDicUtil.GetSdfVectorKey(-1.0f), 0));
                }
            }
        }
        int currentColorID = 0;
        int lastColorID = 0;
        float scale = 0.5f;
        Color currentColor = Color.black;
        Vector3 currentVoxel = Vector3.zero;
        int thickness = 1;
        for (int y = 0; y < texture2D.height; y++)
        {
            for (int x = 0; x < texture2D.width; x++)
            {
                currentColor = texture2D.GetPixel(x, y);
                //if (clampedColor)
                //    currentColorID = BrushManager.holder.volumeColors.IndexOf(ClampColor(RemoveAlpha(currentColor)));
                //else
                currentColorID = BrushManager.holder.volumeColors.IndexOf(RemoveAlpha(currentColor));
                if (currentColorID > ushort.MaxValue)
                    currentColorID = lastColorID;
                lastColorID = currentColorID;
                //currentVoxel = new Vector3(x, y, currentColor.a * scale * texture2D.height);
                currentVoxel = new Vector3(x, y, scale * texture2D.height);

                for (int i = 0; i < thickness; i++)
                { 
                    int id = (int) ((x * gridSize.y + y) * gridSize.z + currentVoxel.z + i);
                    ushort2SDF[Mathf.Clamp(id, 0, ushort2SDF.Count - 1)] = new ushort2(SDFDicUtil.GetSdfVectorKey(1.0f), (ushort)currentColorID);
                }
            }
        }
        CudaMarchingCubesChunks.ushort2SDF = ushort2SDF.ToArray();

        CudaMarchingCubesChunks.cudaMCManagerHolder.switchModeDropdown.value = 1;
        CudaMarchingCubesChunks.cudaMCManagerHolder.SwitchMode(1);
        CudaMarchingCubesChunks.cudaMCManagerHolder.InitChunk();

        yield return new WaitForSeconds(1);
        // CudaMarchingCubesChunks.cudaMCManagerHolder.InitChunk();
        //yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
    }

    public IEnumerator ImageToBrush(Texture2D texture2D)
    {
        proceduralModeling.ResetCanvas();
        yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);

        if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count == 0)
            yield break;
        BrushManager.currentChunk = CudaMarchingCubesChunks.cudaMCManagerHolder.chunks[0].GetComponent<CudaMarchingCubesChunk>();

        BrushManager.holder.SetBrushShape((int)BrushShape.Sphere);
        BrushManager.holder.eraserMode = false;
        BrushManager.holder.colorType = 0;
        float minRadius = BrushManager.holder.minInputRadius / CudaMarchingCubesChunks.cudaMCManager.gridSize.x;
        ProceduralModeling.SetInputRadius(minRadius);

        int currentColorID = 0;
        float scale = 0.5f;
        Color currentColor = Color.black;
        Vector3 currentBrush = Vector3.zero;
        for (int y = 0; y < texture2D.height; y++)
        {
            for (int x = 0; x < texture2D.width; x++)
            {
                currentColor = texture2D.GetPixel(x, y);
                //if (clampedColor)
                //    currentColorID = BrushManager.holder.volumeColors.IndexOf(ClampColor(RemoveAlpha(currentColor)));
                //else
                currentColorID = BrushManager.holder.volumeColors.IndexOf(RemoveAlpha(currentColor));
                BrushManager.holder.colorType = currentColorID;
                currentBrush = new Vector3((float)x / (float)texture2D.width, (float)y / (float)texture2D.height,
                    currentColor.a * scale);
                Debug.Log(currentBrush);
                ProceduralModeling.SetSingleBrush(currentBrush);
                yield return new WaitUntil(() => !BrushManager.currentChunk.hasInput);
            }
        }
    }

    public void ClampImage()
    {
        CopyFrom(ref clampedImage, inputImage);
        Resize(ref inputDepthImageResize, inputDepthImage, inputImage.width, inputImage.height, TextureFormat.R16);
        DepthToAlpha(ref clampedImage, inputDepthImageResize);
        gridImage = new Texture2D(volumeGridSize, volumeGridSize);

        Vector2Int newSize = Vector2Int.one;
        if (inputImage.width > inputImage.height)
            newSize = new Vector2Int(volumeGridSize, Mathf.RoundToInt(volumeGridSize * inputImage.height/ inputImage.width));
        else
            newSize = new Vector2Int(Mathf.RoundToInt(volumeGridSize * inputImage.width / inputImage.height), volumeGridSize);
        Resize(ref clampedImageResize, clampedImage, newSize.x, newSize.y, TextureFormat.RGBA32);
        ClampToCenter(ref gridImage, clampedImageResize);
        TextureToPng(gridImage, Application.dataPath + "/Large Volume Data Interaction Framework/Textures/ClampImage.png");
    }
    void CopyFrom(ref Texture2D result, Texture2D texture2D)
    {
        result = new Texture2D(texture2D.width, texture2D.height);
        result.SetPixels32(texture2D.GetPixels32());
        result.Apply();
    }

    void Resize(ref Texture2D result,Texture2D texture2D, int targetX, int targetY, TextureFormat textureFormat)
    {
        result = new Texture2D(targetX, targetY, textureFormat, true);
        Graphics.ConvertTexture(texture2D, result);
        result.ReadPixels(new Rect(0, 0, result.width, result.height), 0, 0, true);
        result.Apply();
    }

    void DepthToAlpha(ref Texture2D result, Texture2D texture2D)
    {
        //unsafe
        //{
        //var rawBytes = texture2D.GetRawTextureData<byte>();
        //byte* pointer = (byte*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(rawBytes);
        //var ushorts = MemoryMarshal.Cast<byte, ushort>(new(pointer, rawBytes.Length));
        //}
        //[y * result.width + x]
        for (int y = 0; y < result.height; y++)
        {
            for (int x = 0; x < result.width; x++)
            {
                Color rawColor = result.GetPixel(x, y);
                rawColor.a = texture2D.GetPixel(x, y).r;
                result.SetPixel(x, y, rawColor);
            }
        }
        result.Apply();
  
    }

    public void ClampToCenter(ref Texture2D result, Texture2D texture2D)
    {
        for (int y = 0; y < result.height; y++)
        {
            for (int x = 0; x < result.width; x++)
            {
                Color rawColor = Color.black;
                rawColor.a = 0;
                result.SetPixel(x, y, rawColor);
            }
        }
        for (int y = 0; y < texture2D.height; y++)
        {
            for (int x = 0; x < texture2D.width; x++)
            {
                Color rawColor = texture2D.GetPixel(x, y);
                Vector2Int resamplePos = new Vector2Int(x,y);
                if (texture2D.width > texture2D.height)
                    resamplePos.y += Mathf.RoundToInt(0.5f * (result.width- texture2D.height));
                else
                    resamplePos.x += Mathf.RoundToInt(0.5f * (result.width - texture2D.width));
                result.SetPixel(Mathf.Clamp(resamplePos.x, 0, result.width-1), 
                    Mathf.Clamp(resamplePos.y, 0, result.height - 1), rawColor);
            }
        }
        result.Apply();
    }

    void TextureToPng(Texture2D texture2D, string path)
    {
        var png = texture2D.EncodeToPNG();
        File.WriteAllBytes(path, png);
    }

    // Update is called once per frame
    void Update()
    {
        volumeGridSize = (int)CudaMarchingCubesChunks.cudaMCManagerHolder.gridSize.x;
        //if (CudaMarchingCubesChunks.cudaMCManagerHolder.chunks.Count != 0
        //    && BrushManager.currentChunk != null && !BrushManager.currentChunk.hasInput)
        //{
        //if (Input.GetKeyUp(KeyCode.Space))
        //    {
        //        DrawImage();
        //    }
        //}
    }
}
