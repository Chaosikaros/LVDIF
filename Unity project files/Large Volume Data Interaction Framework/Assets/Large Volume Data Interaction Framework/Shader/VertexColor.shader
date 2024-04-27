Shader "Custom/VertexColor"
{
    Properties{
            _Color("Color", Color) = (1,1,1,1)
            _MainTex("Albedo (RGB)", 2D) = "white" {}
            _Glossiness("Smoothness", Range(0,1)) = 0.5
            _Metallic("Metallic", Range(0,1)) = 0.0
            [Toggle(FullShading)]_FullShading("Full Shading", Float) = 1
            [Toggle(WireFrame)]_WireFrame("Wire Frame", Float) = 0
            [Toggle(VertexColor)]_VertexColor("Vertex Color", Float) = 0
            _Radius("Radius", Range(-10, 10)) = 0.01
            _Thickness("Thickness", Range(0, 10)) = 0.02
            _FrontColor("Wireframe Color", color) = (0.0, 0.0, 1.0, 1.0)

            _UnitSize("Unit Size", Vector) = (1,1,1)
            _CenterOffset("Center Offset", Vector) = (1,1,1)
            _VoxelOriginal("Voxel Original", Vector) = (1,1,1)
            _BrushCenterOffset("Brush Center Offset", Vector) = (0,0,0)
            _ColorSmoothing("Color Smoothing", Float) = 0
            _ColorBrushOffset("Color Brush Offset", Float) = 0
    }
        SubShader{
                 //Tags { "Queue" = "Transparent" "RenderType" = "Transparent"}
                 //Tags { "Queue" = "Transparent" "RenderType" = "Opaque" }
                 Tags{"Queue" = "Transparent" "RenderType" = "Transparent"}
                 LOD 200
                 Pass {
                     ColorMask 0
                 }
                 ZWrite Off
                 Blend SrcAlpha OneMinusSrcAlpha
                 ColorMask RGB
                 CGPROGRAM
                 //#pragma surface surf Standard vertex:vert fullforwardshadows
                 #pragma surface surf Standard vertex:vert fullforwardshadows alpha:fade
                 #pragma target 4.0
                 struct Input {
                     float2 uv_MainTex;
                     nointerpolation float3 tA: TEXCOORD1;
                     nointerpolation float3 tB: TEXCOORD2;
                     nointerpolation float3 tC: TEXCOORD3;
                     float2 barycentric : BARYCENTRIC;
                     uint vid : VID;
                 };

                 struct appdata_id
                 {
                     float4 vertex : POSITION;
                     float4 tangent : TANGENT;
                     float3 normal : NORMAL;
                     float4 texcoord : TEXCOORD0;
                     float4 texcoord1 : TEXCOORD1;
                     float4 texcoord2 : TEXCOORD2;
                     float4 texcoord3 : TEXCOORD3;
                     fixed4 color : COLOR;
                     uint vid : SV_VertexID;
                     UNITY_VERTEX_INPUT_INSTANCE_ID
                 };


                 sampler2D _MainTex;

                 float _Radius;
                 half _Glossiness;
                 half _Metallic;
                 fixed4 _Color;
                 fixed4 colorTemp;
                 //fixed4 volumeColorsBuffer[99];
                 float _Thickness;
                 float4 _FrontColor;
                 float3 _UnitSize;
                 float3 _CenterOffset;
                 float3 _VoxelOriginal;
                 float _ColorSmoothing;
                 float _ColorBrushOffset;
                 float3 _BrushCenterOffset;
                 float _WireFrame;
                 float _FullShading;
                 float _VertexColor;
                 //int _TriangleNum;
                 //int _VertexNum;
                 //int _ColorNum;
#ifdef SHADER_API_D3D11
                 StructuredBuffer<int> colorsBuffer;
                 StructuredBuffer<float3> verticesBuffer;
                 StructuredBuffer<float4> volumeColorsBuffer;
#endif
                 //UNITY_INSTANCING_BUFFER_START(Props)
                 //    // put more per-instance properties here
                 //UNITY_INSTANCING_BUFFER_END(Props)
                 void vert(inout appdata_id v, out Input o)
                 {
                     UNITY_INITIALIZE_OUTPUT(Input,o);
                     uint tAID = 0;
                     uint tBID = 0;
                     uint tCID = 0;
                     float det = 0.1f;
                     if (abs((float)v.vid % (float)3 - 0) < det)
                     {
                         tAID = v.vid;
                         tBID = v.vid + 1;
                         tCID = v.vid + 2;
                         o.barycentric = float2(1, 0);
                     } 
                     if (abs((float)v.vid % (float)3 - 1) < det)
                     {
                         tAID = v.vid - 1;
                         tBID = v.vid;
                         tCID = v.vid + 1;
                         o.barycentric = float2(0, 1);
                     }
                     else if (abs((float)v.vid % (float)3 - 2) < det)
                     {
                         tAID = v.vid - 2;
                         tBID = v.vid - 1;
                         tCID = v.vid;
                         o.barycentric = float2(0, 0);
                     }
                     //tAID = clamp(tAID, 0, _VertexNum - 1);
                     //tBID = clamp(tBID, 0, _VertexNum - 1);
                     //tCID = clamp(tCID, 0, _VertexNum - 1);
#ifdef SHADER_API_D3D11
                     o.tA = mul(unity_ObjectToWorld, float4(verticesBuffer[tAID], 1));
                     o.tB = mul(unity_ObjectToWorld, float4(verticesBuffer[tBID], 1));
                     o.tC = mul(unity_ObjectToWorld, float4(verticesBuffer[tCID], 1));
#endif
                     o.vid = v.vid;
                     //o.vid = clamp(o.vid, 0, _VertexNum - 1);
                     //o.vid = 0;
                 }


                 fixed4 lerpColor(float x, float x1, float x2, fixed4 q00, fixed4 q01) {
                     return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
                 }

                 fixed4 biLerpColor(float x, float y, fixed4 q11, fixed4 q12, fixed4 q21, fixed4 q22, float x1, float x2, float y1, float y2) {
                     fixed4 r1 = lerpColor(x, x1, x2, q11, q21);
                     fixed4 r2 = lerpColor(x, x1, x2, q12, q22);
                     return lerpColor(y, y1, y2, r1, r2);
                 }

                 fixed4 triLerpColor(float x, float y, float z,
                     fixed4 q000, fixed4 q001, fixed4 q010, fixed4 q011,
                     fixed4 q100, fixed4 q101, fixed4 q110, fixed4 q111,
                     float x1, float x2, float y1, float y2, float z1, float z2) {
                     fixed4 x00 = lerpColor(x, x1, x2, q000, q100);
                     fixed4 x10 = lerpColor(x, x1, x2, q010, q110);
                     fixed4 x01 = lerpColor(x, x1, x2, q001, q101);
                     fixed4 x11 = lerpColor(x, x1, x2, q011, q111);
                     fixed4 r0 = lerpColor(y, y1, y2, x00, x01);
                     fixed4 r1 = lerpColor(y, y1, y2, x10, x11);
                     return lerpColor(z, z1, z2, r0, r1);
                 }

                 fixed4 tetrahedronInterpolationColor(float x, float y, float z,
                     fixed4 q000, fixed4 q001, fixed4 q010, fixed4 q011,
                     fixed4 q100, fixed4 q101, fixed4 q110, fixed4 q111)
                 {
                     fixed4 result = fixed4(0, 0, 0, 0);
                     if (x > y) {
                         if (y > z) {
                             result = (1 - x) * q000
                                 + (x - y) * q100
                                 + (y - z) * q110
                                 + (z)*q111;
                         }
                         else if (x > z) {
                             result = (1 - x) * q000
                                 + (x - z) * q100
                                 + (z - y) * q101
                                 + (y)*q111;
                         }
                         else {
                             result = (1 - z) * q000
                                 + (z - x) * q001
                                 + (x - y) * q101
                                 + (y)*q111;
                         }
                     }
                     else {
                         if (z > y) {
                             result = (1 - z) * q000
                                 + (z - y) * q001
                                 + (y - x) * q011
                                 + (x)*q111;
                         }
                         else if (z > x) {
                             result = (1 - y) * q000
                                 + (y - z) * q010
                                 + (z - x) * q011
                                 + (x)*q111;
                         }
                         else {
                             result = (1 - y) * q000
                                 + (y - x) * q010
                                 + (x - z) * q110
                                 + (z)*q111;
                         }
                     }
                     return result;
                 }

                 float4 getVolumeColor(Input IN)
                 {
                     // Albedo comes from a texture tinted by color
                     //fixed4 colorA = fixed4(0, 0, 0, 0);
                     //fixed4 colorB = fixed4(0, 0, 0, 0);
                     //fixed4 colorC = fixed4(0, 0, 0, 0);

                     fixed4 cubeVertexColor[8];
                     for (int k = 0; k < 8; k++)
                         cubeVertexColor[k] = fixed4(0, 0, 0, 0);
                     float3 cubeVertex[8];
                     for (int k = 0; k < 8; k++)
                         cubeVertex[k] = 0;
                     float3 cubeCenter;
                     //float3 tA;
                     //float3 tB;
                     //float3 tC;
                     float cubeWidth = 0;
                     float brushDistance;
                     float3 brushVoxelCenter;
                     int brushShape;
                     fixed4 brushColor;
                     fixed4 backgroundColor;
                     float brushDistanceLast;
                     float3 brushVoxelCenterLast;
                     int brushShapeLast;
                     fixed4 brushColorLast;
                     fixed4 backgroundColorLast;

                     int brushID = 999999999;
         #ifdef SHADER_API_D3D11
                     //for (int k = 0; k < 8; k++)
                     //{
                     //    colorA += (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 8 + k]];
                     //    colorB += (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.y / 3) * 8 + k]];
                     //    colorC += (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.z / 3) * 8 + k]];
                     //}
                     //colorA = colorA / 8;
                     //colorB = colorB / 8;
                     //colorC = colorC / 8;
                     //float pA = colorsBuffer[(IN.vid.x / 3) * 9 + 2] / 1000;
                     //float pB = colorsBuffer[(IN.vid.x / 3) * 9 + 5] / 1000;
                     //float pC = colorsBuffer[(IN.vid.x / 3) * 9 + 8] / 1000;

                     //colorA = pA * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 0]] +
                     //    (1 - pA) * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 1]];
                     //colorB = pB * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 3]] +
                     //    (1 - pB) * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 4]];
                     //colorC = pC * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 6]] +
                     //    (1 - pC) * (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid.x / 3) * 9 + 7]];

                     cubeWidth = _UnitSize.x;
                     cubeCenter = _CenterOffset + float3(_UnitSize.x * (float)colorsBuffer[(IN.vid / 3) * 11 + 8],
                         _UnitSize.y * (float)colorsBuffer[(IN.vid / 3) * 11 + 9],
                         _UnitSize.z * (float)colorsBuffer[(IN.vid / 3) * 11 + 10]) + 0.5 * _UnitSize;
                     if (_ColorSmoothing == 0)
                     {
                         //for (int k = 0; k < 8; k++)
                         //{
                         //    cubeVertexColor[k] = (fixed4)volumeColorsBuffer[0];
                         //}
                         for (int k = 0; k < 8; k++)
                         {
                             cubeVertexColor[k] = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 11 + k]];
                             //fixed delta = 0;
                             //if(cubeVertexColor[k].x <= delta && cubeVertexColor[k].y <= delta && cubeVertexColor[k].z <= delta)
                             //    cubeVertexColor[k] = (fixed4)volumeColorsBuffer[0];
                             //cubeVertexColor[k] = (fixed4)volumeColorsBuffer[clamp(colorsBuffer[clamp((IN.vid / 3) * 11 + k, 0 , _TriangleNum * 11 - 1)], 0, _ColorNum - 1)];
                             //cubeVertexColor[k] = (fixed4)volumeColorsBuffer[clamp(colorsBuffer[(IN.vid / 3) * 11 + k], 0 , _ColorNum - 1)];
                             //cubeVertexColor[k] = (fixed4)volumeColorsBuffer[0];
                         }
                         //cubeCenter = mul(unity_WorldToObject, float4(cubeCenter, 1.0));
                         float3 cornerCoords[8];
                         cornerCoords[0] = float3(0, 0, 0);
                         cornerCoords[1] = float3(1, 0, 0);
                         cornerCoords[2] = float3(1, 0, 1);
                         cornerCoords[3] = float3(0, 0, 1);
                         cornerCoords[4] = float3(0, 1, 0);
                         cornerCoords[5] = float3(1, 1, 0);
                         cornerCoords[6] = float3(1, 1, 1);
                         cornerCoords[7] = float3(0, 1, 1);

                         for (int k = 0; k < 8; k++)
                             cubeVertex[k] = cubeCenter + cornerCoords[k] * cubeWidth - 0.5 * _UnitSize;//
                     }
                     else
                     {
                         brushID = colorsBuffer[(IN.vid / 3) * 15 + 14];

                         if (brushID != 999999999)
                         {
                             brushVoxelCenter = float3(colorsBuffer[(IN.vid / 3) * 15 + 0],
                                 colorsBuffer[(IN.vid / 3) * 15 + 1], colorsBuffer[(IN.vid / 3) * 15 + 2]) / 1000;
                             brushDistance = colorsBuffer[(IN.vid / 3) * 15 + 3] / 1000;
                             brushColor = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 15 + 4]];
                             brushShape = colorsBuffer[(IN.vid / 3) * 15 + 5];

                             brushVoxelCenterLast = float3(colorsBuffer[(IN.vid / 3) * 15 + 6],
                                 colorsBuffer[(IN.vid / 3) * 15 + 7], colorsBuffer[(IN.vid / 3) * 15 + 8]) / 1000;
                             brushDistanceLast = colorsBuffer[(IN.vid / 3) * 15 + 9] / 1000;
                             brushColorLast = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 15 + 10]];
                             brushShapeLast = colorsBuffer[(IN.vid / 3) * 15 + 11];

                             backgroundColor = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 15 + 12]];
                             backgroundColorLast = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 15 + 13]];
                         }
                         else
                             brushColor = (fixed4)volumeColorsBuffer[colorsBuffer[(IN.vid / 3) * 15 + 12]];
                     }
                     //tA = verticesBuffer[IN.vid.x];
                     //tB = verticesBuffer[IN.vid.y];
                     //tC = verticesBuffer[IN.vid.z];
                     //tA = mul(unity_ObjectToWorld, float4(tA, 1.0));
                     //tB = mul(unity_ObjectToWorld, float4(tB, 1.0));
                     //tC = mul(unity_ObjectToWorld, float4(tC, 1.0));
         #endif
                     float3 coord = float3(IN.barycentric, 1.0 - IN.barycentric.x - IN.barycentric.y);
                     float3 pixelPosOnTriangle = coord.x * IN.tA + coord.y * IN.tB + coord.z * IN.tC;

                     float3 pixelPosOnTriangleNormalized = pixelPosOnTriangle - cubeVertex[0];
                     pixelPosOnTriangleNormalized = pixelPosOnTriangleNormalized / cubeWidth;
                     float4 averageColor = fixed4(0, 0, 0, 0);

                     if (_ColorSmoothing == 0)
                     {
                         //cornerCoords[0] = float3(0, 0, 0);
                         //cornerCoords[1] = float3(1, 0, 0);
                         //cornerCoords[2] = float3(1, 0, 1);
                         //cornerCoords[3] = float3(0, 0, 1);
                         //cornerCoords[4] = float3(0, 1, 0);
                         //cornerCoords[5] = float3(1, 1, 0);
                         //cornerCoords[6] = float3(1, 1, 1);
                         //cornerCoords[7] = float3(0, 1, 1);

                         //averageColor = triLerpColor(pixelPosOnTriangle.x, pixelPosOnTriangle.y, pixelPosOnTriangle.z,
                         //    cubeVertexColor[0], cubeVertexColor[3], cubeVertexColor[4], cubeVertexColor[7],
                         //    cubeVertexColor[1], cubeVertexColor[2], cubeVertexColor[5], cubeVertexColor[6],
                         //    cubeVertex[0].x, cubeVertex[6].x,
                         //    cubeVertex[0].y, cubeVertex[6].y,
                         //    cubeVertex[0].z, cubeVertex[6].z);

                         if (_VertexColor)
                             averageColor = cubeVertexColor[0];
                         else
                         {
                             //float delta = 0.01;
                             //if(coord.x < delta && coord.y < delta && coord.z < delta)
                                 //averageColor = cubeVertexColor[0];
                             //else
                            averageColor = tetrahedronInterpolationColor(pixelPosOnTriangleNormalized.x,
                                pixelPosOnTriangleNormalized.y, pixelPosOnTriangleNormalized.z,
                                cubeVertexColor[0], cubeVertexColor[3], cubeVertexColor[4], cubeVertexColor[7],
                                cubeVertexColor[1], cubeVertexColor[2], cubeVertexColor[5], cubeVertexColor[6]);
                         }

                         //float pixelCubeVertexDistance[8];
                         //for (int k = 0; k < 8; k++)
                         //    pixelCubeVertexDistance[k] = distance(pixelPosOnTriangle, cubeVertex[k]);

                         //float distanceSum = 0;
                         //for (int k = 0; k < 8; k++)
                         //    distanceSum += pixelCubeVertexDistance[k];
                         //for (int k = 0; k < 8; k++)
                         //    averageColor += (pixelCubeVertexDistance[k] / distanceSum) * cubeVertexColor[k];

                         //if (pixelPosOnTriangleNormalized.z > _Radius)
                         //    averageColor = cubeVertexColor[0];
                         //else
                         //    averageColor = fixed4(0, 0, 0, 0);

                     }
                     else
                     {

                         float3 inputPos;
                         float3 output;
                         if (brushID != 999999999)
                         {
                             //pixelPosOnTriangle = mul(unity_WorldToObject, float4(pixelPosOnTriangle, 1));
                             inputPos = pixelPosOnTriangle - _VoxelOriginal;
                             output = float3(abs(inputPos.z / _UnitSize.z),
                                 abs(inputPos.y / _UnitSize.y),
                                 abs(inputPos.x / _UnitSize.x)) +
                                 float3(_BrushCenterOffset.x * _UnitSize.x,
                                     _BrushCenterOffset.y * _UnitSize.y,
                                     _BrushCenterOffset.z * _UnitSize.z);
                             //float pixelToBrushCenter = 0;
                             float checkLength = 0;
                             float brushDistanceTemp = 0;
                             brushDistanceTemp = distance(output, brushVoxelCenter);
                             if (brushShape == 0)
                             {
                                 checkLength = brushDistanceTemp - brushDistance;
                             }
                             else if (brushShape == 1)
                             {
                                 float3 brushVector = abs(output - brushVoxelCenter);
                                 float3 checkVector = brushVector - float3(brushDistance, brushDistance, brushDistance);
                                 checkLength = length(float3(max(checkVector.x, 0), max(checkVector.y, 0), max(checkVector.z, 0)))
                                     + min(max(checkVector.x, max(checkVector.y, checkVector.z)), 0);
                             }
                             float checkLengthLast = 0;
                             float brushDistanceTempLast = 0;
                             brushDistanceTempLast = distance(output, brushVoxelCenterLast);
                             if (brushShapeLast == 0)
                             {
                                 checkLengthLast = brushDistanceTempLast - brushDistanceLast;
                             }
                             else if (brushShapeLast == 1)
                             {
                                 float3 brushVector = abs(output - brushVoxelCenterLast);
                                 float3 checkVector = brushVector - float3(brushDistanceLast, brushDistanceLast, brushDistanceLast);
                                 checkLengthLast = length(float3(max(checkVector.x, 0), max(checkVector.y, 0), max(checkVector.z, 0)))
                                     + min(max(checkVector.x, max(checkVector.y, checkVector.z)), 0);
                             }

                             if (checkLength > _ColorBrushOffset * 0.5 && checkLengthLast > _ColorBrushOffset * 0.5)
                             {
                                 //if (abs(checkLength) < abs(checkLengthLast))
                                 //{
                                 //    averageColor = backgroundColorLast;
                                 //}
                                 //else
                                 //{
                                 //    if (brushDistanceTemp < brushDistanceTempLast)
                                 //    {
                                 //        averageColor = backgroundColorLast;
                                 //    }
                                 //    else
                                 //        averageColor = backgroundColor;
                                 //}
                                 //if (abs(checkLength) < abs(checkLengthLast))
                                 //    averageColor = backgroundColor * brushDistance / brushDistanceTemp;
                                 //else
                                 //    averageColor = backgroundColorLast * brushDistanceLast / brushDistanceTempLast;

                                 //if (abs(checkLength) > _ColorBrushOffset || abs(checkLengthLast) > _ColorBrushOffset)
                                 //    averageColor = backgroundColorLast;

                                 averageColor = backgroundColorLast;
                             }
                             else
                             if (checkLength < _ColorBrushOffset * 0.5 && checkLengthLast < _ColorBrushOffset * 0.5)
                             {
                                 if (abs(checkLength) < abs(checkLengthLast))
                                 {
                                     averageColor = brushColor;
                                 }
                                 else
                                 {
                                     if (brushDistanceTemp < brushDistanceTempLast)
                                     {
                                         averageColor = brushColor;
                                     }
                                     else
                                         averageColor = brushColorLast;
                                 }
                             }
                             else
                             if (checkLength < _ColorBrushOffset * 0.5 && checkLengthLast > _ColorBrushOffset * 0.5)
                             {
                                 averageColor = brushColor;
                             }
                             else if (checkLength > _ColorBrushOffset * 0.5 && checkLengthLast < _ColorBrushOffset * 0.5)
                             {
                                 averageColor = brushColorLast;
                             }
                             //else if (checkLength < _ColorBrushOffset * 0.5)
                             //{
                             //    averageColor = brushColor;
                             //}
                             //else if (checkLengthLast < _ColorBrushOffset * 0.5)
                             //{
                             //    averageColor = brushColorLast;
                             //}
                             //else if (checkLength < _ColorBrushOffset * 0.5 && checkLengthLast < _ColorBrushOffset * 0.5)
                             //{
                             //    if(checkLength > checkLengthLast)
                             //        averageColor = brushColorLast;
                             //}
                             //else if (checkLength > _ColorBrushOffset * 0.5 && checkLengthLast < _ColorBrushOffset * 0.5)
                             //{
                             //    averageColor = brushColorLast;
                             //}
                             //else if (checkLength < _ColorBrushOffset * 0.5 && checkLengthLast > _ColorBrushOffset * 0.5)
                             //{
                             //    averageColor = brushColor;
                             //}
                             //else
                             //    averageColor = backgroundColorLast;

                             //if (brushDistanceTempLast > _Radius)
                             //    averageColor = fixed4(1, 1, 1, 1);
                             //else
                             //    averageColor = fixed4(0, 0, 0, 0);
                         }
                         else
                             averageColor = brushColor;
                         //float3 triangleCenter = (IN.tA + IN.tB + IN.tC) / 3;
                         //if (distance(output, brushVoxelCenter) > _Radius)
                         //    averageColor = fixed4(1, 1, 1, 1);
                         //else
                         //    averageColor = fixed4(0, 0, 0, 0);
                     }



                     //int minDistanceID = 0;
                     //for (int k = 0; k < 8; k++)
                     //{
                     //    if (pixelCubeVertexDistance[k] < pixelCubeVertexDistance[minDistanceID])
                     //        minDistanceID = k;
                     //}
                     //averageColor = cubeVertexColor[minDistanceID];
                     //float3 triangleCenter = (tA + tB + tC) / 3;
                     //if (distanceSum > _Radius)
                     //    averageColor = cubeVertexColor[0];
                     //else
                     //    averageColor = fixed4(0, 0, 0, 0);

                     //float3 triangleCenter = (IN.tA + IN.tB + IN.tC) / 3;

                     //float3 p = pixelPosOnTriangle;
                     //float area = length(cross((IN.tB - IN.tA), (IN.tC - IN.tA))) * 0.5;
                     //float u = length(cross((IN.tB - p), (IN.tC - p)) * 0.5) / area;
                     //float v = length(cross((IN.tA - p), (IN.tC - p)) * 0.5) / area;
                     //float w = length(cross((IN.tA - p), (IN.tB - p)) * 0.5) / area;

                     ////float3 p1 = float3(1, 0, 0);
                     ////float3 p2 = float3(0, 1, 0);
                     ////float3 p3 = float3(0, 0, 0);
                     ////float3 p = float3(0.5, 0.5, 0);
                     ////float area = length(cross((p2 - p1), (p3 - p1))) * 0.5;
                     ////float u = length(cross((p2 - p), (p3 - p)) * 0.5) / area;
                     ////float v = length(cross((p1 - p), (p3 - p)) * 0.5) / area;
                     ////float w = length(cross((p1 - p), (p2 - p)) * 0.5) / area;

                     //float3 pixelBary = float3(u, v, w);
                     //float distanceToCenter = distance(pixelBary, float3(0,0,0));

                     //float distanceA = distance(pixelPosOnTriangle, IN.tA);
                     //float distanceB = distance(pixelPosOnTriangle, IN.tB);
                     //float distanceC = distance(pixelPosOnTriangle, IN.tC);
                     //float distanceSum = distanceA + distanceB + distanceC; 

                     //float4 averageColor = fixed4(0, 0, 0, 0);
                     ////if (distanceToCenter > _Radius)
                     ////     averageColor += colorA;
                     ////if (distanceA > _Radius)
                     //    averageColor += (distanceA / distanceSum) * colorA;
                     ////if (distanceB > _Radius)
                     //    averageColor += (distanceB / distanceSum) * colorB;
                     ////if (distanceC > _Radius)
                     //    averageColor += (distanceC / distanceSum) * colorC;

                     //coord = smoothstep(fwidth(coord) * 0.1, fwidth(coord) * 0.1 + fwidth(coord), coord);
                     //float4 averageColor = float4(coord.x * colorA + coord.y * colorB + coord.z * colorC);

                     if (_WireFrame)
                     {
                         coord = smoothstep(fwidth(coord) * _Thickness, fwidth(coord) * _Thickness + fwidth(coord), coord); //wireframe pixels
                         averageColor = float4(lerp(_FrontColor, averageColor, min(coord.x, min(coord.y, coord.z)).xxx), 1.0);
                     }

                     return averageColor;
                 }

                 void surf(Input IN, inout SurfaceOutputStandard o)
                 {
                     // Albedo comes from a texture tinted by color
                     colorTemp = fixed4(0, 0, 0, 0);
                     colorTemp = getVolumeColor(IN);
                     //colorTemp = (fixed4)volumeColorsBuffer[0];
                     fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * colorTemp;\
                     o.Albedo = colorTemp.rgb;
                     if (_FullShading)
                     {
                         o.Metallic = _Metallic;
                         o.Smoothness = _Glossiness;
                     }
                     o.Alpha = colorTemp.a;
                 }
                 ENDCG
             }
                 FallBack "Diffuse"
}
