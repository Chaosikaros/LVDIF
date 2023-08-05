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

namespace ChaosIkaros.LVDIF
{
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct uint3
	{
		public uint x;
		public uint y;
		public uint z;
		public uint3(ushort inputX, ushort inputY, ushort inputZ) : this()
		{
			x = inputX;
			y = inputY;
			z = inputZ;
		}
	}

	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort2
	{
		public ushort x;
		public ushort y;

		public ushort2(ushort inputX, ushort inputY) : this()
		{
			x = inputX;
			y = inputY;
		}
	}

	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct float2
	{
		public float x;
		public float y;

		public float2(float inputX, float inputY) : this()
		{
			x = inputX;
			y = inputY;
		}
	}

	public class SDFDicUtil : MonoBehaviour
    {
		public const float MinVoxel = -0.0000001f;
		public const float ClampBound = 4;
		public const float ClampSize = 2 * 4 * 1000 + 2;

        public static List<float2> sdfDictionary1D = new List<float2>();
  
        public static void Initialized()
        {
			int dictionarySize = ushort.MaxValue;
			for (int i = 0; i < dictionarySize; i++)
			{
				sdfDictionary1D.Add(new float2(MinVoxel, 0));
			}
			sdfDictionary1D[0] = new float2(MinVoxel, 0);
			for (int i = 2; i < dictionarySize; i += 2)
			{
				if (i <= ClampSize)
				{
					int clampID = (int)i / 2;
					sdfDictionary1D[Mathf.Clamp(i,2, dictionarySize - 1)] = new float2(clampID * 0.001f, i);
					sdfDictionary1D[Mathf.Clamp(i+1, 2, dictionarySize - 1)] = new float2(-clampID * 0.001f, i + 1);
				}
				else
				{
					int clampID = (int)(i - ClampSize) / 2;
					sdfDictionary1D[Mathf.Clamp(i, 2, dictionarySize - 1)] = new float2(ClampBound + clampID * 0.1f, i);
					sdfDictionary1D[Mathf.Clamp(i+1, 2, dictionarySize - 1)] = new float2(-(ClampBound + clampID * 0.1f), i + 1);
				}
			}
		}
		public static float GetSdfVectorValue(float inputSdf)
		{
			int precisionClose = 3;
			int precisionFar = 1;
			float dist = inputSdf;
			int d = 1;
			if (Mathf.Abs(inputSdf) < ClampBound)
				d = (int)Mathf.Pow(10, precisionClose);
			else
				d = (int)Mathf.Pow(10, precisionFar);
			dist = Mathf.Round(dist * d) / d;
			return dist;
		}

		public static ushort GetSdfVectorKey(float inputSdf)
		{
			float inputSdfAbs = Mathf.Abs(inputSdf);
			int mapSdfDictionaryKey = 0;
			if (inputSdf >= MinVoxel && inputSdf <= 0)
			{
				mapSdfDictionaryKey = 0;
			}
			else if (inputSdfAbs <= ClampBound + 0.002)
			{
				mapSdfDictionaryKey = (int)(2 * inputSdfAbs * 1000);
				if ((inputSdf < 0 && mapSdfDictionaryKey % 2 == 0) || (inputSdf > 0 && mapSdfDictionaryKey % 2 == 1))
					mapSdfDictionaryKey++;
			}
			else
			{
				mapSdfDictionaryKey = (int)(2 * (inputSdfAbs - ClampBound) * 10 + ClampSize);
				if ((inputSdf < 0 && mapSdfDictionaryKey % 2 == 0) || (inputSdf > 0 && mapSdfDictionaryKey % 2 == 1))
					mapSdfDictionaryKey++;
			}
			mapSdfDictionaryKey = Mathf.Clamp(mapSdfDictionaryKey, 0, ushort.MaxValue - 1);
			return (ushort)mapSdfDictionaryKey;
		}
		public static float SampleVolumeColor(ushort2[] data, uint3 p, uint3 gridSize)
		{
			if (p.x == 0 || p.y == 0 || p.z == 0 ||
				p.x >= gridSize.x - 2 || p.y >= gridSize.y - 2 || p.z >= gridSize.z - 2)
				return 0;
			p.x = (uint)Mathf.Min(p.x, gridSize.x);
			p.y = (uint)Mathf.Min(p.y, gridSize.y);
			p.z = (uint)Mathf.Min(p.z, gridSize.z);
			uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
			return sdfDictionary1D[data[i].y].y;
		}

		public static float SampleVolume(ushort2[] data, uint3 p, uint3 gridSize)
		{
			if (p.x == 0 || p.y == 0 || p.z == 0 ||
				p.x == gridSize.x || p.y == gridSize.y || p.z == gridSize.z)
				return MinVoxel;
			p.x = (uint)Mathf.Min(p.x, gridSize.x);
			p.y = (uint)Mathf.Min(p.y, gridSize.y);
			p.z = (uint)Mathf.Min(p.z, gridSize.z);
			uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
			return sdfDictionary1D[data[i].x].x;
		}

		// Start is called before the first frame update
		void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}
