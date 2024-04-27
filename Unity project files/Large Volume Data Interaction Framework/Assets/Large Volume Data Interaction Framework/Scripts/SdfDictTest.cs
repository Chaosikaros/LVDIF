using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SdfDictTest : MonoBehaviour
{
	float MinVoxel = -0.0000001f;
	float ClampBound = 4;
	float ClampSize = 2 * 4 * 1000 + 2;
	public List<Vector2> sdfDictionary1D = new List<Vector2>();
	public List<Vector2> sdfDictionary1DKey = new List<Vector2>();
	// Start is called before the first frame update
	void Start()
    {
		int dictionarySize = 65536;
		for (int i = 0; i < dictionarySize; i++)
		{
			sdfDictionary1D.Add(new Vector2(MinVoxel, 0));
			sdfDictionary1DKey.Add(new Vector2(MinVoxel, 0));
		}
		sdfDictionary1D[0] = new Vector2(MinVoxel, 0);
		sdfDictionary1D[1] = new Vector2(0, 0);
		for (int i = 2; i < dictionarySize; i += 2)
		{
			if (i <= ClampSize)
			{
				int clampID = (int)i / 2;
				sdfDictionary1D[i] = new Vector2(clampID * 0.001f, i);
				sdfDictionary1D[i + 1] = new Vector2(-clampID * 0.001f, i + 1);
			}
			else
			{
				int clampID = (int) (i - ClampSize) / 2;
				sdfDictionary1D[i] = new Vector2(ClampBound + clampID * 0.1f, i);
				sdfDictionary1D[i + 1] = new Vector2(-(ClampBound + clampID * 0.1f), i + 1);
			}
		}
		for (int i = 0; i < dictionarySize; i++)
		{
			float inputSdf = sdfDictionary1D[i].x;
			float absInputSdf = Mathf.Abs(inputSdf);
			int mapSdfDictionaryKey = 0;
			if (inputSdf >= MinVoxel && inputSdf <= 0)
			{
				mapSdfDictionaryKey = 0;
			}
			else if (absInputSdf <= ClampBound + 0.002)
			{
				if (inputSdf > 0)
					mapSdfDictionaryKey = (int)(2 * absInputSdf * 1000);
				else
					mapSdfDictionaryKey = (int)(2 * absInputSdf * 1000 + 1);
			}
			else
			{
				if (inputSdf > 0)
					mapSdfDictionaryKey = (int)(2 * (absInputSdf - ClampBound)* 10 + ClampSize + 1);
				else
					mapSdfDictionaryKey = (int)(2 * (absInputSdf - ClampBound) * 10 + ClampSize + 2);
			}
			sdfDictionary1DKey[i] = new Vector2(mapSdfDictionaryKey, inputSdf);
		}	
	}

    // Update is called once per frame
    void Update()
    {
        
    }
}
