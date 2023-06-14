using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ChaosIkaros.LVDIF;
#if LVDIF_Oculus_VR
using Node = UnityEngine.XR.XRNode;
#endif

public class OculusVRInput : AbstractInputDevice
{
    public override InputDevice InputDeviceType()
    {
        return InputDevice.VR;
    }
    public Transform penEnd;
    public Transform penStart;
    public GameObject penCursor;
    public GameObject penModel;
    // Start is called before the first frame update
    void Start()
    {
        penCursor.GetComponent<MeshRenderer>().sortingOrder = 1;
        BrushManager.holder.inputMethod = InputMethod.DrillBrush;
        BrushManager.holder.inputMethodDropdown.value = 3;
        BrushManager.holder.UpdateColor();
    }

    // Update is called once per frame
    void Update()
    {
        penModel.transform.position = penEnd.position;
        penModel.transform.rotation = Quaternion.LookRotation(penStart.position - penEnd.position, Vector3.up);
        if (BrushManager.holder.cursor.GetComponent<TrailRenderer>().startWidth != BrushManager.holder.cursor.transform.localScale.x)
        {
            BrushManager.holder.sizeText.text = "Size: " + (int)BrushManager.holder.inputRadius;
            BrushManager.holder.ResizeTrial();
            penCursor.transform.localScale = BrushManager.holder.cursor.transform.localScale * 0.7f;
        }
        UpdateColor();
#if LVDIF_Oculus_VR
        if (OVRInput.GetUp(OVRInput.Button.Three))
        {
            BrushManager.holder.SetInputMethod((int)InputMethod.DrillBrush);
            BrushManager.holder.realTimeBrush = true;
        }
        if (OVRInput.GetUp(OVRInput.Button.Four))
        {
            BrushManager.holder.SetInputMethod((int)InputMethod.DynamicBrush);
            BrushManager.holder.realTimeBrush = false;
        }

#endif
    }

    public void UpdateColor()
    {
#if LVDIF_Oculus_VR
        if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickUp))
            BrushManager.holder.colorType--;
        if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickDown))
            BrushManager.holder.colorType++;
        if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickUp) || OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickDown))
        {
            BrushManager.holder.UpdateColor();
        }
#endif
    }
    public override float ScrollWheelValue()
{
#if LVDIF_Oculus_VR
if(OVRInput.GetUp(OVRInput.Button.SecondaryThumbstickUp))
    return 1.0f;
if(OVRInput.GetUp(OVRInput.Button.SecondaryThumbstickDown))
    return -1.0f;
return 0;
#else
        return Input.GetAxis("Mouse ScrollWheel");
#endif
}
public override bool GetMiddleButtonUp()
{
#if LVDIF_Oculus_VR
        return OVRInput.GetUp(OVRInput.Button.PrimaryThumbstick);
#else
        return Input.GetMouseButtonUp(2);
#endif
}

public override bool GetLeftButtonUp()
{
#if LVDIF_Oculus_VR
    return OVRInput.GetUp(OVRInput.Button.PrimaryIndexTrigger);
#else
        return Input.GetMouseButtonUp(0);
#endif
}
public override bool GetRightButtonUp()
{
#if LVDIF_Oculus_VR
        return Input.GetMouseButtonUp(1);
#else
    return Input.GetMouseButtonUp(1);
#endif
}

public override void AftertLeftButtonDown()
{

}

public override bool GetLeftButtonDown()
{
#if LVDIF_Oculus_VR
     return OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger);
#else
        return Input.GetMouseButtonDown(0);
#endif
}
public override bool GetRightButtonDown()
{
#if LVDIF_Oculus_VR
        return Input.GetMouseButtonDown(1);
#else
    return Input.GetMouseButtonDown(1);
#endif
}
public override bool GetLeftButton()
{
#if LVDIF_Oculus_VR
     return OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger);
#else
        return Input.GetMouseButton(0);
#endif
}
public override bool GetRightButton()
{
#if LVDIF_Oculus_VR
        return Input.GetMouseButton(1);
#else
    return Input.GetMouseButton(1);
#endif
}
}

