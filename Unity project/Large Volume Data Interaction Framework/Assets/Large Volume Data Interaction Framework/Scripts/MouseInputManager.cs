using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ChaosIkaros.LVDIF;

public class MouseInputManager : AbstractInputDevice
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public override InputDevice InputDeviceType()
    {
        return InputDevice.Mouse;
    }
    public override float ScrollWheelValue()
    {
        return Input.GetAxis("Mouse ScrollWheel");
    }
    public override bool GetMiddleButtonUp()
    {
        return Input.GetMouseButtonUp(2);
    }

    public override bool GetLeftButtonUp()
    {
        return Input.GetMouseButtonUp(0);
    }
    public override bool GetRightButtonUp()
    {
        return Input.GetMouseButtonUp(1);
    }
    public override void AftertLeftButtonDown()
    {     
    }
    public override bool GetLeftButtonDown()
    {
        return Input.GetMouseButtonDown(0);
    }
    public override bool GetRightButtonDown()
    {
        return Input.GetMouseButtonDown(1);
    }
    public override bool GetLeftButton()
    {
        return Input.GetMouseButton(0);
    }
    public override bool GetRightButton()
    {
        return Input.GetMouseButton(1);
    }
}
