using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ChaosIkaros.LVDIF
{
    public enum InputDevice
    {
        Mouse,
        VR,
        Haptic
    }
    public abstract class AbstractInputDevice : MonoBehaviour
    {
        public abstract InputDevice InputDeviceType();
        public abstract float ScrollWheelValue();

        public abstract bool GetMiddleButtonUp();

        public abstract bool GetLeftButtonUp();

        public abstract bool GetRightButtonUp();

        public abstract void AftertLeftButtonDown();

        public abstract bool GetLeftButtonDown();

        public abstract bool GetRightButtonDown();

        public abstract bool GetLeftButton();

        public abstract bool GetRightButton();
    }
}
