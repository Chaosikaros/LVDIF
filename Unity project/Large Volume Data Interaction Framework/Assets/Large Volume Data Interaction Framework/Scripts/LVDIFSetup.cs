using System.Collections;
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;

namespace ChaosIkaros.LVDIF
{
#if UNITY_EDITOR
    public class LVDIFSetup : EditorWindow
    {
        const string VRSymbol = "LVDIF_Oculus_VR";
        bool allowUnSafeCode = false;
        bool enableDX11 = false;
        bool setUpVR = false;
        [MenuItem("Tools/LVDIF/Set up")]
        private static void LVDIFSetUpWindow()
        {
            EditorWindow.GetWindow(typeof(LVDIFSetup), false, "LVDIF Setup Panel");
        }
        void OnGUI()
        {
            GUILayout.Label("Setup Common Player Settings", EditorStyles.boldLabel);
            allowUnSafeCode = PlayerSettings.allowUnsafeCode;
            GraphicsDeviceType[] graphicsDeviceTypes = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows);

            enableDX11 = !PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows) &&
                graphicsDeviceTypes.Length == 1 && graphicsDeviceTypes[0] == GraphicsDeviceType.Direct3D11;
            allowUnSafeCode = EditorGUILayout.Toggle("Allow unsafe code", allowUnSafeCode);
            enableDX11 = EditorGUILayout.Toggle("Set Graphics API to DX11", enableDX11);
            if (GUILayout.Button("Set up"))
            {
                SetUp();
            }

            GUILayout.Label("Optional Setup", EditorStyles.boldLabel);
            setUpVR = false;
            string[] symbols = PlayerSettings.GetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone).Split(';');
            for (int i = 0; i < symbols.Length; i++)
            {
                if (symbols[i] == VRSymbol)
                    setUpVR = true;
            }
            EditorGUILayout.Toggle("Set up VR", setUpVR);
            if (GUILayout.Button("Set up VR"))
            {
                SetUpVR();
            }

            GUILayout.Label("Reset to Default", EditorStyles.boldLabel);
            if (GUILayout.Button("Reset all"))
            {
                ResetAll();
            }
        }

        void SetUp()
        {
            PlayerSettings.allowUnsafeCode = true;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows, new GraphicsDeviceType[] { GraphicsDeviceType.Direct3D11 });
            GraphicsDeviceType[] graphicsDeviceTypes = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows);

            allowUnSafeCode = PlayerSettings.allowUnsafeCode;
            enableDX11 = !PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows) &&
                graphicsDeviceTypes.Length == 1 && graphicsDeviceTypes[0] == GraphicsDeviceType.Direct3D11;
        }

        void SetUpVR()
        {
            string[] symbols = PlayerSettings.GetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone).Split(';');
            for (int i = 0; i < symbols.Length; i++)
            {
                if (symbols[i] == VRSymbol)
                    setUpVR = true;
            }
            if (!setUpVR)
                PlayerSettings.SetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone,
                    PlayerSettings.GetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone) + ";" + VRSymbol);
        }

        void ResetAll()
        {
            allowUnSafeCode = false;
            enableDX11 = false;
            setUpVR = false;

            PlayerSettings.allowUnsafeCode = false;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows, true);

            string[] symbols = PlayerSettings.GetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone).Split(';');
            for (int i = 0; i < symbols.Length; i++)
            {
                if (symbols[i] == VRSymbol)
                    symbols[i] = "";
            }
            string newSymbols = "";
            for (int i = 0; i < symbols.Length; i++)
            {
                if (symbols[i] != "")
                    newSymbols += symbols[i] + ";";
            }
            if (newSymbols.Contains(';'))
                newSymbols = newSymbols.Remove(newSymbols.Length - 1);
            PlayerSettings.SetScriptingDefineSymbols(UnityEditor.Build.NamedBuildTarget.Standalone, newSymbols);
        }
    }
#endif
}
