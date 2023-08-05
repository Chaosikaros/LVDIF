# Large Volume Data Interaction Framework

### [Video](https://youtu.be/0G11HCm5MvA) | [Paper](https://link.springer.com/article/10.1007/s00371-023-02976-x)

Large Volume Data Interaction Framework (LVDIF) is an open-source interaction framework for real-time interact with large volume data. It is integrated into Unity3D to benefit from interaction scenarios in game engine. LVDIF contains 2 main components: a CUDA library that provides volume data container, mesh to SDF algorithm, and volume rendering pipeline; an interaction framework that provides input device interaction and volume interaction interface in Unity3D.

## Features

- A robust and efficient approach for the sign calculation of large SDF volume generation from triangle mesh.
- A GPU-based strategy for optimization of interactive 1024<sup>3</sup> RGBA SDF volume.
- Cross-device input support (mouse, VR controller, etc.)
- Scripting language for procedural modelling.

![](images/image1.jpeg)

LVDIF supports interactive editing of large volume data with cross-device input support. This figure shows some interactive editing examples using a mouse-based brush on a 1024<sup>3</sup> RGBA SDF volume of the Stanford bunny. (a) basic union and subtraction operations of the 200<sup>3</sup> cube and sphere primitives; (b) brush input with a radius of 15 for painting and erasing using a mouse: (from top to bottom): basic brush, dynamic brush, and drill brush; (c) subtraction on a union of three 200<sup>3</sup> sphere primitives with different colors; and (d) a painting brush example in a radius of 3 on a 200<sup>3</sup> cube primitive.

## Installation & Set-up

System requirement: Windows

Unity3D: 2020.1 and above (The sample project is built in 2021.2.3f1)

The DLL project is built in Visual Studio 2019. The Unity project contains the prebuilt DLL.

1. Install NVIDIA CUDA toolkit 11.0 or above.
2. Import the Unity package or open the project file.
3.  Enable "Allow 'unsafe' code" (for Native Array) in Unity Player Settings.
4. Click the set up button in editor window: Tools -> LVDIF -> Set up.

## Getting Started Guide

The example scenes are in Examples folder (also see this [video](https://youtu.be/0G11HCm5MvA)).
The demo applications contain RGBA SDF volume files of a 512<sup>3</sup> horse and a 128 <sup>3</sup> dragon.

### Mouse&Keyboard (available demo: "LVDIF Mouse&Keyboard.7z" in release page)

Scene: Mouse example.unity

![](images/image2.png)

This scene contains the main functions of LVDIF.

Instruction: UI on/off: Esc; Camera movement: WASD and right mouse button; Switch color: Q and E. 

Input (Load or generate a volume before input): left mouse button; Eraser on/off: middle mouse button; Change Brush size: mouse wheel.

Mesh to SDF for 1024<sup>3</sup> volume requires about 8 GB GPU memory.

### Optional set-up for VR (available demo: "LVDIF Oculus VR.7z" in release page)

Scene: Oculus rift s VR example.unity

Click the set up VR button in editor window: Tools -> LVDIF -> Set up VR

The VR example needs the Oculus Integration from Unity Asset Store.

The VR input is defined in OculusVRInput.cs:

Oculus VR input instruction: Switch color: left controller thumb stick. Switch input method: left controller X and Y. Move the pen: right controller trigger.

Input (Load or generate a volume before input): left controller trigger; Eraser on/off: click left controller thumb stick; Change Brush size: right controller thumb stick.

Integrate with other input device: see AbstractInputDevice.cs.

### Procedural modelling

![](images/image3.jpeg)

Load a volume file then press F7 to F8. The example scene contains 2 procedural modelling simple: (a): colorful 3D spiral lines; (b) a dice.

### Input Action Record and Replay

Record: Press start/stop recording; Replay: Press play recording.
Testing: load the 512<sup>3</sup> horse volume then click on "Play recording" button.

### Experimental features 

1: 2D image to volume (Image to Volume-Mouse example.scene)

An example of usthort2 SDF volume generation in Unity C# (ImageToVolume.cs): image to texture2D to usthort2 SDF (max volume size: 512<sup>3</sup> )

![](D:\LVDIF Github\Repo\LVDIF\images\image2volume.jpg)

(a) 2D test image; (b) depth image of (a); (c) volume of (a); (d) volume (c) with offset from (b)

Source:

[Test image](https://civitai.com/images/1805059?period=AllTime&periodMode=published&sort=Newest&view=categories&username=yuki_hotaru&withTags=false) [Depth image from an online midas model](https://imageamigo.com/photodepth/) 

### Known issues
1. CUDA dll (?) cannot be run from folder contains non-ASCII characters in path (运行路径不能包含中文，似乎是CUDA dll的问题)
2. Pixel flickering at low rendering resolution (may be caused by the vertex color shader)

## To do list
1. Add Compute shader as alternative backend of rendering pipeline to support more GPU device. It is possible to replace the CUDA kernel and single large GPU buffer for large volume data with Compute shader and a group of smaller 3D textures. 
2. Smooth color rendering.
3. Change the input recording file to JSON.
4. Implement ray marching and other volume rendering algorithms. 
5. API document and Wiki.
6. CMake build script for the DLL.
7. Add support for other game engine such as Unreal.

## Contributions
We welcome community contributions to this repository.

## Thanks
Many thanks to the authors of these open-source repositories:
1. [Distance functions for basic primitives](https://iquilezles.org/articles/distfunctions/)
2. [Unity Marching Cubes](https://github.com/SebLague/Marching-Cubes)

## Citations
```bibtex
@article{wang2023lvdif,
  title={LVDIF: a framework for real-time interaction with large volume data},
  author={Wang, Jialin and Xiang, Nan and Kukreja, Navjot and Yu, Lingyun and Liang, Hai-Ning},
  journal={The Visual Computer},
  pages={1--14},
  year={2023},
  publisher={Springer}
}
```

Link to [Paper](https://link.springer.com/article/10.1007/s00371-023-02976-x)

Contact: chaosikaros@outlook.com