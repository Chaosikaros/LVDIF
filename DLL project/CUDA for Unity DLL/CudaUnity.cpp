#include "CudaUnity.h"
#include "MarchingCubesChunk.h"
#include <conio.h>

using namespace CudaUnity;
using namespace std;

#define MaxPrintStackNameLen 9999
HANDLE hConsole = NULL;
HANDLE hConIn = NULL;
HANDLE hConOut = NULL;
CPPCallback DebugLogCallBack = NULL;
McCallback mcCallBack = NULL;
SdfCallback sdfCallBack = NULL;
MeshChunkCallback meshChunkCallBack = NULL;
bool msgBox = false;
bool msgConsole = true;
int debugLevel = 0;//0 means no debug output; the bigger level, means the more debug content.
int cudaDeviceID = 0;
StopWatchInterface *timer = NULL;

int maxChunks = 0;
vector<MarchingCubesChunk> mcChunks;



#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
	int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		DebugLogToUnity(sFormator(
			stderr,
			"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

namespace CudaUnity
{
	const std::string currentDateTime() {
		using namespace std::chrono;

		// get current time
		auto now = system_clock::now();

		// get number of milliseconds for the current second
		// (remainder after division into seconds)
		auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

		// convert to std::time_t in order to convert to std::tm (broken time)
		auto timer = system_clock::to_time_t(now);

		// convert to broken time
		std::tm bt = *std::localtime(&timer);

		std::ostringstream oss;

		oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S"); // HH:MM:SS
		oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << " : ";
		return oss.str();
	}

	inline void setcolor(int textcol, int backcol)
	{
		if ((textcol % 16) == (backcol % 16))textcol++;
		textcol %= 16; backcol %= 16;
		unsigned short wAttributes = ((unsigned)backcol << 4) | (unsigned)textcol;
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		SetConsoleTextAttribute(hConsole, wAttributes);
	}

	void refresh()
	{
		HWND hwnd = FindWindowEx(NULL, NULL, "CabinetWClass", NULL);
		while (hwnd != NULL)
		{
			PostMessage(hwnd, WM_COMMAND, 41504, 0);
			hwnd = FindWindowEx(NULL, hwnd, "CabinetWClass", NULL);
		}
	}

	DWORD WINAPI MessageBoxThread(LPVOID lpParam) {
		MessageBox(NULL, (char*)lpParam ,"CUDA error", MB_ICONEXCLAMATION | MB_OK);
		return 0;
	}

	void WriteInWindow(string output)
	{
		if (msgConsole)
			std::cout << output.c_str() << std::endl;
	}

	inline void cudaAssert(cudaError_t code, const char *file, int line)
	{
		if (code != cudaSuccess)
		{
			string errorMsg = sFormator("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
			if(msgBox)
				CreateThread(NULL, NULL, MessageBoxThread, (LPVOID)errorMsg.c_str(), NULL, NULL);
			WriteInWindow(currentDateTime() + errorMsg);
			DebugLogToUnity(errorMsg);
			//exit(EXIT_FAILURE);
		}
	}

	inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) 
	{
		cudaError_t err = cudaGetLastError();

		if (cudaSuccess != err) {
			string errorMsg = sFormator("%s(%i) : getLastCudaError() CUDA error :"" %s : (%d) %s.\n",
				file, line, errorMessage, static_cast<int>(err),cudaGetErrorString(err));
			if (msgBox)
				CreateThread(NULL, NULL, MessageBoxThread, (LPVOID)errorMsg.c_str(), NULL, NULL);
			WriteInWindow(currentDateTime() + errorMsg);
			DebugLogToUnity(errorMsg);
			//exit(EXIT_FAILURE);
		}
	}

	void printStack(CONTEXT* ctx) //Prints stack trace based on context record
	{
		BOOL    result;
		HANDLE  process;
		HANDLE  thread;
		HMODULE hModule;

		STACKFRAME64        stack;
		ULONG               frame;
		DWORD64             displacement;

		DWORD disp;
		IMAGEHLP_LINE64 *line;

		char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
		char name[MaxPrintStackNameLen];
		char module[MaxPrintStackNameLen];
		PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;

		memset(&stack, 0, sizeof(STACKFRAME64));

		process = GetCurrentProcess();
		thread = GetCurrentThread();
		displacement = 0;
#if !defined(_M_AMD64)
		stack.AddrPC.Offset = (*ctx).Eip;
		stack.AddrPC.Mode = AddrModeFlat;
		stack.AddrStack.Offset = (*ctx).Esp;
		stack.AddrStack.Mode = AddrModeFlat;
		stack.AddrFrame.Offset = (*ctx).Ebp;
		stack.AddrFrame.Mode = AddrModeFlat;
#endif

		SymInitialize(process, NULL, TRUE); //load symbols

		for (frame = 0; ; frame++)
		{
			//get next call from stack
			result = StackWalk64
			(
#if defined(_M_AMD64)
				IMAGE_FILE_MACHINE_AMD64
#else
				IMAGE_FILE_MACHINE_I386
#endif
				,
				process,
				thread,
				&stack,
				ctx,
				NULL,
				SymFunctionTableAccess64,
				SymGetModuleBase64,
				NULL
			);

			if (!result) break;

			//get symbol name for address
			pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
			pSymbol->MaxNameLen = MAX_SYM_NAME;
			SymFromAddr(process, (ULONG64)stack.AddrPC.Offset, &displacement, pSymbol);

			line = (IMAGEHLP_LINE64 *)malloc(sizeof(IMAGEHLP_LINE64));
			line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);

			//try to get line
			if (SymGetLineFromAddr64(process, stack.AddrPC.Offset, &disp, line))
			{
				WriteInWindow(sFormator("\tat %s in %s: line: %lu: address: 0x%0X\n", pSymbol->Name, line->FileName, line->LineNumber, pSymbol->Address));
			}
			else
			{
				//failed to get line
				WriteInWindow(sFormator("\tat %s, address 0x%0X.\n", pSymbol->Name, pSymbol->Address));
				hModule = NULL;
				lstrcpyA(module, "");
				GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					(LPCTSTR)(stack.AddrPC.Offset), &hModule);

				//at least print module name
				if (hModule != NULL)GetModuleFileNameA(hModule, module, MaxPrintStackNameLen);

				WriteInWindow(sFormator("in %s\n", module));
			}

			free(line);
			line = NULL;
		}
	}

	int seh_filter(_EXCEPTION_POINTERS* ex)
	{
		printf("*** Exception 0x%x occured ***\n\n", ex->ExceptionRecord->ExceptionCode);
		printStack(ex->ContextRecord);

		return EXCEPTION_EXECUTE_HANDLER;
	}
	
	void DebugLogToUnity(string output)
	{
		DebugLogCallBack((_bstr_t)output.c_str());
	}

	void ThrowErrorWindow(string output)
	{
		if (msgBox)
			CreateThread(NULL, NULL, MessageBoxThread, (LPVOID)output.c_str(), NULL, NULL);
		WriteInWindow(currentDateTime() + output);
		DebugLogToUnity(output);
	}

	void handle_eptr(std::exception_ptr eptr, string msg)
	{
		try {
			if (eptr) {
				std::rethrow_exception(eptr);
			}
		}
		catch (const std::exception& e) {
			ThrowErrorWindow(msg + e.what());
		}
	}

	string sFormator(const std::string sFormat, ...) {

		const char * const zcFormat = sFormat.c_str();

		va_list vaArgs;
		va_start(vaArgs, sFormat);

		va_list vaCopy;
		va_copy(vaCopy, vaArgs);
		const int iLen = std::vsnprintf(NULL, 0, zcFormat, vaCopy);
		va_end(vaCopy);

		vector<char> zc(iLen + 1);
		vsnprintf(zc.data(), zc.size(), zcFormat, vaArgs);
		va_end(vaArgs);
		return string(zc.data(), zc.size());
	}

	string GetStringOfLastError(cudaError_t err)
	{
		if(cudaSuccess != err)
			return sFormator("getLastCudaError() CUDA error: (%d) %s.\n", static_cast<int>(err), cudaGetErrorString(err));
		else
			return "has no error";
	}

	int nextPowerCounterOfTwo(int x) {
		int power = 1;
		int counter = 0;
		while (power < x) {
			power *= 2;
			counter++;
		}
		return counter;
	}

}

void SetTimer()
{
	if (debugLevel >= 1)
	{
		sdkStartTimer(&timer);
		sdkResetTimer(&timer);
	}

}

bool CheckMcChunks(int chunkID)
{
	if (!mcChunks.empty())
	{
		if (chunkID < maxChunks)
		{
			return true;
		}
		else
		{
			DebugLogToUnity("ChunkID exceeds maxChunks.");
			return false;
		}
	}
	else
	{
		DebugLogToUnity("mcChunks[] is empty.");
		return false;
	}
}

void SaveSdfData(int chunkID, char* fileName)
{
	if (CheckMcChunks(0))
		return mcChunks[0].SaveSdfData(chunkID, fileName);
}

void GetSdfArray(int chunkID, int size, float* sdfData)
{
	if (CheckMcChunks(0))
		return mcChunks[0].GetSDF(size, sdfData);
}
void GetColorArray(int chunkID, int size, int* colorData)
{
	if (CheckMcChunks(0))
		return mcChunks[0].GetVolumeColor(size, colorData);
}
//bool SetInputForVoxel(int chunkID, int inputRadius, Vector3 inputPos, bool eraserMode)
//{
//	if (CheckMcChunks(0))
//		return mcChunks[0].SetInputForVoxel(inputRadius, inputPos, eraserMode);
//	return false;
//}

bool SetInputBrushForVoxel(int chunkID, int inputRadius, int brushSize, Vector4* inputPos, int colorType, float colorBrushOffset, bool eraserMode, int brushShape)
{
	if (CheckMcChunks(0))
		return mcChunks[0].SetInputBrushForVoxel(chunkID, inputRadius, brushSize, inputPos, colorType, colorBrushOffset, eraserMode, brushShape);
	return false;
}
bool SetMeshForVoxel(Vector3 gridSize, Vector3 boundsMax, Vector3 boundsMin, Vector3* vertices, int* triangles, Vector3* normals, int verticeSize, int triangleSize, int normalSize)
{
	if (CheckMcChunks(0))
		return mcChunks[0].SetMeshForVoxel(gridSize, boundsMax, boundsMin, vertices, triangles, normals, verticeSize, triangleSize, normalSize);
	return false;
}

bool SetMarchingCubesChunks(int num, int minChunkSizeLog2, int chunkThreads)
{
	//if(maxChunks != 0)
	//	for (int i = 0; i < maxChunks; i++)
	//		mcChunks[i].FreeMemoryForMC();
	mcChunks.clear();
	for (int i = 0; i < num; i++)
		mcChunks.push_back(i);
	for (int i = 0; i < num; i++)
		mcChunks[i].InitMC(i,mcCallBack,sdfCallBack, meshChunkCallBack, minChunkSizeLog2, chunkThreads);
	maxChunks = num;
	return true;
}

void GetExtractMarchingCubesData(int chunkID, int size, Vector3 *vertices, Vector3 *normals, int *triangles)
{
	if (CheckMcChunks(chunkID))
		mcChunks[chunkID].GetExtractMarchingCubesData(size,vertices,normals, triangles);
}
void GetExtractMarchingCubesChunkData(int chunkID, int childChunkID, int size, Vector3* vertices, Vector3* normals, int* triangles, int* colors
	, float GridWI, int triangleThreads, float IsoLevel, bool EnableSmooth)
{
	if (CheckMcChunks(chunkID))
		mcChunks[chunkID].GetExtractMarchingCubesChunkData(childChunkID, size, vertices, normals, triangles, colors, GridWI, triangleThreads, IsoLevel, EnableSmooth);
}

void SetMarchingCubesKernelInThread(int chunkID, int size, int voxelThreads, int triangleThreads, float3 CenterPosI,
	float GridWI, float octreeBboxSizeOffset, float IsoLevel, bool EnableSmooth,
	bool loadMesh, bool loadSdf, bool loadSdfFromUnity, ushort2* sdfData, int gridSizeLog2OBox, bool exportTexture3D, char* sdfFileName, float filterValue, int sleepTime, int maxUpatedChunk, int SVFSetting)
{
	string fileNameS = string(sdfFileName);
	if (CheckMcChunks(chunkID))
		mcChunks[chunkID].SetMarchingCubesThread(size, voxelThreads, triangleThreads,
			CenterPosI, GridWI, octreeBboxSizeOffset, IsoLevel, EnableSmooth, loadMesh, 
			loadSdf, loadSdfFromUnity, sdfData, gridSizeLog2OBox, exportTexture3D, fileNameS, filterValue, sleepTime, maxUpatedChunk, SVFSetting);
}

void SetCallbackForMC(McCallback callback)
{
	mcCallBack = callback;
}
void SetCallbackForMeshChunk(MeshChunkCallback callback)
{
	meshChunkCallBack = callback;
}
void SetCallbackForSdfExport(SdfCallback callback)
{
	sdfCallBack = callback;
}

int GetExtractMarchingTriCount(int chunkID)
{
	if (CheckMcChunks(chunkID))
		return mcChunks[chunkID].triCount;
	return 0;
}

bool GetExtractCubeVoxels(int chunkID,int size, float* cubeVoxel)
{
	if (CheckMcChunks(chunkID))
		return mcChunks[chunkID].GetExtractCubeVoxels(cubeVoxel, size);
	return false;
}

bool MallocMemoryForMC(int chunkID, Vector3 size, float GridWI)
{
	if (CheckMcChunks(chunkID))
		mcChunks[chunkID].MallocMemoryForMC(size, GridWI);
	return true;
}

bool FreeMemoryForMC(int chunkID)
{
	if (CheckMcChunks(chunkID))
	{
		mcChunks[chunkID].FreeMemoryForMC();
	}
	return true;
}

void SetCallback(CPPCallback callback)
{
	DebugLogCallBack = callback;
}

int GetDeviceNumber()
{
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	return deviceCount;
}

void SetDevice(int deviceID)
{
	if (deviceID < GetDeviceNumber())
	{
		checkCudaErrors(cudaSetDevice(deviceID));
		cudaDeviceID = deviceID;
	}
}

void GetDeviceInfo()
{
	DebugLogToUnity(sFormator(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n"));

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		DebugLogToUnity(sFormator("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id)));
		DebugLogToUnity(sFormator("Result = FAIL\n"));
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		DebugLogToUnity(sFormator("There are no available device(s) that support CUDA\n"));
	}
	else {
		DebugLogToUnity(sFormator("Detected %d CUDA Capable device(s)\n", deviceCount));
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		DebugLogToUnity(sFormator("\nDevice %d: \"%s\"\n", dev, deviceProp.name));

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		DebugLogToUnity(sFormator("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10));
		DebugLogToUnity(sFormator("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor));

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintfmsg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		DebugLogToUnity(sFormator("%s", msg));

		DebugLogToUnity(sFormator("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount));
		DebugLogToUnity(sFormator(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f));

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		DebugLogToUnity(sFormator("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f));
		DebugLogToUnity(sFormator("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth));

		if (deviceProp.l2CacheSize) {
			DebugLogToUnity(sFormator("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize));
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		DebugLogToUnity(sFormator("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f));
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		DebugLogToUnity(sFormator("  Memory Bus Width:                              %d-bit\n",
			memBusWidth));
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			DebugLogToUnity(sFormator("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize));
		}

#endif

		DebugLogToUnity(sFormator(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]));
		DebugLogToUnity(sFormator(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]));
		DebugLogToUnity(sFormator(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]));

		DebugLogToUnity(sFormator("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem));
		DebugLogToUnity(sFormator("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock));
		DebugLogToUnity(sFormator("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock));
		DebugLogToUnity(sFormator("  Warp size:                                     %d\n",
			deviceProp.warpSize));
		DebugLogToUnity(sFormator("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor));
		DebugLogToUnity(sFormator("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock));
		DebugLogToUnity(sFormator("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]));
		DebugLogToUnity(sFormator("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]));
		DebugLogToUnity(sFormator("  Maximum memory pitch:                          %zu bytes\n",
			deviceProp.memPitch));
		DebugLogToUnity(sFormator("  Texture alignment:                             %zu bytes\n",
			deviceProp.textureAlignment));
		DebugLogToUnity(sFormator(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount));
		DebugLogToUnity(sFormator("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled"));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		DebugLogToUnity(sFormator("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)"));
#endif
		DebugLogToUnity(sFormator("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Device supports Managed Memory:                %s\n",
			deviceProp.managedMemory ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No"));
		DebugLogToUnity(sFormator("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID));

		const char *sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown",
			NULL };
		DebugLogToUnity(sFormator("  Compute Mode:\n"));
		DebugLogToUnity(sFormator("     < %s >\n", sComputeMode[deviceProp.computeMode]));
	}

	// If there are 2 or more GPUs, query to determine whether RDMA is supported
	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

			// Only boards based on Fermi or later can support P2P
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				// on Windows (64-bit), the Tesla Compute Cluster driver for windows
				// must be enabled to support this
				&& prop[i].tccDriver
#endif
				) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		// Show all the combinations of support P2P GPUs
		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					DebugLogToUnity(sFormator("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
						prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
						can_access_peer ? "Yes" : "No"));
				}
			}
		}
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	DebugLogToUnity("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	DebugLogToUnity(sFormator("%s", sProfileString.c_str()));

	DebugLogToUnity("Result = PASS\n");
}

void SetMaxBlockSize(int input)
{
	CudaUnity::SetMaxBlockSize(input);
}

void SetDebugLevel(int input)
{
	debugLevel = input;
}

int GetMallocHeapSize()
{
	size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
	return (unsigned)limit;
}

int SetMallocHeapSize(int MallocHeapSize)
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)MallocHeapSize);
	return MallocHeapSize;
}

void DisplayErrorInMessageBox(bool input)
{
	msgBox = input;
}

void DisplayErrorInExtraConsole(bool input)
{
	msgConsole = input;
	if (!AllocConsole() || !msgConsole) {
		return;
	}
	HWND hwnd = ::GetConsoleWindow();
	if (hwnd != NULL)
	{
		HMENU hMenu = ::GetSystemMenu(hwnd, FALSE);
		if (hMenu != NULL) DeleteMenu(hMenu, SC_CLOSE, MF_BYCOMMAND);
	}
	SetConsoleTitle(_T("CUDA Lib Console"));
	WriteInWindow(currentDateTime() + "CUDA Lib Console");
	// std::cout, std::clog, std::cerr, std::cin
	FILE* fDummy;
	freopen_s(&fDummy, "CONOUT$", "w", stdout);
	freopen_s(&fDummy, "CONOUT$", "w", stderr);
	freopen_s(&fDummy, "CONIN$", "r", stdin);
	std::cout.clear();
	std::clog.clear();
	std::cerr.clear();
	std::cin.clear();

	// std::wcout, std::wclog, std::wcerr, std::wcin
	hConOut = CreateFile(_T("CONOUT$"), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	hConIn = CreateFile(_T("CONIN$"), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	SetStdHandle(STD_OUTPUT_HANDLE, hConOut);
	SetStdHandle(STD_ERROR_HANDLE, hConOut);
	SetStdHandle(STD_INPUT_HANDLE, hConIn);
	std::wcout.clear();
	std::wclog.clear();
	std::wcerr.clear();
	std::wcin.clear();
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
}