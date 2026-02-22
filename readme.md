
Here is a comprehensive and professional README.md file tailored for your GitHub repository. It highlights the high-performance nature of the library and documents the API clearly.
MQL5-GPU-LSTM: CUDA Accelerated Neural Network Library
![alt text](https://img.shields.io/badge/CUDA-11%2B-green.svg)

![alt text](https://img.shields.io/badge/Platform-Windows%20x64-blue.svg)

![alt text](https://img.shields.io/badge/MQL5-Compatible-orange.svg)
MQL5-GPU-LSTM is a high-performance, dynamic link library (DLL) designed to bring Deep Learning capabilities directly to MetaTrader 5 (MQL5). Unlike standard CPU-based implementations, this library leverages NVIDIA GPUs (via CUDA, cuBLAS, and cuRAND) to train and execute multi-layer LSTM networks significantly faster.
Crucially, it supports Asynchronous Training, allowing the heavy computational load to run on a background thread without freezing the MetaTrader terminal UI.
Features
Pure CUDA Implementation: Built on top of cuBLAS (matrix operations) and cuRAND (RNG) for maximum throughput.
Asynchronous Training: Trains models in a background thread. Your indicator/EA continues to receive ticks and update the UI while the GPU crunches data.
Multi-Layer LSTM: Support for stacking multiple LSTM layers with arbitrary hidden sizes.
Modern Optimization: Implements the AdamW optimizer with Weight Decay and Gradient Clipping.
Dropout Regularization: Supports inverted dropout for robust training.
Memory Efficient: Uses persistent GPU buffers to minimize allocation overhead during real-time trading.
State Serialization: Save and load full network states (weights + optimizer moments) to/from MQL5 byte arrays.
Prerequisites
NVIDIA GPU (Compute Capability 6.0+ recommended).
NVIDIA Drivers installed and up-to-date.
MetaTrader 5 (64-bit).
CUDA Runtime DLLs: The compiled DLL requires cudart64_xx.dll, cublas64_xx.dll, and curand64_xx.dll to be present in the system PATH or the MT5 Libraries folder.
Installation
Download the compiled MQL5GPULibrary_LSTM.dll.
Place the DLL into your MetaTrader 5 Data Folder: MQL5\Libraries.
Ensure "Allow DLL imports" is enabled in your MetaTrader settings (Tools -> Options -> Expert Advisors).
API Documentation
The library exports stdcall functions compatible with MQL5.
1. Instance Management
Function	Description
int DN_Create()	Creates a new LSTM network instance on the GPU. Returns a handle (int) ID.
void DN_Free(int h)	Releases all GPU memory and resources associated with handle h.
2. Configuration & Architecture
Function	Description
int DN_SetSequenceLength(int h, int seq_len)	Sets the time-step lookback window size.
int DN_SetMiniBatchSize(int h, int mbs)	Sets the batch size for training.
int DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop)	Adds an LSTM layer. <br>in: Input dim, out: Hidden size. <br>act/ln: Reserved (0). <br>drop: Dropout probability (0.0 - 1.0).
int DN_SetOutputDim(int h, int out_dim)	Sets the dimension of the final dense (linear) output layer.
int DN_SetGradClip(int h, double clip)	Sets the gradient clipping threshold to prevent exploding gradients.
3. Data Loading
Function	Description
int DN_LoadBatch(...)	Uploads training data from MQL5 (double array) to GPU (float tensor). <br> X[]: Input features, T[]: Target values.
4. Asynchronous Training (Non-Blocking)
Function	Description
int DN_TrainAsync(int h, int epochs, double mse, double lr, double wd)	Starts training on a background thread. Returns immediately. <br>lr: Learning Rate, wd: Weight Decay.
int DN_GetTrainingStatus(int h)	Checks the status of the background training. <br>Returns: <br>0: IDLE <br>1: RUNNING <br>2: COMPLETED <br>-1: ERROR
void DN_GetTrainingResult(int h, double &mse, int &ep)	Retrieves the final Mean Squared Error and epoch count after training is COMPLETED.
void DN_StopTraining(int h)	Signals the background thread to stop immediately.
5. Inference (Prediction)
Function	Description
int DN_PredictBatch(...)	Performs a forward pass. <br>Input X[] -> Output Y[]. Dropout is automatically disabled during inference.
6. State Management & Weights
Function	Description
int DN_SnapshotWeights(int h)	Saves the current weights as the "best" known state in GPU memory.
int DN_RestoreWeights(int h)	Reverts weights to the last snapshot (useful if retraining diverges).
int DN_SaveState(int h)	Serializes the entire network to an internal buffer. Returns the size in bytes.
int DN_GetState(int h, char &buf[], int len)	Copies the serialized data into an MQL5 char array.
int DN_LoadState(int h, const char &buf[])	Reconstructs the network from a char array.
7. Diagnostics
Function	Description
void DN_GetError(short &buf[], int len)	Retrieves the last error message (CUDA/cuBLAS error string) into a short array (string).
double DN_GetGradNorm(int h)	Returns the L2 norm of the last computed gradients.
MQL5 Usage Example
Here is a simplified snippet of how to use the async training in an Indicator or Expert Advisor:
code
C++
#import "MQL5GPULibrary_LSTM.dll"
   int DN_Create();
   int DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
   int DN_GetTrainingStatus(int h);
   // ... other imports
#import

int net_handle = 0;

int OnInit() {
   net_handle = DN_Create();
   // ... Add layers, Load data ...
   
   // Start training without freezing MT5
   DN_TrainAsync(net_handle, 10000, 0.001, 0.001, 0.0001);
   
   EventSetTimer(1); // Check status every second
   return INIT_SUCCEEDED;
}

void OnTimer() {
   int status = DN_GetTrainingStatus(net_handle);
   
   if(status == 1) {
      Print("Training in progress..."); 
   }
   else if(status == 2) {
      Print("Training Complete!");
      EventKillTimer();
      // Proceed to prediction logic...
   }
}

Building from Source
To build this project, you need:
Visual Studio 2019 or 2022.
CUDA Toolkit 11.x or 12.x.
Configure the project output to Release / x64.
Ensure Generate GPU Debug Information (-G) is set to No for maximum performance.

License
MIT License. See LICENSE file for details.