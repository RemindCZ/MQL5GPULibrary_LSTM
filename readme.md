
Here is a comprehensive and professional README.md file tailored for your GitHub repository. It highlights the high-performance nature of the library and documents the API clearly.
MQL5-GPU-LSTM: CUDA Accelerated Neural Network Library
![alt text](https://img.shields.io/badge/CUDA-11%2B-green.svg)

![alt text](https://img.shields.io/badge/Platform-Windows%20x64-blue.svg)

![alt text](https://img.shields.io/badge/MQL5-Compatible-orange.svg)

# MQL5-GPU-LSTM

CUDA-Accelerated Neural Network Library for MetaTrader 5

MQL5-GPU-LSTM is a high-performance dynamic link library designed to bring deep learning capabilities directly into MetaTrader 5 (MQL5). Unlike traditional CPU-based solutions, it leverages NVIDIA GPUs through CUDA, cuBLAS, and cuRAND to train and execute multi-layer LSTM networks significantly faster.

A feature of the library is asynchronous training. Heavy computations run in a background thread, allowing the MetaTrader terminal to remain fully responsive while continuing to process ticks and update the user interface.

---

## Features

Pure CUDA implementation  
Built on top of cuBLAS for matrix operations and cuRAND for random number generation to achieve maximum throughput.

Asynchronous training  
Model training runs in a background thread without blocking the terminal UI.

Multi-layer LSTM support  
Allows stacking multiple LSTM layers with arbitrary hidden sizes.

Modern optimization  
Implements the AdamW optimizer with weight decay and gradient clipping.

Dropout regularization  
Supports inverted dropout for improved training stability.

Memory efficiency  
Uses persistent GPU buffers to minimize allocation overhead during real-time trading.

State serialization  
Full network state, including weights and optimizer moments, can be saved and restored using MQL5 byte arrays.

---

## Requirements

NVIDIA GPU (Compute Capability 6.0 or higher recommended)  
Up-to-date NVIDIA drivers  
MetaTrader 5 (64-bit)  

Required CUDA runtime libraries must be available in the system PATH or in the MT5 Libraries folder:

cudart64_xx.dll  
cublas64_xx.dll  
curand64_xx.dll  

---

## Installation

1. Download the compiled MQL5GPULibrary_LSTM.dll file.

2. Place the DLL into the MetaTrader 5 data folder:

MQL5\Libraries

3. Enable DLL imports in MetaTrader:

Tools → Options → Expert Advisors → Allow DLL imports

---

## API Documentation

The library exports stdcall functions compatible with MQL5.

---

### Instance Management

DN_Create()  
Creates a new LSTM network instance on the GPU and returns an integer handle.

DN_Free(int h)  
Releases all GPU resources associated with the given handle.

---

### Configuration and Architecture

DN_SetSequenceLength(int h, int seq_len)  
Sets the time-step lookback window size.

DN_SetMiniBatchSize(int h, int batch_size)  
Sets the mini-batch size used during training.

DN_AddLayerEx(int h, int in, int out, int act, int ln, double dropout)  
Adds an LSTM layer.  
"in" defines input dimension, "out" defines hidden size.  
"act" and "ln" are reserved parameters.  
"dropout" specifies dropout probability between 0.0 and 1.0.

DN_SetOutputDim(int h, int out_dim)  
Defines the dimension of the final dense output layer.

DN_SetGradClip(int h, double clip)  
Sets gradient clipping threshold to prevent exploding gradients.

---

### Data Loading

DN_LoadBatch(...)  
Uploads training data from MQL5 arrays to GPU tensors.

---

### Asynchronous Training

DN_TrainAsync(int h, int epochs, double target_mse, double lr, double weight_decay)  
Starts training in a background thread and returns immediately.

DN_GetTrainingStatus(int h)  
Returns current training state:

0 = idle  
1 = running  
2 = completed  
-1 = error  

DN_GetTrainingResult(int h, double &mse, int &epochs)  
Retrieves final MSE and epoch count after training completes.

DN_StopTraining(int h)  
Immediately signals the background thread to stop.

---

### Inference

DN_PredictBatch(...)  
Performs forward pass from input features to output predictions.  
Dropout is automatically disabled during inference.

---

### State Management

DN_SnapshotWeights(int h)  
Stores the current weights as a reference state.

DN_RestoreWeights(int h)  
Restores weights from the last snapshot.

DN_SaveState(int h)  
Serializes the full network state and returns the buffer size.

DN_GetState(int h, char &buffer[], int length)  
Copies serialized state into an MQL5 array.

DN_LoadState(int h, const char &buffer[])  
Restores network state from serialized data.

---

### Diagnostics

DN_GetError(short &buffer[], int length)  
Retrieves the last CUDA or cuBLAS error message.

DN_GetGradNorm(int h)  
Returns the L2 norm of the most recent gradient values.

---

## Example Usage in MQL5

#import "MQL5GPULibrary_LSTM.dll"
int DN_Create();
int DN_TrainAsync(int h,int epochs,double mse,double lr,double wd);
int DN_GetTrainingStatus(int h);
#import

int net_handle = 0;

int OnInit()
{
   net_handle = DN_Create();

   DN_TrainAsync(net_handle, 10000, 0.001, 0.001, 0.0001);

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   int status = DN_GetTrainingStatus(net_handle);

   if(status == 1)
      Print("Training in progress");

   if(status == 2)
   {
      Print("Training completed");
      EventKillTimer();
   }
}

---

## Building from Source

Required tools:

Visual Studio 2019 or 2022  
CUDA Toolkit version 11.x or 12.x  

Recommended project settings:

Configuration: Release  
Platform: x64  
GPU debug information disabled for maximum performance

---

## License

MIT License. See the LICENSE file for full details.