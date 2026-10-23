#property strict
#property script_show_inputs
#include <MQL5GPULibrary.mqh>

// Synthetic data only; does not submit orders or read account information.
string LastLibraryError()
{
   short message[1024];
   DN_GetError(message, 1024);
   string text = "";
   for(int i = 0; i < 1024 && message[i] != 0; ++i)
      text += ShortToString(message[i]);
   return text;
}

void OnStart()
{
   int h = DN_Create();
   if(h <= 0)
   {
      Print("LSTM create failed: ", LastLibraryError());
      return;
   }
   double x[12] = {0.1,0.2,0.3, 0.2,0.3,0.4, 0.3,0.4,0.5, 0.4,0.5,0.6};
   double targets[4] = {0.4,0.5,0.6,0.7};
   double predicted[4];
   bool ok = DN_SetSequenceLength(h, 3) && DN_SetMiniBatchSize(h, 4)
          && DN_AddLayerEx(h, 1, 8, 0, 0, 0.0) && DN_SetOutputDim(h, 1)
          && DN_LoadBatch(h, x, targets, 4, 3, 1, 0)
          && DN_Train(h, 20, 0.0, 0.005, 0.0)
          && DN_PredictBatch(h, x, 4, 3, 0, predicted);
   if(ok)
   {
      double mse = 0.0;
      int epochs = 0;
      DN_GetTrainingResult(h, mse, epochs);
      PrintFormat("LSTM smoke test passed: epochs=%d, MSE=%.8f, prediction=%.6f", epochs, mse, predicted[3]);
   }
   else
   {
      Print("LSTM smoke test failed: ", LastLibraryError());
   }
   DN_Free(h);
}
