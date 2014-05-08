#include "LayerCuda.cuh"

__device__ float cnn::cuda::add(float* address, float value)
{
  float old = value;  
  float ret=atomicExch(address, 0.0f);
  float new_old=ret+old;
  while ((old = atomicExch(address, new_old))!=0.0f)
  {
	new_old = atomicExch(address, 0.0f);
	new_old += old;
  }
  return ret;
}

__global__ void cnn::cuda::calculateSigmoidalOutput(float* output, uint32 neuronsNr, float* weights, float* biases)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=1/(1+expf(-output[idx]+(weights[idx]*biases[idx])));//aktywacja+bias
	}
}

__global__ void cnn::cuda::calculateTanhOutput(float* output, uint32 neuronsNr, float* weights, float* biases)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=tanhf(output[idx]+weights[idx]*biases[idx]);//aktywacja+bias
	}
}

__global__ void cnn::cuda::calculateTahnDelta(float* output, uint32 neuronsNr, float* errorRates, float* weights)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		errorRates[idx]*=(1-output[idx]*output[idx])*weights[idx];//1-aktywacja^2*waga polaczena
	}
}

__global__ void cnn::cuda::calculateSigmoidalDelta(float* output, uint32 neuronsNr, float* errorRates, float* weights)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		errorRates[idx]*=output[idx]*(1-output[idx])*weights[idx];//1-aktywacja^2*waga polaczena
	}
}
