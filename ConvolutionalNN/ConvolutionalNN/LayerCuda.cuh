#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "Types.h"

namespace cnn{
	namespace cuda{

__device__ float add(float* address, float value)
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

template<typename T>
__global__ void calculatePotential(T* input, float* weights, float* output, uint32 inputLength, uint32 neuronsNr)
{
	uint32 weightsLength=inputLength*neuronsNr;
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<weightsLength)
	{
		uint32 neuronNr=idx%inputLength;
		add(&output[neuronNr], input[neuronNr]*weights[idx]);
	}
}

__global__ void calculateSigmoidalOutput(float* output, uint32 neuronsNr)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=1/(1+expf(-output[idx]));
	}
}

__global__ void calculateTanhOutput(float* output, uint32 neuronsNr)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=tanhf(output[idx]);
	}
}

__global__ void calculateMaxOutput(float* output, uint32 neuronsNr)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=output[idx]>0?output[idx]:0;
	}
}
	}}




