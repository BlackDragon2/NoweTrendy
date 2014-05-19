#include "LayerCuda.cuh"

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

__global__ void cnn::cuda::calculateTahnDelta(float* output, uint32 neuronsNr, uint32 weightsLength , float* errorRates)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 neuronNr=idx%neuronsNr;
	if(idx<weightsLength)
	{
		errorRates[idx]=(1-output[neuronNr]*output[neuronNr]);//1-aktywacja^2
	}
}

__global__ void cnn::cuda::calculateSigmoidalDelta(float* output, uint32 neuronsNr, uint32 weightsLength, float* errorRates)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 neuronNr=idx%neuronsNr;
	if(idx<weightsLength)
	{
		errorRates[idx]=output[neuronNr]*(1-output[neuronNr]);//aktywacja*(1-aktywacja)
	}
}
