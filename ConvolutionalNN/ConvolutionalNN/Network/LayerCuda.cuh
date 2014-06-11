#ifndef CNN_CUDA_LAYER_H_
#define CNN_CUDA_LAYER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "../GPU/CudaUtils.cuh"
#include "../Types.h"

namespace cnn{
	namespace cuda{

		extern __device__ float add(float* address, float value);

template<typename T>
__global__ void calculatePotential(T* input, float* weights, float* output, uint32 inputLength, uint32 neuronsNr)
{
	uint32 weightsLength=inputLength*neuronsNr;
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<weightsLength)
	{
		uint32 neuronNr=idx/inputLength;
		add(&output[neuronNr], input[neuronNr]*weights[idx]);
	}
}

template<typename T>
__global__ void reset(T* toBeReseted, uint32 resetLength)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<resetLength)
		toBeReseted[idx]=0;
}

template<typename T>
__global__ void setWeightsUpdates(T* input, uint32 weightsLength, uint32 neuronsNr, float* biases, uint32 inputLength, float* updates, float* errorRates, float learningRate)
{
	uint32 idx		= (blockIdx.x * blockDim.x) + threadIdx.x;
	if(idx<weightsLength)
	{
		if(idx<neuronsNr)//liczba wag biasów (1 waga na neuron)
			add(&updates[idx], learningRate*errorRates[idx]*biases[idx]);
		else
			add(&updates[idx], learningRate*errorRates[idx/inputLength]*input[idx%inputLength]);//DO POPRAWY
	}
}

__global__ void calculateSigmoidalOutput(float* output, uint32 neuronsNr, float* weights, float* biases);
__global__ void calculateTanhOutput(float* output, uint32 neuronsNr, float* weights, float* biases);
__global__ void calculateSigmoidalDelta(float* output, uint32 neuronsNr, float* errorRates, float* errorRatesProp);
__global__ void calculateTahnDelta(float* output, uint32 neuronsNr, float* errorRates, float* errorRatesProp);
__global__ void calculateSigmoidalError(uint32 exampleClass, float* output, uint32 neuronsNr, float* errorRates, float* error);
__global__ void calculateTanhError(uint32 exampleClass, float* output, uint32 neuronsNr, float* errorRates, float* error);
__global__ void calculateWeightedError(float* errorRates, float* weights, float* weightedError, uint32 inputLength, uint32 neuronsNr);
__global__ void updateWeights(float* weights, float* weigthsUpdate, uint32 weightsLength);
	}}


#endif	/* CNN_CUDA_LAYER_H_ */


