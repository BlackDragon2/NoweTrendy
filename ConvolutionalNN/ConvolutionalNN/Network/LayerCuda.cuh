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

__global__ void calculateSigmoidalOutput(float* output, uint32 neuronsNr, float* weights, float* biases);
__global__ void calculateTanhOutput(float* output, uint32 neuronsNr, float* weights, float* biases);
__global__ void calculateSigmoidalDelta(float* output, uint32 neuronsNr, uint32 weightsLength, float* errorRates);//dla wag miedzy warstwa i a j - output warstwy i, neuronsNr i weightsLength warstwy j
__global__ void calculateTahnDelta(float* output, uint32 neuronsNr, uint32 weightsLength, float* errorRates);//dla wag miedzy warstwa i a j - output warstwy i, neuronsNr i weightsLength warstwy j

	}}


#endif	/* CNN_CUDA_LAYER_H_ */


