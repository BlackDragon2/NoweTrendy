#ifndef CNN_CUDA_NETWORK_H_
#define CNN_CUDA_NETWORK_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "../Types.h"

namespace cnn{
	namespace cuda{

template<typename T>
__global__ void calculateError(T* classes, float* realOutput, float* errorRates, T exampleClass, uint32 neuronsNr)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		if(classes[idx]==exampleClass)
			errorRates[idx]=1-realOutput[idx];
		else
			errorRates[idx]=-realOutput[idx];
	}
}
	}}


#endif	/* CNN_CUDA_NETWORK_H_ */


