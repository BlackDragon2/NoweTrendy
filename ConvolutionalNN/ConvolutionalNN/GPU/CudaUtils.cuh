#ifndef CNN_CUDA_UTILS_H_
#define CNN_CUDA_UTILS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "../Types.h"

namespace cnn{
	namespace cuda{

__device__ float add(float* address, float value);
	}}
#endif	/* CNN_CUDA_UTILS_H_ */


