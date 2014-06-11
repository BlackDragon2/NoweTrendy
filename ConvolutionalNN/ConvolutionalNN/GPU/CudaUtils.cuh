#ifndef CNN_CUDA_UTILS_H_
#define CNN_CUDA_UTILS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "../Types.h"

namespace cnn{
	namespace cuda{

__device__ inline float add(float* address, float value)
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
	}}
#endif	/* CNN_CUDA_UTILS_H_ */


