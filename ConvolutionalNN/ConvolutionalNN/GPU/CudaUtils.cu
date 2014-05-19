#include "CudaUtils.cuh"

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