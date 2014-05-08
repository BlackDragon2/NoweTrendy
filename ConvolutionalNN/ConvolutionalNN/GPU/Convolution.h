
#ifndef CNN_CONVOLUTION_H_
#define CNN_CONVOLUTION_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Convolution {
public:
	typedef std::shared_ptr<Convolution<T>> PtrS;


public:
	Convolution();
	virtual ~Convolution(); 


	virtual void compute(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer,
		uint32					pKernelOffsetX,
		uint32					pKernelOffsetY) = 0;

	virtual void operator()(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer,
		uint32					pKernelOffsetX,
		uint32					pKernelOffsetY);
};


template <typename T>
Convolution<T>::Convolution(){

}


template <typename T>
Convolution<T>::~Convolution(){

}


template <typename T>
void Convolution<T>::operator()(
	ImageBatch<T> const&	pInputImageBatch,
	GpuBuffer&				pInputImageBuffer,
	ImageBatch<T> const&	pKernelsImageBatch,
	GpuBuffer&				pKernelsImageBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputImageBuffer,
	uint32					pKernelOffsetX,
	uint32					pKernelOffsetY)
{
	compute(
		pInputImageBatch, pInputImageBuffer, 
		pKernelsImageBatch, pKernelsImageBuffer, 
		pOutputImageBatch, pOutputImageBuffer,
		pKernelOffsetX, pKernelOffsetY);
}


	}
}

#endif	/* CNN_CONVOLUTION_H_ */