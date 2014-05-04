
#ifndef CNN_SAMPLER_H_
#define CNN_SAMPLER_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Sampler {
public:
	Sampler();
	virtual ~Sampler();
	
	virtual void sample(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer) = 0;

	void operator()(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);
};


template <typename T>
Sampler<T>::Sampler(){

}


template <typename T>
Sampler<T>::~Sampler(){

}


template <typename T>
void Sampler<T>::operator()(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	sample(pImageBatch, pInputBuffer, pOutputBuffer);
}


	}
}

#endif	/* CNN_SAMPLER_H_ */