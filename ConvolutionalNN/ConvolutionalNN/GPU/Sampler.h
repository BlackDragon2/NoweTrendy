
#ifndef CNN_SAMPLER_H_
#define CNN_SAMPLER_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Sampler {
public:
	typedef std::shared_ptr<Sampler<T>> PtrS;


public:
	Sampler();
	virtual ~Sampler();
	
	virtual void sample(
		ImageBatch<T> const&	pInputImageBatch,  
		GpuBuffer&				pInputBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputBuffer,
		size_t					pSampleWidth,
		size_t					pSampleHeight) = 0;

	void operator()(
		ImageBatch<T> const&	pInputImageBatch,  
		GpuBuffer&				pInputBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputBuffer,
		size_t					pSampleWidth,
		size_t					pSampleHeight);
};


template <typename T>
Sampler<T>::Sampler(){

}


template <typename T>
Sampler<T>::~Sampler(){

}


template <typename T>
void Sampler<T>::operator()(
	ImageBatch<T> const&	pInputImageBatch, 
	GpuBuffer&				pInputBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputBuffer,
	size_t					pSampleWidth,
	size_t					pSampleHeight)
{
	sample(pInputImageBatch, pInputBuffer, pOutputImageBatch, pOutputBuffer, pSampleWidth, pSampleHeight);
}


	}
}

#endif	/* CNN_SAMPLER_H_ */