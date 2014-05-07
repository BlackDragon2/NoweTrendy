
#ifndef CNN_CONVERTER_H_
#define CNN_CONVERTER_H_

#include "../Config.h"

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"



namespace cnn {
	namespace gpu {


template <typename F, typename T>
class Converter {
public:
	virtual void convert(
		ImageBatch<F> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer); 

	virtual void operator()(
		ImageBatch<F> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer); 

};


template <typename F, typename T>
__global__ void simpleConverter(
	F*		pImagesInput,
	T*		pImagesOutput,
	size_t	pElements)
{
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pElements)
		return;

	*(pImagesOutput + idx) = static_cast<T>(*(pImagesInput + idx));
}

template <typename F, typename T> 
void Converter<F, T>::convert(
	ImageBatch<F> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t batchUnitSize	= pImageBatch.getBatchByteSize() / sizeof(F);
	size_t blocks			= utils::blocksCount(batchUnitSize, config::Cuda::THREADS_PER_BLOCK);
	
	simpleConverter<F, T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<F>(),
		pOutputBuffer.getDataPtr<T>(),
		batchUnitSize);
}


template <typename F, typename T> 
void Converter<F, T>::operator()(
	ImageBatch<F> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	convert(pImageBatch, pInputBuffer, pOutputBuffer);
}


	}
}


#endif	/* CNN_CONVERTER_H_ */