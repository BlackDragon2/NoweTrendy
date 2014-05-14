
#ifndef CNN_GPU_PACKER_H_
#define CNN_GPU_PACKER_H_

#include "Converter.cuh"


namespace cnn {
	namespace gpu {


template <typename F, typename T>
class Packer : public Converter<F, T>{
public:
	virtual void convert(
		ImageBatch<F> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer); 
};


template <typename F, typename T>
__global__ void packer(
	F*		pImagesInput,
	T*		pImagesOutput,
	size_t	pRowUnitSize,
	size_t	pAlignedRowUnitSize,
	size_t	pElements)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 offset	= idx % pAlignedRowUnitSize;

	if(idx >= pElements || offset >= pRowUnitSize)
		return;

	uint32 rowId = idx / pAlignedRowUnitSize;

	*(pImagesOutput + rowId * pRowUnitSize + offset) = static_cast<T>(*(pImagesInput + idx));
}


template <typename F, typename T> 
void Packer<F, T>::convert(
	ImageBatch<F> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t batchUnitSize	= pImageBatch.getBatchByteSize() / sizeof(F);
	size_t blocks			= utils::blocksCount(batchUnitSize, config::Cuda::THREADS_PER_BLOCK);
	
	packer<F, T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<F>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getImageByteSize() / sizeof(F),
		pImageBatch.getAlignedImageByteSize() / sizeof(F),
		batchUnitSize);
}


	}
}


#endif	/* CNN_GPU_PACKER_H_ */