
#ifndef CNN_VARIANCE_CENTERIZER_H_
#define CNN_VARIANCE_CENTERIZER_H_

#include "AverageCenterizer.cuh"


namespace cnn {
	namespace gpu {


template <typename T>
class VarianceCenterizer : public AverageCenterizer<T> {
public:
	VarianceCenterizer();
	virtual ~VarianceCenterizer();
	
	virtual void build(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);
};


template <typename T>
VarianceCenterizer<T>::VarianceCenterizer(){

}


template <typename T>
VarianceCenterizer<T>::~VarianceCenterizer(){

}


template <typename T>
__global__ void buildVarianceCenterMap(
	T*		pImagesInput, 
	T*		pCenterMap, 
	size_t	pAlignedImageUnitSize, 
	size_t	pImageCount)
{
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pAlignedImageUnitSize)
		return;

	float sum = 0UL;
	for(size_t i=0UL; i<pImageCount; ++i)
		sum += (*(pImagesInput + pAlignedImageUnitSize * i + idx));

	float mean = static_cast<float>(sum) / static_cast<float>(pImageCount);

	float var = 0;
	for(size_t i=0UL; i<pImageCount; ++i){
		float value = static_cast<float>(*(pImagesInput + pAlignedImageUnitSize * i + idx));
		var += (value - mean) * (value - mean); 
	}
	var = var / pImageCount;

	*(pCenterMap + idx) = static_cast<T>(mean * (10.0F - var));
}

/*
template <>
__global__ void buildVarianceCenterMap<uchar>(
	uchar*	pImagesInput, 
	uchar*	pCenterMap, 
	size_t	pAlignedImageUnitSize, 
	size_t	pImageCount)
{
	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 packagesCount	= pAlignedImageUnitSize >> 2;
	if(idx >= packagesCount)
		return;

	size_t sum0 = 0UL;
	size_t sum1 = 0UL;
	size_t sum2 = 0UL;
	size_t sum3 = 0UL;

	uint32* imgs = reinterpret_cast<uint32*>(pImagesInput);
	uint32* cntm = reinterpret_cast<uint32*>(pCenterMap);

	for(size_t i=0UL; i<pImageCount; ++i){
		uint32 val = *(imgs + packagesCount * i + idx);
		sum0 += (val & 0x000000FF);
		sum1 += (val & 0x0000FF00) >> 8;
		sum2 += (val & 0x00FF0000) >> 16;
		sum3 += (val & 0xFF000000) >> 24;
	}

	*(cntm + idx) = 
		((sum3 / pImageCount) << 24	)	| 
		((sum2 / pImageCount) << 16	)	| 
		((sum1 / pImageCount) << 8	)	| 
		((sum0 / pImageCount)		);
}
*/

template <typename T> 
void VarianceCenterizer<T>::build(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t units	= pImageBatch.getAlignedImageByteSize() / sizeof(T);
	size_t blocks	= blocksCount(units, config::Cuda::THREADS_PER_BLOCK);
	
	buildVarianceCenterMap<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		units,
		pImageBatch.getImagesCount());
}


	}
}

#endif	/* CNN_VARIANCE_CENTERIZER_H_ */