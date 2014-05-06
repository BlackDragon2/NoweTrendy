
#ifndef CNN_AVERAGE_CENTERIZER_H_
#define CNN_AVERAGE_CENTERIZER_H_

#include "Normalizer.h"


namespace cnn {
	namespace gpu {


template <typename T>
class AverageCenterizer : public Normalizer<T> {
public:
	AverageCenterizer();
	virtual ~AverageCenterizer();
	
	virtual void build(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);

	virtual void normalize(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pBuilderBuffer,
		GpuBuffer&				pOutputBuffer);
};


template <typename T>
AverageCenterizer<T>::AverageCenterizer(){

}


template <typename T>
AverageCenterizer<T>::~AverageCenterizer(){

}


template <typename T>
__global__ void buildMeanCenterMap(
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

	*(pCenterMap + idx) = sum / pImageCount;
}

template <>
__global__ void buildMeanCenterMap<uchar>(
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

template <typename T> 
void AverageCenterizer<T>::build(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t units	= pImageBatch.getAlignedImageByteSize() / sizeof(T);
	size_t blocks	= utils::blocksCount(units, config::Cuda::THREADS_PER_BLOCK);
	
	buildMeanCenterMap<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		units,
		pImageBatch.getImagesCount());
}


template <typename T>
__global__ void centerizeWithMap(
	T*		pImagesInput, 
	T*		pCenterMap,
	T*		pImagesOutput,
	size_t	pAlignedImageUnitSize, 
	size_t	pImageCount)
{
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pAlignedImageUnitSize)
		return;

	T mean = *(pCenterMap + idx);
	for(size_t i=0UL; i<pImageCount; ++i){
		T val = *(pImagesInput + pAlignedImageUnitSize * i + idx);
		*(pImagesOutput + pAlignedImageUnitSize * i + idx) = (val >= mean ? val - mean : 0.0F);
	}
}

template <>
__global__ void centerizeWithMap<uchar>(
	uchar*	pImagesInput, 
	uchar*	pCenterMap,
	uchar*	pImagesOutput,
	size_t	pAlignedImageUnitSize,
	size_t	pImageCount)
{
	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 packagesCount	= pAlignedImageUnitSize >> 2;
	if(idx >= packagesCount)
		return;

	uint32* imgs = reinterpret_cast<uint32*>(pImagesInput);
	uint32* cntm = reinterpret_cast<uint32*>(pCenterMap);
	uint32* oupt = reinterpret_cast<uint32*>(pImagesOutput);
	
	uint32 means = *(cntm + idx);
	uint32 mean0 = means & 0x000000FF;
	uint32 mean1 = means & 0x0000FF00;
	uint32 mean2 = means & 0x00FF0000;
	uint32 mean3 = means & 0xFF000000;
	
	for(size_t i=0UL; i<pImageCount; ++i){
		uint32*	adr = imgs + packagesCount * i + idx;
		uint32	val = *adr;
		val -= MIN(mean0, val & 0x000000FF);
		val -= MIN(mean1, val & 0x0000FF00);
		val -= MIN(mean2, val & 0x00FF0000);
		val -= MIN(mean3, val & 0xFF000000);
		*(oupt + packagesCount * i + idx) = val;
	}
}

template <typename T>
void AverageCenterizer<T>::normalize(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pBuilderBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t units	= pImageBatch.getAlignedImageByteSize() / sizeof(T);
	size_t blocks	= static_cast<size_t>(std::ceil(static_cast<double>(units) / config::Cuda::THREADS_PER_BLOCK));

	centerizeWithMap<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<T>(), 
		pBuilderBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		units,
		pImageBatch.getImagesCount());
}


	}
}

#endif	/* CNN_AVERAGE_CENTERIZER_H_ */