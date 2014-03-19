
#ifndef CNN_NORMALIZATIONS_H_
#define CNN_NORMALIZATIONS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Types.h"
#include "GpuBuffer.cuh"
#include "../ImagesBatch.h"


__global__ void meanImageFrom(
	uchar const*	pImages, 
	size_t			pImageSize, 
	size_t			pImageCount, 
	uchar*			pOutput)
{
	size_t			sum = 0UL;
	unsigned int	idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	for(size_t i=0UL; i<pImageCount; ++i)
		sum += *(pImages + pImageSize * i + idx);

	pOutput[idx] = static_cast<uchar>(sum / pImageCount);
}


__global__ void centerImagesWith(
	uchar*			pImages, 
	size_t			pImageSize, 
	size_t			pImageCount, 
	uchar const*	pMeanValues)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	for(size_t i=0UL; i<pImageCount; ++i)
		*(pImages + pImageSize * i + idx) -= MIN(pMeanValues[idx], *(pImages + pImageSize * i + idx));
}


__global__ void centerImages(
	uchar* pImages, 
	size_t pImageSize, 
	size_t pImageCount)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(idx >= pImageSize)
		return;

	size_t sum = 0UL;
	for(size_t i=0UL; i<pImageCount; ++i)
		sum += *(pImages + pImageSize * i + idx);

	uchar mean = static_cast<uchar>(sum / pImageCount);

	for(size_t i=0UL; i<pImageCount; ++i)
		*(pImages + pImageSize * i + idx) -= MIN(mean, *(pImages + pImageSize * i + idx));
}


// about 2.8x times faster tha uchar
__global__ void centerImages(
	uint*	pImages, 
	size_t	pImageSize, 
	size_t	pImageCount)
{
	unsigned int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pImageSize)
		return;

	size_t sum0 = 0UL;
	size_t sum1 = 0UL;
	size_t sum2 = 0UL;
	size_t sum3 = 0UL;
	for(size_t i=0UL; i<pImageCount; ++i){
		uint val = *(pImages + pImageSize * i + idx);
		uint mod = 0xFF;
		sum0 += (val & (mod			));
		sum1 += (val & (mod << 8	)) >> 8;
		sum2 += (val & (mod << 16	)) >> 16;
		sum3 += (val & (mod << 24	)) >> 24;
	}

	size_t mean0 = (sum0 / pImageCount);
	size_t mean1 = (sum1 / pImageCount) << 8;
	size_t mean2 = (sum2 / pImageCount) << 16;
	size_t mean3 = (sum3 / pImageCount) << 24;

	for(size_t i=0UL; i<pImageCount; ++i){
		size_t* adr = pImages + pImageSize * i + idx;
		uint	val = *adr;
		uint	res = 0UL;
		uint	mod = 0xFF;
		val -= MIN(mean0, val & (mod		));
		val -= MIN(mean1, val & (mod << 8	));
		val -= MIN(mean2, val & (mod << 16	));
		val -= MIN(mean3, val & (mod << 24	));
		*(adr) = val;
	}
}


namespace cnn {
	namespace gpu {


template <typename T>
class Normalizations {
public:
	static void centerize(ImagesBatch<T>::PtrS& pImageBatch, GpuBuffer<T>& pBuffer);
	static void centerize(ImagesBatch<T>::PtrS& pImageBatch);
};


template <typename T>
void Normalizations<T>::centerize(ImagesBatch<T>::PtrS& pImageBatch, GpuBuffer<T>& pBuffer){
	T* imgsData		= &pBuffer;
	size_t imgSize	= pImageBatch->getAlignedImageUnitSize();
	size_t imgCount	= pImageBatch->getImagesCount(); 
	size_t blocks	= static_cast<size_t>(std::ceil(static_cast<double>(imgSize) / 512));
	centerImages<<<blocks, 512>>>(imgsData, imgSize, imgCount);
}


template <typename T>
void Normalizations<T>::centerize(ImagesBatch<T>::PtrS& pImageBatch){
	//centerize(pImageBatch, GpuBuffer<T>(pImageBatch->getAlignedImageByteSize(), pImageBatch->getImagesData()));
}


	}
}


#endif	/* CNN_NORMALIZATIONS_H_ */