
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
	size_t pUnitImageSize, 
	size_t pAlignedUnitImageSize,  
	size_t pImageCount)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(idx >= pUnitImageSize)
		return;

	size_t sum = 0UL;
	for(size_t i=0UL; i<pImageCount; ++i)
		sum += *(pImages + pAlignedUnitImageSize * i + idx);

	uchar mean = static_cast<uchar>(sum / pImageCount);

	for(size_t i=0UL; i<pImageCount; ++i)
		*(pImages + pAlignedUnitImageSize * i + idx) -= MIN(mean, *(pImages + pAlignedUnitImageSize * i + idx));
}


// about 3.12x times faster than uchar for 20 images
// about 3.47x times faster than uchar for 200 images
__global__ void centerImages(
	uint*	pImages, 
	size_t	pUnitImageSize, 
	size_t	pAlignedUnitImageSize, 
	size_t	pImageCount)
{
	unsigned int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pUnitImageSize)
		return;

	size_t sum0 = 0UL;
	size_t sum1 = 0UL;
	size_t sum2 = 0UL;
	size_t sum3 = 0UL;

	for(size_t i=0UL; i<pImageCount; ++i){
		uint val = *(pImages + pAlignedUnitImageSize * i + idx);
		sum0 += (val & 0x000000FF);
		sum1 += (val & 0x0000FF00) >> 8;
		sum2 += (val & 0x00FF0000) >> 16;
		sum3 += (val & 0xFF000000) >> 24;
	}

	size_t mean0 = (sum0 / pImageCount);
	size_t mean1 = (sum1 / pImageCount) << 8;
	size_t mean2 = (sum2 / pImageCount) << 16;
	size_t mean3 = (sum3 / pImageCount) << 24;

	for(size_t i=0UL; i<pImageCount; ++i){
		size_t* adr = pImages + pAlignedUnitImageSize * i + idx;
		uint	val = *adr;
		val -= MIN(mean0, val & 0x000000FF);
		val -= MIN(mean1, val & 0x0000FF00);
		val -= MIN(mean2, val & 0x00FF0000);
		val -= MIN(mean3, val & 0xFF000000);
		*(adr) = val;
	}
}


template <typename T, typename U>
__global__ void normalize(
	T*		pInput,
	size_t	pInputSize,
	T		pInputMultiplier,
	U*		pOutput)
{
	unsigned int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pInputSize)
		return;


}


namespace cnn {
	namespace gpu {


class Normalizations {
private:
	static const size_t THREADS = 512UL;

public:
	template <typename T> static void centerize(
		ImagesBatch::PtrS&	pImageBatch, 
		GpuBuffer&			pBuffer);

	template <typename T, typename U> static void normalize(
		GpuBuffer& pInput,
		GpuBuffer& pOutput);
};



template <typename T>
void Normalizations::centerize(
	ImagesBatch::PtrS&	pImageBatch, 
	GpuBuffer&			pBuffer)
{
	T*		imgsData		= pBuffer.dataPtr<T>();
	size_t	imgUnitSize		= pImageBatch->getImageByteSize();
	size_t	aligImgUnitSize	= pImageBatch->getAlignedImageByteSize() / sizeof(T);
	size_t	imgCount		= pImageBatch->getImagesCount(); 
	size_t	blocks			= static_cast<size_t>(std::ceil(
		static_cast<double>(aligImgUnitSize) / THREADS));
	centerImages<<<blocks, THREADS>>>(imgsData, imgUnitSize, aligImgUnitSize, imgCount);
}


template <typename T, typename U> 
static void normalize(
	GpuBuffer&	pInput,
	T			pInputMultiplier,
	GpuBuffer&	pOutput)
{
	T*		input		= pInput.dataPtr<T>();
	U*		output		= pInput.dataPtr<U>();
	size_t	inputSize	= pInput.getByteSize() / sizeof(T);

	size_t blocks = static_cast<size_t>(std::ceil(
		static_cast<double>(inputSize) / THREADS));

	normalize<T, U><<<blocks, THREADS>>>(input, inputSize, pInputMultiplier, output);
}


	}
}


#endif	/* CNN_NORMALIZATIONS_H_ */