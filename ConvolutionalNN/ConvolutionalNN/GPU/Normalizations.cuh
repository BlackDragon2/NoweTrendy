
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

/*
template <typename From, typename To>
__global__ void normalize(
	From*	pImages, 
	From	pInputMaxValue,
	size_t	pInputUnitsSize,
	size_t	pInputPackDensity,
	To*		pOutputData)
{
	unsigned int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pSize)
		return;

	while(pInputPackDensity-- > 0UL){
		To val = static_cast<To>(*(pImages + idx + pInputPackDensity)) / pMaxValue;
		*(pOutputData + idx + pInputPackDensity) = val;
	}
}
*/


namespace cnn {
	namespace gpu {


class Normalizations {
private:
	static const size_t THREADS = 512UL;
/*
public:
	template <typename T>
	static void centerize(
		ImagesBatch<T>::PtrS&	pImageBatch, 
		GpuBuffer<T>&			pBuffer,
		size_t					pThreadsPerUnit = 1UL);

	
	template <typename From, typename To>
	static void normalize(
		GpuBuffer<From>&	pBufferInput,
		From				pMaxValue,
		size_t				pInputPackDensity,
		GpuBuffer<To>&		pBufferOutput);
};


// threadsPerUnit:
// uchar	1
// uint		1
// float	TODO	4 
template <typename T>
void Normalizations::centerize(
	ImagesBatch<T>::PtrS&	pImageBatch, 
	GpuBuffer<T>&			pBuffer,
	size_t					pThreadsPerUnit)
{
	T*		imgsData	= &pBuffer;
	size_t	imgSize		= pImageBatch->getImageUnitSize();
	size_t	aligImgSize	= pImageBatch->getAlignedImageUnitSize();
	size_t	imgCount	= pImageBatch->getImagesCount(); 
	size_t	blocks		= static_cast<size_t>(std::ceil(
		static_cast<double>(imgSize * pThreadsPerUnit) / THREADS));
	centerImages<<<blocks, THREADS>>>(imgsData, imgSize, aligImgSize, imgCount);
}


template <typename From, typename To>
void Normalizations::normalize(
	GpuBuffer<From>&	pBufferInput,
	From				pMaxValue,
	size_t				pInputPackDensity,
	GpuBuffer<To>&		pBufferOutput)
{
	From*	imgsData	= &pBufferInput;
	To*		outputData	= &pBufferOutput;	
	size_t	unitSize	= pBufferInput->getBufferUnitSize();
	size_t	blocks		= static_cast<size_t>(std::ceil(
		static_cast<double>(pBufferInput->getBufferByteSize()) / (THREADS * pInputPackDensity)));
	normalize<<<blocks, THREADS>>>(imgsData, pMaxValue, unitSize, pInputPackDensity, outputData);*/
};


	}
}


#endif	/* CNN_NORMALIZATIONS_H_ */