
#ifndef CNN_TASKS_H_
#define CNN_TASKS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GpuBuffer.cuh"
#include "../Types.h"
#include "../ImageBatch.h"


////////// build center map
__global__ void buildCenterMapK(
	uchar*	pImagesInput, 
	uchar*	pCenterMap, 
	size_t	pAlignedImageByteSize, 
	size_t	pImageCount)
{
	uint32	idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize = pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	size_t sum0 = 0UL;
	size_t sum1 = 0UL;
	size_t sum2 = 0UL;
	size_t sum3 = 0UL;

	uint32* imgs = reinterpret_cast<uint32*>(pImagesInput);
	uint32* cntm = reinterpret_cast<uint32*>(pCenterMap);

	for(size_t i=0UL; i<pImageCount; ++i){
		uint32 val = *(imgs + alignImage4xSize * i + idx);
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


__global__ void buildCenterMapK(
	float*	pImagesInput, 
	float*	pCenterMap, 
	size_t	pAlignedImageByteSize, 
	size_t	pImageCount)
{
	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t alignImage4xSize	= pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	float sum = 0UL;
	for(size_t i=0UL; i<pImageCount; ++i)
		sum += (*(pImagesInput + alignImage4xSize * i + idx));

	*(pCenterMap + idx) = sum / pImageCount;
}


//////////// centerize with map
__global__ void centerizeWithMapK(
	uchar*	pImagesInput, 
	uchar*	pCenterMap,
	uchar*	pImagesOutput,
	size_t	pAlignedImageByteSize, 
	size_t	pImageCount)
{
	uint32	idx					= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize	= pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	uint32* imgs = reinterpret_cast<uint32*>(pImagesInput);
	uint32* cntm = reinterpret_cast<uint32*>(pCenterMap);
	uint32* oupt = reinterpret_cast<uint32*>(pImagesOutput);
	
	size_t means = *(cntm + idx);
	uint32 mean0 = means & 0x000000FF;
	uint32 mean1 = means & 0x0000FF00;
	uint32 mean2 = means & 0x00FF0000;
	uint32 mean3 = means & 0xFF000000;
	
	for(size_t i=0UL; i<pImageCount; ++i){
		uint32*	adr = imgs + alignImage4xSize * i + idx;
		uint32	val = *adr;
		val -= MIN(mean0, val & 0x000000FF);
		val -= MIN(mean1, val & 0x0000FF00);
		val -= MIN(mean2, val & 0x00FF0000);
		val -= MIN(mean3, val & 0xFF000000);
		*(oupt + alignImage4xSize * i + idx) = val;
	}
}


__global__ void centerizeWithMapK(
	float* pImagesInput, 
	float* pCenterMap,
	float* pImagesOutput,
	size_t	pAlignedImageByteSize, 
	size_t	pImageCount)
{
	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t alignImage4xSize	= pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	for(size_t i=0UL; i<pImageCount; ++i){
		float val	= *(pImagesInput + alignImage4xSize * i + idx);
		float mean	= *(pCenterMap + alignImage4xSize * i + idx);
		*(pImagesOutput + alignImage4xSize * i + idx) = MAX(0.0F, val - mean);
	}
}


//////////////// finding images boundaries
// output size is:	imgs_count * imgs_channels * 2
// structure is:	[c1_min, c1_max, c2_min, c2_max, ...]
__global__ void findEachImageBoundariesK(
	uchar*	pImagesInput, 
	uchar*	pBoundariesOutput,
	size_t	pAlignedImageByteSize, 
	size_t	pAlignedImageRowByteSize, 
	size_t	pImageWidth,
	size_t	pImageHeight,
	size_t	pImageChannels,
	size_t	pImageCount)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 threads	= pImageCount * pImageChannels;
	if(idx >= threads)
		return;
	
	uint32 myImage		= idx / pImageChannels;
	uint32 myChannel	= idx % pImageChannels;
	uchar* myimg		= pImagesInput + myImage * pAlignedImageByteSize;

	uchar min			= 255;
	uchar max			= 0;
	size_t rowLength	= pImageWidth * pImageChannels;

	for(size_t i=0UL; i<pImageHeight; ++i){
		for(size_t re=myChannel; re<rowLength; re+=pImageChannels){
			uchar value = *(myimg + i * pAlignedImageRowByteSize + re);
			min = MIN(min, value);
			max = MAX(max, value);
		}
	}

	// TODO change from uchar to short?
	uchar* cntm	= pBoundariesOutput + myImage * (pImageChannels << 1) + (myChannel << 1);
	(*(cntm)	) = min;
	(*(cntm + 1)) = max;
}


//////////////// erode per column, per channel, why not
__global__ void erodeEachImageUsingBoundariesK(
	uchar*	pImagesInput, 
	uchar*	pImageBoundaries,
	uchar*	pImagesOutput,
	size_t	pImageRowAlignmentByteSize, 
	size_t	pImageAlignedByteSize,
	size_t	pImageHeight, 
	size_t	pImageCount,
	size_t	pImageChannels,
	uchar	pMultiplier)
{
	uint32 idx			= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 colsCount	= pImageCount * pImageRowAlignmentByteSize;

	if(idx >= colsCount)
		return;

	uint32 myImage		= idx / pImageRowAlignmentByteSize;
	uint32 myCol		= idx % pImageRowAlignmentByteSize;
	uint32 myChannel	= idx % pImageChannels;

	uchar min	= *(pImageBoundaries + myImage * pImageChannels + myChannel * 2);
	uchar max	= *(pImageBoundaries + myImage * pImageChannels + myChannel * 2 + 1);
	uchar diff	= max - min;

	for(size_t i=0UL; i<pImageHeight; ++i){
		uchar val = *(pImagesInput + myImage * pImageAlignedByteSize + i * pImageRowAlignmentByteSize + myCol);
		uchar res = static_cast<uchar>((static_cast<float>(val - min) / diff) * pMultiplier);
		*(pImagesOutput + myImage * pImageAlignedByteSize + i * pImageRowAlignmentByteSize + myCol) = res;
	}
}


namespace cnn {
	namespace gpu {


class Tasks {
private:
	static const size_t THREADS = 512UL;

public:
	template <typename T> static void buildCenterMap(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);

	template <typename T> static void centerizeWithMap(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pImagesBuffer,
		GpuBuffer&				pCenterMapBuffer,
		GpuBuffer&				pOutputBuffer);

	template <typename T> static void findEachImageBoundaries(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);

	template <typename T> static void erodeEachImageUsingBoundaries(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pBoundaries,
		GpuBuffer&				pOutputBuffer,
		T						pMultiplier);
};



template <typename T> 
void Tasks::buildCenterMap(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = static_cast<size_t>(std::ceil(
		static_cast<double>(pImageBatch.getAlignedImageByteSize() >> 2) / THREADS));
	
	buildCenterMapK<<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageByteSize(),
		pImageBatch.getImagesCount());
}


template <typename T>
void Tasks::centerizeWithMap(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pImagesBuffer,
	GpuBuffer&				pCenterMapBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = static_cast<size_t>(std::ceil(
		static_cast<double>(pImageBatch.getAlignedImageByteSize() >> 2) / THREADS));

	centerizeWithMapK<<<blocks, THREADS>>>(
		pImagesBuffer.getDataPtr<T>(), 
		pCenterMapBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageByteSize(),
		pImageBatch.getImagesCount());
}


template <typename T> 
void Tasks::findEachImageBoundaries(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = static_cast<size_t>(std::ceil(
		static_cast<double>(pImageBatch.getImagesCount() * pImageBatch.getImageChannelsCount()) / THREADS));

	findEachImageBoundariesK<<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageByteSize(),
		pImageBatch.getAlignedImageRowByteSize(),
		pImageBatch.getImageWidth(),
		pImageBatch.getImageHeight(),
		pImageBatch.getImageChannelsCount(),
		pImageBatch.getImagesCount());
}


template <typename T>
void Tasks::erodeEachImageUsingBoundaries(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pBoundaries,
	GpuBuffer&				pOutputBuffer,
	T						pMultiplier)
{
	size_t blocks = static_cast<size_t>(std::ceil(static_cast<double>(
		pImageBatch.getImagesCount() * 
		pImageBatch.getImageChannelsCount() * 
		pImageBatch.getAlignedImageWidth()) / THREADS));

	erodeEachImageUsingBoundariesK<<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pBoundaries.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageRowByteSize(),
		pImageBatch.getAlignedImageByteSize(),
		pImageBatch.getImageHeight(),
		pImageBatch.getImagesCount(),
		pImageBatch.getImageChannelsCount(),
		pMultiplier);
}

/*
uchar*	pImagesInput, 
uchar*	pImageBoundaries,
uchar*	pImagesOutput,
size_t	pImageRowAlignmentByteSize, 
size_t	pImageAlignedByteSize,
size_t	pImageHeight, 
size_t	pImageCount,
size_t	pImageChannels,
uchar	pMultiplier
*/

	}
}


#endif	/* CNN_TASKS_H_ */