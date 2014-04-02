
#ifndef CNN_TASKS_H_
#define CNN_TASKS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GpuBuffer.cuh"
#include "../Types.h"
#include "../ImageBatch.h"



/////////// type conversion
template <typename F, typename T>
__global__ void convertK(
	F*		pImagesInput,
	T*		pImagesOutput,
	size_t	pElements)
{
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pElements)
		return;

	*(pImagesOutput + idx) = static_cast<T>(*(pImagesInput + idx));
}


////////// build center map
template <typename T>
__global__ void buildCenterMapK(
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
__global__ void buildCenterMapK<uchar>(
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


//////////// centerize with map
template <typename T>
__global__ void centerizeWithMapK(
	T*		pImagesInput, 
	T*		pCenterMap,
	T*		pImagesOutput,
	size_t	pAlignedImageUnitSize, 
	size_t	pImageCount)
{
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= pAlignedImageUnitSize)
		return;

	for(size_t i=0UL; i<pImageCount; ++i){
		float val	= *(pImagesInput + pAlignedImageUnitSize * i + idx);
		float mean	= *(pCenterMap + pAlignedImageUnitSize * i + idx);
		*(pImagesOutput + pAlignedImageUnitSize * i + idx) = MAX(0.0F, val - mean);
	}
}


template <>
__global__ void centerizeWithMapK<uchar>(
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


//////////////// finding images boundaries
// output size is:	imgs_count * imgs_channels * 2
// structure is:	[c1_min, c1_max, c2_min, c2_max, ...]
template <typename T>
__global__ void findEachImageBoundariesK(
	T*		pImagesInput, 
	T*		pBoundariesOutput,
	size_t	pAlignedImageUnitSize, 
	size_t	pAlignedImageRowUnitSize, 
	size_t	pImageWidth,
	size_t	pImageHeight,
	size_t	pImageChannels,
	size_t	pImageCount,
	T		pMinValue = std::numeric_limits<T>::min(),
	T		pMaxValue = std::numeric_limits<T>::max())
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 threads	= pImageCount * pImageChannels;
	if(idx >= threads)
		return;
	
	uint32	myImage		= idx / pImageChannels;
	uint32	myChannel	= idx % pImageChannels;
	T*		myimg		= pImagesInput + myImage * pAlignedImageUnitSize;

	T		min			= pMaxValue;
	T		max			= pMinValue;
	size_t	rowLength	= pImageWidth * pImageChannels;

	for(size_t i=0UL; i<pImageHeight; ++i){
		for(size_t re=myChannel; re<rowLength; re+=pImageChannels){
			T value = *(myimg + i * pAlignedImageRowUnitSize + re);
			if(min > value)
				min = value;
			if(max < value)
				max = value;
		}
	}

	// TODO change from uchar to short?
	T* cntm	= pBoundariesOutput + myImage * ((pImageChannels + myChannel) << 1);
	(*(cntm)	) = min;
	(*(cntm + 1)) = max;
}


//////////////// erode per column, per channel, why not
template <typename T>
__global__ void erodeEachImageUsingBoundariesK(
	T*		pImagesInput, 
	T*		pImageBoundaries,
	T*		pImagesOutput,
	size_t	pImageRowAlignmentUnitSize, 
	size_t	pImageAlignedUnitSize,
	size_t	pImageHeight, 
	size_t	pImageCount,
	size_t	pImageChannels,
	T		pMultiplier)
{
	uint32 idx			= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 colsCount	= pImageCount * pImageRowAlignmentUnitSize;
	if(idx >= colsCount)
		return;

	uint32 myImage		= idx / pImageRowAlignmentUnitSize;
	uint32 myCol		= idx % pImageRowAlignmentUnitSize;
	uint32 myChannel	= idx % pImageChannels;

	T min	= *(pImageBoundaries + myImage * ((pImageChannels + myChannel) << 1));
	T max	= *(pImageBoundaries + myImage * ((pImageChannels + myChannel) << 1) + 1);
	T diff	= max - min;// > 0 ? max - min : 1;

	for(size_t i=0UL; i<pImageHeight; ++i){
		T val = *(pImagesInput + myImage * pImageAlignedUnitSize + i * pImageRowAlignmentUnitSize + myCol);
		T res = static_cast<T>((static_cast<float>(val - min) / diff) * pMultiplier);
		*(pImagesOutput + myImage * pImageAlignedUnitSize + i * pImageRowAlignmentUnitSize + myCol) = res;
	}
}


namespace cnn {
	namespace gpu {


class Tasks {
private:
	static const size_t THREADS = 512UL;

public:
	template <typename F, typename T> static void convert(
		ImageBatch<F> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);

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


template <typename F, typename T> 
void Tasks::convert(
	ImageBatch<F> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t elems = pImageBatch.getBatchByteSize() / sizeof(F);
	size_t blocks = static_cast<size_t>(std::ceil(static_cast<double>(elems) / THREADS));
	
	convertK<F, T><<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<F>(),
		pOutputBuffer.getDataPtr<T>(),
		elems);
}


template <typename T> 
void Tasks::buildCenterMap(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t units	= pImageBatch.getAlignedImageByteSize() / sizeof(T);
	size_t blocks	= static_cast<size_t>(std::ceil(static_cast<double>(units) / THREADS));
	
	buildCenterMapK<T><<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		units,
		pImageBatch.getImagesCount());
}


template <typename T>
void Tasks::centerizeWithMap(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pImagesBuffer,
	GpuBuffer&				pCenterMapBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t units	= pImageBatch.getAlignedImageByteSize() / sizeof(T);
	size_t blocks	= static_cast<size_t>(std::ceil(static_cast<double>(units) / THREADS));

	centerizeWithMapK<T><<<blocks, THREADS>>>(
		pImagesBuffer.getDataPtr<T>(), 
		pCenterMapBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		units,
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

	findEachImageBoundariesK<T><<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageByteSize() / sizeof(T),
		pImageBatch.getAlignedImageRowByteSize() / sizeof(T),
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
		pImageBatch.getImageRowByteSize() / sizeof(T)) / THREADS));

	erodeEachImageUsingBoundariesK<T><<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pBoundaries.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageRowByteSize() / sizeof(T),
		pImageBatch.getAlignedImageByteSize() / sizeof(T),
		pImageBatch.getImageHeight(),
		pImageBatch.getImagesCount(),
		pImageBatch.getImageChannelsCount(),
		pMultiplier);
}


	}
}


#endif	/* CNN_TASKS_H_ */