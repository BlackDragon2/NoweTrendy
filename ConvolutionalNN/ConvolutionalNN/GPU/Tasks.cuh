
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
	uint	idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize = pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	size_t sum0 = 0UL;
	size_t sum1 = 0UL;
	size_t sum2 = 0UL;
	size_t sum3 = 0UL;

	uint* imgs = reinterpret_cast<uint*>(pImagesInput);
	uint* cntm = reinterpret_cast<uint*>(pCenterMap);

	for(size_t i=0UL; i<pImageCount; ++i){
		uint val = *(imgs + alignImage4xSize * i + idx);
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
	uint	idx					= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize	= pAlignedImageByteSize >> 2;

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
	uint	idx					= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize	= pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	uint* imgs = reinterpret_cast<uint*>(pImagesInput);
	uint* cntm = reinterpret_cast<uint*>(pCenterMap);
	uint* oupt = reinterpret_cast<uint*>(pImagesOutput);
	
	size_t means = *(cntm + idx);
	uint mean0 = means & 0x000000FF;
	uint mean1 = means & 0x0000FF00;
	uint mean2 = means & 0x00FF0000;
	uint mean3 = means & 0xFF000000;
	
	for(size_t i=0UL; i<pImageCount; ++i){
		uint*	adr = imgs + alignImage4xSize * i + idx;
		uint	val = *adr;
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
	uint	idx					= ((blockIdx.x * blockDim.x) + threadIdx.x);
	size_t	alignImage4xSize	= pAlignedImageByteSize >> 2;

	if(idx >= alignImage4xSize)
		return;

	for(size_t i=0UL; i<pImageCount; ++i){
		float val	= *(pImagesInput + alignImage4xSize * i + idx);
		float mean	= *(pCenterMap + alignImage4xSize * i + idx);
		*(pImagesOutput + alignImage4xSize * i + idx) = MAX(0.0F, val - mean);
	}
}



// per row, why not
template <typename T, typename U>
__global__ void erode(
	T*		pImagesInput, 
	U*		pImagesOutput,
	size_t	pImageRowAlignmentUnitSize, 
	size_t	pImageAlignedUnitSize,
	size_t	pImageHeight, 
	size_t	pImageCount,
	size_t	pImageChannels,
	U		pMultiplier,
	T		pBoundaries[6])
{
	uint idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint rowsCount	= pImageCount * pImageHeight;

	if(idx >= rowsCount)
		return;

	uint	myimg	= static_cast<uint>(idx / pImageCount);
	T*		inrow	= pImagesInput + myimg * pImageAlignedUnitSize + pImageRowAlignmentUnitSize * (idx % pImageHeight);
	U*		outrow	= pImagesOutput + myimg * pImageAlignedUnitSize + pImageRowAlignmentUnitSize * (idx % pImageHeight);

	for(size_t i=0UL; i<pImageRowAlignmentUnitSize; i+=pImageChannels){
		for(size_t c=0UL; c<pImageChannels; c+=2){
			*(outrow + i) = pMultiplier * (static_cast<float>((*(inrow + i)) - pBoundaries[c]) / pBoundaries[c + 1UL]);
		}
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
};



template <typename T> 
void Tasks::buildCenterMap(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = static_cast<size_t>(std::ceil(
		static_cast<double>(pImageBatch.getAlignmentImageByteSize() >> 2) / THREADS));
	
	buildCenterMapK<<<blocks, THREADS>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignmentImageByteSize(),
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
		static_cast<double>(pImageBatch.getAlignmentImageByteSize() >> 2) / THREADS));

	centerizeWithMapK<<<blocks, THREADS>>>(
		pImagesBuffer.getDataPtr<T>(), 
		pCenterMapBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignmentImageByteSize(),
		pImageBatch.getImagesCount());
}


	}
}


#endif	/* CNN_TASKS_H_ */