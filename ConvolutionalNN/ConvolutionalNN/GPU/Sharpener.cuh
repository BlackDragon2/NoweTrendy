
#ifndef CNN_SHARPENER_H_
#define CNN_SHARPENER_H_

#include "Normalizer.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Sharpener : public Normalizer<T> {
public:
	Sharpener(float pStrength = 255.0F);
	virtual ~Sharpener();
	
	virtual void build(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer);

	virtual void normalize(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pImagesBuffer,
		GpuBuffer&				pCenterMapBuffer,
		GpuBuffer&				pOutputBuffer);


private:
	float mStrength;
};


template <typename T>
Sharpener<T>::Sharpener(float pStrength)
:
	mStrength(pStrength)
{

}


template <typename T>
Sharpener<T>::~Sharpener(){

}


//////////////// finding images boundaries
// output size is:	imgs_count * imgs_channels * 2
// structure is:	[c1_min, c1_max, c2_min, c2_max, ...]
template <typename T>
__global__ void findEachImageBoundaries(
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
	T* cntm	= pBoundariesOutput + myImage * (pImageChannels << 1) + (myChannel << 1);
	(*(cntm)	) = min;
	(*(cntm + 1)) = max;
}

template <typename T> 
void Sharpener<T>::build(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = utils::blocksCount(pImageBatch.getImagesCount() * pImageBatch.getImageChannelsCount(), config::Cuda::THREADS_PER_BLOCK);

	findEachImageBoundaries<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageByteSize() / sizeof(T),
		pImageBatch.getAlignedImageRowByteSize() / sizeof(T),
		pImageBatch.getImageWidth(),
		pImageBatch.getImageHeight(),
		pImageBatch.getImageChannelsCount(),
		pImageBatch.getImagesCount());
}


//////////////// erode per column, per channel, why not
template <typename T>
__global__ void sharpEachImageUsingBoundaries(
	T*		pImagesInput, 
	T*		pImageBoundaries,
	T*		pImagesOutput,
	size_t	pImageRowAlignedUnitSize, 
	size_t	pImageAlignedUnitSize,
	size_t	pImageHeight, 
	size_t	pImageCount,
	size_t	pImageChannels,
	float	pMultiplier)
{
	uint32 idx			= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 colsCount	= pImageCount * pImageRowAlignedUnitSize;
	if(idx >= colsCount)
		return;

	uint32 myImage		= idx / pImageRowAlignedUnitSize;
	uint32 myCol		= idx % pImageRowAlignedUnitSize;
	uint32 myChannel	= myCol % pImageChannels;

	T min	= *(pImageBoundaries + myImage * (pImageChannels << 1) + (myChannel << 1));
	T max	= *(pImageBoundaries + myImage * (pImageChannels << 1) + (myChannel << 1) + 1);
	T diff	= max - min > 0 ? max - min : 1;

	for(size_t i=0UL; i<pImageHeight; ++i){
		T val = *(pImagesInput + myImage * pImageAlignedUnitSize + i * pImageRowAlignedUnitSize + myCol);
		T res = static_cast<T>((pMultiplier * static_cast<float>(val - min)) / diff);
		*(pImagesOutput + myImage * pImageAlignedUnitSize + i * pImageRowAlignedUnitSize + myCol) = res;
	}
}

template <typename T>
void Sharpener<T>::normalize(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pBuilderBuffer,
	GpuBuffer&				pOutputBuffer)
{
	size_t blocks = utils::blocksCount(pImageBatch.getImagesCount() * pImageBatch.getImageRowByteSize() / sizeof(T), config::Cuda::THREADS_PER_BLOCK);
	/*size_t blocks = static_cast<size_t>(
		std::ceil(
			static_cast<double>(
				pImageBatch.getImagesCount() * 
				pImageBatch.getImageRowByteSize() / 
				sizeof(T)
			) 
			/ config::Cuda::THREADS_PER_BLOCK)
		);*/

	sharpEachImageUsingBoundaries<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr<T>(),
		pBuilderBuffer.getDataPtr<T>(),
		pOutputBuffer.getDataPtr<T>(),
		pImageBatch.getAlignedImageRowByteSize() / sizeof(T),
		pImageBatch.getAlignedImageByteSize() / sizeof(T),
		pImageBatch.getImageHeight(),
		pImageBatch.getImagesCount(),
		pImageBatch.getImageChannelsCount(),
		mStrength);
}


	}
}

#endif	/* CNN_SHARPENER_H_ */