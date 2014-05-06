
#ifndef CNN_MAX_POOLING_H_
#define CNN_MAX_POOLING_H_

#include "Sampler.h"
#include "../Utils/Utils.h"
#include "../Config.h"


namespace cnn {
	namespace gpu {


template <typename T>
class MaxPooling {
public:
	struct InputImageParams {
		T*		buffer;
		size_t	imageWidth;
		size_t	imageHeight;
		size_t	alignedImageUnitSize;
		size_t	alignedImageRowUnitSize;
	};

	struct OutputImageParams {
		T*		buffer;
		size_t	alignedImageUnitSize;
		size_t	alignedImageRowUnitSize;
	};

	struct GeneralParams {
		size_t	imagesCount;
		size_t	sampleWidth;
		size_t	sampleHeight;
		size_t	threadsPerRow;
		size_t	threadsPerCol;
		size_t	imageChannels,
	}

public:
	MaxPooling();
	virtual ~MaxPooling();
	
	virtual void sample(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer,
		size_t					pSampleWidth,
		size_t					pSampleHeight);
};


template <typename T>
MaxPooling<T>::MaxPooling(){

}


template <typename T>
MaxPooling<T>::~MaxPooling(){

}


template <typename T>
__global__ void samplerMaxPooling(
	typename MaxPooling<T>::InputImageParams pInputImageParams,
	typename MaxPooling<T>::OutputImageParams pOutputImageParams,
	typename MaxPooling<T>::GeneralParams pGeneralParams)
{
	typename MaxPooling<T>::InputImageParams	ip = pInputImageParams;
	typename MaxPooling<T>::OutputImageParams	op = pOutputImageParams;
	typename MaxPooling<T>::GeneralParams		gp = pGeneralParams;

	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 threadsPerImage	= gp.threadsPerRow * gp.threadsPerCol;

	if(idx >= threadsPerImage * gp.pImagesCount)
		return;

	uint32 myImg	= idx / threadsPerImage;
	uint32 myColumn	= threadsPerImage % gp.threadsPerRow;			
	uint32 myRow	= threadsPerImage / gp.threadsPerRow;

	uint32 skipX = gp.sampleWidth * gp.imageChannels;
	uint32 skipY = gp.sampleHeight * ip.alignedImageRowUnitSize;

	T* myImageRect = ip.buffer + myImg * ip.alignedImageUnitSize + 
		myRow * skipY + myColumn * skipX;
	T* myOutputPixel = op.buffer + myImg * op.alignedImageUnitSize +
		myRow * op.alignedImageRowUnitSize + myColumn * gp.imageChannels;

	T maximes[3] = {0, 0, 0};

	for(uint32 row=0U; row<gp.sampleHeight; ++row){
		T* myImageRow = myImageRect + row * ip.alignedImageRowUnitSize;
		for(uint32 col=0U; col<gp.sampleWidth; ++col){
			T* myImagePixel = myImageRow + col * gp.imageChannels;
			for(uint32 ch=0U; ch<gp.imageChannels; ++ch){
				T value = *(myImagePixel + ch);
				if(value > maximes[ch])
					maximes[ch] = value;
			}
		}
	}

	for(uint32 ch=0U; ch<pSampleWidth; ++ch){
		myOutputPixel[ch] = maximes[ch];
}


template <typename T>
void MaxPooling<T>::sample(
	ImageBatch<T> const&	pImageBatch, 
	GpuBuffer&				pInputBuffer,
	GpuBuffer&				pOutputBuffer,
	size_t					pSampleWidth,
	size_t					pSampleHeight)
{
	size_t rowThreads		= static_cast<size_t>(std::ceil(static_cast<double>(pImageBatch.getImageWidth()) / pSampleWidth));
	size_t colThreads		= static_cast<size_t>(std::ceil(static_cast<double>(pImageBatch.getImageHeight()) / pSampleHeight));
	size_t blocks			= utils::blocksCount(rowThreads * colThreads * pImageBatch.getImagesCount(), config::Cuda::THREADS_PER_BLOCK);

	samplerMaxPooling<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(
		pInputBuffer.getDataPtr(),
		pOutputBuffer.getDataPtr(),
		pImageBatch.getImagesCount(),
		pImageBatch.getImageWidth(),
		pImageBatch.getImageHeight(),
		pImageBatch.getImageChannelsCount(),
		pImageBatch.getAlignedImageByteSize() / sizeof(T),
		pImageBatch.getAlignedImageRowByteSize() / sizeof(T),
		pSampleWidth,
		pSampleHeight,
		rowThreads,
		colThreads);
}


	}
}

#endif	/* CNN_MAX_POOLING_H_ */