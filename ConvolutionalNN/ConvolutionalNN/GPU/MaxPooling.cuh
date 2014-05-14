
#ifndef CNN_MAX_POOLING_H_
#define CNN_MAX_POOLING_H_

#include "Sampler.h"
#include "../Utils/Utils.h"
#include "../Config.h"


namespace cnn {
	namespace gpu {


template <typename T>
class MaxPooling : public Sampler<T> {
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
		size_t	imageChannels;
	};

public:
	MaxPooling(uint32 pWidth, uint32 pHeight);
	virtual ~MaxPooling();
	
	virtual void sample(
		ImageBatch<T> const&	pInputImageBatch, 
		GpuBuffer&				pInputBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputBuffer);
	 
	virtual uint32 countOutputImageUnitSize(ImageBatch<T> const& pInputImageBatch) const;
};


template <typename T>
MaxPooling<T>::MaxPooling(
	uint32 pWidth, 
	uint32 pHeight)
:
	Sampler<T>(pWidth, pHeight)
{

}


template <typename T>
MaxPooling<T>::~MaxPooling(){

}


template <typename T>
__global__ void samplerMaxPooling(
	typename MaxPooling<T>::InputImageParams	pInputImageParams,
	typename MaxPooling<T>::OutputImageParams	pOutputImageParams,
	typename MaxPooling<T>::GeneralParams		pGeneralParams)
{
	typename MaxPooling<T>::InputImageParams&	ip = pInputImageParams;
	typename MaxPooling<T>::OutputImageParams&	op = pOutputImageParams;
	typename MaxPooling<T>::GeneralParams&		gp = pGeneralParams;

	uint32 idx				= ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32 threadsPerImage	= gp.threadsPerRow * gp.threadsPerCol;

	if(idx >= threadsPerImage * gp.imagesCount)
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

	// TODO bounds
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

	for(uint32 ch=0U; ch<gp.sampleWidth; ++ch){
		myOutputPixel[ch] = maximes[ch];
	}
}


template <typename T>
void MaxPooling<T>::sample(
	ImageBatch<T> const&	pInputImageBatch, 
	GpuBuffer&				pInputBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputBuffer)
{
	size_t rowThreads		= static_cast<size_t>(std::ceil(static_cast<double>(pInputImageBatch.getImageWidth()) / getWidth()));
	size_t colThreads		= static_cast<size_t>(std::ceil(static_cast<double>(pInputImageBatch.getImageHeight()) / getHeight()));
	size_t blocks			= utils::blocksCount(rowThreads * colThreads * pInputImageBatch.getImagesCount(), config::Cuda::THREADS_PER_BLOCK);

	InputImageParams ip;
	ip.buffer					= pInputBuffer.getDataPtr<T>();
	ip.imageWidth				= pInputImageBatch.getImageWidth();
	ip.imageHeight				= pInputImageBatch.getImageHeight();
	ip.alignedImageUnitSize		= pInputImageBatch.getAlignedImageByteSize() / sizeof(T);
	ip.alignedImageRowUnitSize	= pInputImageBatch.getAlignedImageRowByteSize() / sizeof(T);

	OutputImageParams op;
	op.buffer					= pOutputBuffer.getDataPtr<T>();
	op.alignedImageUnitSize		= pOutputImageBatch.getAlignedImageByteSize() / sizeof(T);
	op.alignedImageRowUnitSize	= pOutputImageBatch.getAlignedImageRowByteSize() / sizeof(T);

	GeneralParams gp;
	gp.imagesCount		= pInputImageBatch.getImagesCount();
	gp.sampleWidth		= getWidth();
	gp.sampleHeight		= getHeight();
	gp.threadsPerRow	= rowThreads;
	gp.threadsPerCol	= colThreads;
	gp.imageChannels	= pInputImageBatch.getImageChannelsCount();

	samplerMaxPooling<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(ip, op, gp);
}


template <typename T>
uint32 MaxPooling<T>::countOutputImageUnitSize(ImageBatch<T> const& pInputImageBatch) const {
	size_t rowThreads = static_cast<size_t>(std::ceil(static_cast<double>(pInputImageBatch.getImageWidth()) / getWidth()));
	size_t colThreads = static_cast<size_t>(std::ceil(static_cast<double>(pInputImageBatch.getImageHeight()) / getHeight()));
	return utils::align(
		rowThreads * 
		colThreads * 
		pInputImageBatch.getImageChannelsCount() *
		sizeof(T), 
		pInputImageBatch.getImageRowByteAligment() * 
		sizeof(T));
}


	}
}

#endif	/* CNN_MAX_POOLING_H_ */