
#ifndef CNN_SAMPLER_H_
#define CNN_SAMPLER_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Sampler {
public:
	typedef std::shared_ptr<Sampler<T>> PtrS;


public:
	Sampler(uint32 pWidth, uint32 pHeight);
	virtual ~Sampler();
	
	virtual void sample(
		ImageBatch<T> const&	pInputImageBatch,  
		GpuBuffer&				pInputBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputBuffer) = 0;

	void operator()(
		ImageBatch<T> const&	pInputImageBatch,  
		GpuBuffer&				pInputBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputBuffer);


	uint32 getWidth()	const;
	uint32 getHeight()	const;

	void setWidth(uint32 pWidth);
	void setHeight(uint32 pHeight);

	virtual uint32 countOutputImageUnitSize(ImageBatch<T> const& pInputImageBatch) const;
	virtual uint32 countOutputImageByteSize(ImageBatch<T> const& pInputImageBatch) const;
	
	virtual uint32 countOutputUnitSize(ImageBatch<T> const& pInputImageBatch) const;
	virtual uint32 countOutputByteSize(ImageBatch<T> const& pInputImageBatch) const;

	virtual uint32 sampledImageSizeX(ImageBatch<T> const& pInputBatch) const = 0;
	virtual uint32 sampledImageSizeY(ImageBatch<T> const& pInputBatch) const = 0;

private:
	uint32 mWidth;
	uint32 mHeight;
};


template <typename T>
Sampler<T>::Sampler(
	uint32 pWidth,
	uint32 pHeight)
:
	mWidth(pWidth),
	mHeight(pHeight)
{

}


template <typename T>
Sampler<T>::~Sampler(){

}


template <typename T>
void Sampler<T>::operator()(
	ImageBatch<T> const&	pInputImageBatch, 
	GpuBuffer&				pInputBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputBuffer)
{
	sample(pInputImageBatch, pInputBuffer, pOutputImageBatch, pOutputBuffer);
}


template <typename T>
uint32 Sampler<T>::getWidth() const {
	return mWidth;
}


template <typename T>
uint32 Sampler<T>::getHeight() const {
	return mHeight;
}


template <typename T>
void Sampler<T>::setWidth(uint32 pWidth){
	mWidth = pWidth;
}


template <typename T>
void Sampler<T>::setHeight(uint32 pHeight){
	mHeight = pHeight;
}


template <typename T>
uint32 Sampler<T>::countOutputImageUnitSize(ImageBatch<T> const& pInputImageBatch) const {
	return 
		utils::align(
			sampledImageSizeX(pInputImageBatch) * 
			pInputImageBatch.getImageChannelsCount() * sizeof(T), 
			pInputImageBatch.getImageRowByteAligment() * sizeof(T)
		) * sampledImageSizeY(pInputImageBatch);
}


template <typename T>
uint32 Sampler<T>::countOutputImageByteSize(ImageBatch<T> const& pInputImageBatch) const {
	return countOutputImageUnitSize(pInputImageBatch) * sizeof(T);
}


template <typename T>
uint32 Sampler<T>::countOutputUnitSize(ImageBatch<T> const& pInputImageBatch) const {
	return countOutputImageUnitSize(pInputImageBatch) * pInputImageBatch.getImagesCount();
}


template <typename T>
uint32 Sampler<T>::countOutputByteSize(ImageBatch<T> const& pInputImageBatch) const {
	return countOutputImageByteSize(pInputImageBatch) * pInputImageBatch.getImagesCount();
}


	}
}

#endif	/* CNN_SAMPLER_H_ */