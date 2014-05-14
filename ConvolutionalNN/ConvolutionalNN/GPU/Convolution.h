
#ifndef CNN_CONVOLUTION_H_
#define CNN_CONVOLUTION_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Convolution {
public:
	typedef std::shared_ptr<Convolution<T>> PtrS;


public:
	Convolution(
		uint32 pOffsetX,
		uint32 pOffsetY);
	virtual ~Convolution(); 


	virtual void compute(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer) = 0;

	virtual void operator()(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer);


	uint32 getOffsetX()	const;
	uint32 getOffsetY()	const;

	void setOffsetX(uint32 pOffsetX);
	void setOffsetY(uint32 pOffsetY);

	virtual uint32 countOutputImageUnitSize(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const = 0;
	virtual uint32 countOutputImageByteSize(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const;

	virtual uint32 countOutputUnitSize(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const;
	virtual uint32 countOutputByteSize(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const;


private:
	uint32 mOffsetX;
	uint32 mOffsetY;
};


template <typename T>
Convolution<T>::Convolution(
	uint32 pOffsetX,
	uint32 pOffsetY)
:
	mOffsetX(pOffsetX),
	mOffsetY(pOffsetY)
{

}


template <typename T>
Convolution<T>::~Convolution(){

}


template <typename T>
void Convolution<T>::operator()(
	ImageBatch<T> const&	pInputImageBatch,
	GpuBuffer&				pInputImageBuffer,
	ImageBatch<T> const&	pKernelsImageBatch,
	GpuBuffer&				pKernelsImageBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputImageBuffer)
{
	compute(
		pInputImageBatch, pInputImageBuffer, 
		pKernelsImageBatch, pKernelsImageBuffer, 
		pOutputImageBatch, pOutputImageBuffer);
}


template <typename T>
uint32 Convolution<T>::getOffsetX()	const {
	return mOffsetX;
}


template <typename T>
uint32 Convolution<T>::getOffsetY()	const {
	return mOffsetY;
}


template <typename T>
void Convolution<T>::setOffsetX(uint32 pOffsetX){
	mOffsetX = pOffsetX;
}


template <typename T>
void Convolution<T>::setOffsetY(uint32 pOffsetY){
	mOffsetY = pOffsetY;
}

 
template <typename T>
uint32 Convolution<T>::countOutputImageByteSize(
	ImageBatch<T> const& pInputBatch,
	ImageBatch<T> const& pKernelsBatch) const 
{
	return countOutputImageUnitSize(pInputBatch, pKernelsBatch) * sizeof(T);
}


template <typename T>
uint32 Convolution<T>::countOutputUnitSize(
	ImageBatch<T> const& pInputBatch,
	ImageBatch<T> const& pKernelsBatch) const 
{
	return countOutputImageUnitSize(pInputBatch, pKernelsBatch) * 
		pInputBatch.getImagesCount();
}


template <typename T>
uint32 Convolution<T>::countOutputByteSize(
	ImageBatch<T> const& pInputBatch,
	ImageBatch<T> const& pKernelsBatch) const 
{
	return countOutputImageByteSize(pInputBatch, pKernelsBatch) * 
		pInputBatch.getImagesCount();
}


	}
}

#endif	/* CNN_CONVOLUTION_H_ */