
#ifndef CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_
#define CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_

#include "../ImageBatch.h"
#include "../GPU/GpuBuffer.cuh"
#include "../GPU/Convolution.h"
#include "../GPU/Sampler.h"


namespace cnn {
	namespace cnetwork {


template <typename T>
class ConvolutionLayer {
public:
	typedef std::shared_ptr<ConvolutionLayer<T>> PtrS;


public:
	ConvolutionLayer(
		gpu::Convolution<T>::PtrS const&	pConvolution,
		gpu::Sampler<T>::PtrS const&		pSampler,
		ImageBatch<T>::PtrS const&			pKernelsBatch,
		gpu::GpuBuffer::PtrS const&			pKernelsBuffer,
		ImageBatch<T>::PtrS const&			pInputBatch,
		gpu::GpuBuffer::PtrS const&			pInputBuffer,
		ImageBatch<T>::PtrS const&			pMiddleBatch,
		gpu::GpuBuffer::PtrS const&			pMiddleBuffer,
		ImageBatch<T>::PtrS const&			pOutputBatch,
		gpu::GpuBuffer::PtrS const&			pOutputBuffer);


	void complex();
	void simple();

	void operator()(); 


	gpu::Convolution<T>::PtrS const&	getConvolution()	const;
	gpu::Sampler<T>::PtrS const&		getSampler()		const;

	ImageBatch<T>::PtrS const&	getKernelsBatch()	const;
	gpu::GpuBuffer::PtrS const&	getKernelsBuffer()	const;

	ImageBatch<T>::PtrS	const&	getInputBatch()		const;
	gpu::GpuBuffer::PtrS const&	getInputBuffer()	const;
	ImageBatch<T>::PtrS	const&	getMiddleBatch()	const;
	gpu::GpuBuffer::PtrS const&	getMiddleBuffer()	const;
	ImageBatch<T>::PtrS	const&	getOutputBatch()	const;
	gpu::GpuBuffer::PtrS const&	getOutputBuffer()	const;

	
private:
	gpu::Convolution<T>::PtrS	mConvolution;
	gpu::Sampler<T>::PtrS		mSampler;
	
	ImageBatch<T>::PtrS		mKernelsBatch;
	gpu::GpuBuffer::PtrS	mKernelsBuffer;

	ImageBatch<T>::PtrS		mInputBatch;
	gpu::GpuBuffer::PtrS	mInputBuffer;
	ImageBatch<T>::PtrS		mMiddleBatch;
	gpu::GpuBuffer::PtrS	mMiddleBuffer;
	ImageBatch<T>::PtrS		mOutputBatch;
	gpu::GpuBuffer::PtrS	mOutputBuffer;
};


template <typename T>
ConvolutionLayer<T>::ConvolutionLayer(
	gpu::Convolution<T>::PtrS const&	pConvolution,
	gpu::Sampler<T>::PtrS const&		pSampler,
	ImageBatch<T>::PtrS const&			pKernelsBatch,
	gpu::GpuBuffer::PtrS const&			pKernelsBuffer,
	ImageBatch<T>::PtrS const&			pInputBatch,
	gpu::GpuBuffer::PtrS const&			pInputBuffer,
	ImageBatch<T>::PtrS const&			pMiddleBatch,
	gpu::GpuBuffer::PtrS const&			pMiddleBuffer,
	ImageBatch<T>::PtrS const&			pOutputBatch,
	gpu::GpuBuffer::PtrS const&			pOutputBuffer)
:
	mConvolution(pConvolution),
	mSampler(pSampler),
	mKernelsBatch(pKernelsBatch),
	mKernelsBuffer(pKernelsBuffer),
	mInputBatch(pInputBatch),
	mInputBuffer(pInputBuffer),
	mMiddleBatch(pMiddleBatch),
	mMiddleBuffer(pMiddleBuffer),
	mOutputBatch(pOutputBatch),
	mOutputBuffer(pOutputBuffer)
{

}


template <typename T>
void ConvolutionLayer<T>::complex(){
	mConvolution->compute(
		*mInputBatch,
		*mInputBuffer,
		*mKernelsBatch,
		*mKernelsBuffer,
		*mMiddleBatch,
		*mMiddleBuffer);
}


template <typename T>
void ConvolutionLayer<T>::simple(){
	mSampler->sample(
		*mMiddleBatch,
		*mMiddleBuffer,
		*mOutputBatch,
		*mOutputBuffer);
}


template <typename T>
void ConvolutionLayer<T>::operator()(){
	complex();
	simple();
} 


template <typename T>
gpu::Convolution<T>::PtrS const& ConvolutionLayer<T>::getConvolution() const {
	return mConvolution;
}


template <typename T>
gpu::Sampler<T>::PtrS const& ConvolutionLayer<T>::getSampler() const {
	return mSampler;
}


template <typename T>
ImageBatch<T>::PtrS const& ConvolutionLayer<T>::getKernelsBatch() const {
	return mKernelsBatch;
}


template <typename T>
gpu::GpuBuffer::PtrS const& ConvolutionLayer<T>::getKernelsBuffer() const {
	return mKernelsBuffer;
}


template <typename T>
ImageBatch<T>::PtrS	const& ConvolutionLayer<T>::getInputBatch() const {
	return mInputBatch;
}


template <typename T>
gpu::GpuBuffer::PtrS const& ConvolutionLayer<T>::getInputBuffer() const {
	return mInputBuffer;
}


template <typename T>
ImageBatch<T>::PtrS	const& ConvolutionLayer<T>::getMiddleBatch() const {
	return mMiddleBatch;
}


template <typename T>
gpu::GpuBuffer::PtrS const& ConvolutionLayer<T>::getMiddleBuffer() const {
	return mMiddleBuffer;
}


template <typename T>
ImageBatch<T>::PtrS	const& ConvolutionLayer<T>::getOutputBatch() const {
	return mOutputBatch;
}


template <typename T>
gpu::GpuBuffer::PtrS const& ConvolutionLayer<T>::getOutputBuffer() const {
	return mOutputBuffer;
}


	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_ */