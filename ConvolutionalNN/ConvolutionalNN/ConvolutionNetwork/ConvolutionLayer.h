
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
		mInputBatch,
		mInputBuffer,
		mKernelsBatch,
		mKernelsBuffer,
		mOutputBatch,
		mOutputBuffer);
}


template <typename T>
void ConvolutionLayer<T>::simple(){

}


template <typename T>
void ConvolutionLayer<T>::operator()(){
	//mConvolution->compute()
	//mSampler->sample()
}



	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_ */