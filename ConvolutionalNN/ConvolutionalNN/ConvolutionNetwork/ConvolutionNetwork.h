
#ifndef CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_
#define CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_

#include "ConvolutionLayer.h"
#include "../GPU/Packer.cuh"


namespace cnn {
	namespace cnetwork {


template <typename Input, typename Output>
class ConvolutionNetwork {
public:
	ConvolutionNetwork();
	ConvolutionNetwork(
		std::vector<ConvolutionLayer<Input>::PtrS> const&	pLayers,
		gpu::GpuBuffer::PtrS const&							pOutputBuffer	= nullptr,
		gpu::Converter<Input, Output>::PtrS const&			pConverter		= gpu::Converter<Input, Output>::PtrS(new gpu::Packer<Input, Output>()));
	virtual ~ConvolutionNetwork();

	void initialize(uint32 pLayers);
	void buildOutputBuffer();
	void run();

	void setOutputBuffer(gpu::GpuBuffer::PtrS const& pBuffer);

	gpu::GpuBuffer::PtrS const& getOutputBuffer() const;
	
	void addLayer(ConvolutionLayer<Input>::PtrS const& pLayer);
	void addLayer(
		gpu::Convolution<Input>::PtrS const&	pConvolution,
		gpu::Sampler<Input>::PtrS const&		pSampler,
		ImageBatch<Input>::PtrS const&			pKernelsBatch,
		ImageBatch<Input>::PtrS const&			pInputBatch = nullptr);

	ConvolutionLayer<Input>::PtrS const& getLayer(uint32 pIndex)	const;
	ConvolutionLayer<Input>::PtrS const& getFirstLayer()			const;
	ConvolutionLayer<Input>::PtrS const& getLastLayer()				const;


private:
	std::vector<ConvolutionLayer<Input>::PtrS>	mLayers;
	gpu::Converter<Input, Output>::PtrS			mConverter;
	gpu::GpuBuffer::PtrS						mOutputBuffer;
};


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::ConvolutionNetwork()
:
	mLayers(0),
	mOutputBuffer(nullptr),
	mConverter(nullptr)
{

}


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::ConvolutionNetwork(
	std::vector<ConvolutionLayer<Input>::PtrS> const&	pLayers,
	gpu::GpuBuffer::PtrS const&							pOutputBuffer,
	gpu::Converter<Input, Output>::PtrS const&			pConverter)
:
	mLayers(pLayers),
	mOutputBuffer(pOutputBuffer),
	mConverter(pConverter)
{

}


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::~ConvolutionNetwork(){

}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::setOutputBuffer(gpu::GpuBuffer::PtrS const& pBuffer){
	mOutputBuffer = pBuffer;
}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::initialize(uint32 pLayers){
	assert(pLayers > 0 && false && "not implemented yet"); 
}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::buildOutputBuffer(){
	float ratio			= static_cast<float>(sizeof(Output)) / sizeof(Input);
	auto const& layer	= getLastLayer();
	uint32 bytes		= layer->getOutputBatch()->getImageByteSize() * layer->getOutputBatch()->getImagesCount() * sizeof(Output);
	uint32 aligment		= static_cast<size_t>(static_cast<float>(layer->getOutputBatch()->getImageRowByteAligment()) * ratio);
	mOutputBuffer.reset(new gpu::GpuBuffer(bytes, aligment));
}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::run(){
	for (uint32 i=0UL; i<mLayers.size(); ++i){
		(*mLayers[i])();
	}
	if (mConverter != nullptr){
		assert(mOutputBuffer != nullptr);
		mConverter->convert(
			*getLastLayer()->getOutputBatch(),
			*getLastLayer()->getOutputBuffer(),
			*mOutputBuffer);
	}
}


template<typename Input, typename Output>
gpu::GpuBuffer::PtrS const& ConvolutionNetwork<Input, Output>::getOutputBuffer() const {
	return mOutputBuffer == nullptr ? 
		mLayers[mLayers.size() - 1]->getOutputBuffer() :
		mOutputBuffer;
}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::addLayer(ConvolutionLayer<Input>::PtrS const& pLayer){
	mLayers.push_back(pLayer);
}


template<typename Input, typename Output>
void ConvolutionNetwork<Input, Output>::addLayer(
	gpu::Convolution<Input>::PtrS const&	pConvolution,
	gpu::Sampler<Input>::PtrS const&		pSampler, 
	ImageBatch<Input>::PtrS const&			pKernelsBatch,
	ImageBatch<Input>::PtrS const&			pInputBatch)
{
	assert(pInputBatch != nullptr || mLayers.size() > 0);

	typename ImageBatch<Input>::PtrS	inputBatch;
	gpu::GpuBuffer::PtrS				inputBuffer;

	if(pInputBatch == nullptr){ 
		inputBatch	= getLastLayer()->getOutputBatch();
		inputBuffer	= getLastLayer()->getOutputBuffer();
	} else {
		inputBatch	= pInputBatch;
		inputBuffer	= gpu::GpuBuffer::PtrS(new gpu::GpuBuffer(
			inputBatch->getBatchByteSize(),
			inputBatch->getImageRowByteAligment(),
			inputBatch->getBatchDataPtr(),
			0U));
	}

	// batches
	typename ImageBatch<Input>::PtrS middleBatch(
		new ImageBatch<Input>(
			pConvolution->convolvedImageSizeX(*inputBatch, *pKernelsBatch),
			pConvolution->convolvedImageSizeY(*inputBatch, *pKernelsBatch),
			inputBatch->getImageChannelsCount(),
			inputBatch->getImageRowByteAligment()));
	middleBatch->allocateSpaceForImages(
		pKernelsBatch->getImagesCount() * inputBatch->getImagesCount(), true);

	typename ImageBatch<Input>::PtrS outputBatch(
		new ImageBatch<Input>(
			pSampler->sampledImageSizeX(*middleBatch),
			pSampler->sampledImageSizeY(*middleBatch),
			inputBatch->getImageChannelsCount(),
			inputBatch->getImageRowByteAligment()));
	outputBatch->allocateSpaceForImages(middleBatch->getImagesCount(), true);

	// buffers
	gpu::GpuBuffer::PtrS kernels(new gpu::GpuBuffer(
		pKernelsBatch->getBatchByteSize(),
		inputBatch->getImageRowByteAligment(),
		pKernelsBatch->getBatchDataPtr(),
		0U));

	gpu::GpuBuffer::PtrS middle(new gpu::GpuBuffer(
		middleBatch->getBatchByteSize(),
		inputBatch->getImageRowByteAligment()));

	gpu::GpuBuffer::PtrS output(new gpu::GpuBuffer(
		outputBatch->getBatchByteSize(),
		inputBatch->getImageRowByteAligment()));

	typename ConvolutionLayer<Input>::PtrS layer(new ConvolutionLayer<Input>(
		pConvolution,
		pSampler,
		pKernelsBatch,
		kernels,
		inputBatch,
		inputBuffer,
		middleBatch,
		middle,
		outputBatch,
		output));

	addLayer(layer);
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getLayer(uint32 pIndex) const {
	assert(pIndex < mLayers.size());
	return mLayers[pIndex];
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getFirstLayer() const {
	assert(mLayers.size() > 0);
	return getLayer(mLayers[0]);
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getLastLayer() const {
	assert(mLayers.size() > 0);
	return *mLayers.rbegin();
}


	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_ */