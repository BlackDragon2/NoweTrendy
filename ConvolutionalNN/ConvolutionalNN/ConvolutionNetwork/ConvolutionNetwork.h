
#ifndef CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_
#define CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_

#include "ConvolutionLayer.h"
#include "../GPU/Packer.cuh"


namespace cnn {
	namespace cnetwork {


template <typename Input, typename Output>
class ConvolutionNetwork {
public:
	ConvolutionNetwork(
		std::vector<ConvolutionLayer<Input>::PtrS> const&	pLayers,
		gpu::GpuBuffer::PtrS const&							pOutputBuffer,
		gpu::Converter<Input, Output>::PtrS const&			pConverter		= gpu::Converter<Input, Output>::PtrS(new gpu::Packer<Input, Output>()));
	virtual ~ConvolutionNetwork();

	virtual void run();

	gpu::GpuBuffer::PtrS const& getOutputBuffer() const;
	
	ConvolutionLayer<Input>::PtrS const& getLayer(uint32 pIndex)	const;
	ConvolutionLayer<Input>::PtrS const& getFirstLayer()			const;
	ConvolutionLayer<Input>::PtrS const& getLastLayer()				const;


private:
	std::vector<ConvolutionLayer::PtrS> mLayers;
	gpu::Converter<Input, Output>		mConverter;
	gpu::GpuBuffer::PtrS				mOutputBuffer;
};


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
void ConvolutionNetwork<Input, Output>::run(){
	for (ConvolutionLayer<Input> const& layer : mLayers){
		(*layer)();
	}
	if (mConverter != nullptr && mOutputBuffer != nullptr){
		mConverter.convert(
			getLastLayer()->getOutputBatch(),
			getLastLayer()->getOutputBuffer(),
			mOutputBuffer);
	}
}


template<typename Input, typename Output>
gpu::GpuBuffer::PtrS const& ConvolutionNetwork<Input, Output>::getOutputBuffer() const {
	return mOutputBuffer == nullptr ? 
		mLayers[mLayers.size() - 1]->getOutputBuffer() :
		mOutputBuffer;
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getLayer(uint32 pIndex) const {
	return mLayers[pIndex];
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getFirstLayer() const {
	return getLayer(mLayers[0]);
}


template<typename Input, typename Output>
ConvolutionLayer<Input>::PtrS const& ConvolutionNetwork<Input, Output>::getLastLayer() const {
	return getLayer((*mLayers.rbegin()));
}


	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_ */