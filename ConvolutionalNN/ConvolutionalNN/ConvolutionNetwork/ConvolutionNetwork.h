
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
		gpu::Converter<Input, Output>::PtrS const&			pConverter);
	virtual ~ConvolutionNetwork();

	virtual void run();


private:
	std::vector<ConvolutionLayer::PtrS> mLayers;
	gpu::Converter<Input, Output>		mConverter;
};


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::ConvolutionNetwork(
	std::vector<ConvolutionLayer<Input>::PtrS> const&	pLayers,
	gpu::Converter<Input, Output>::PtrS const&			pConverter)
:
	mLayers(pLayers),
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
	if (mConverter != nullptr){
		mConverter.convert(
			mLayers[mLayers.size() - 1]->getOutputBatch(),
			mLayers[mLayers.size() - 1]->getOutputBuffer(),
			nullptr);
	}
}


	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_ */