
#ifndef CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_
#define CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_

#include "ConvolutionLayer.h"
#include "../GPU/Packer.cuh"


namespace cnn {
	namespace cnetwork {


template <typename Input, typename Output>
class ConvolutionNetwork {
public:
	ConvolutionNetwork(std::vector<ConvolutionLayer<Input>::PtrS> const& pLayers);
	virtual ~ConvolutionNetwork();


private:
	std::vector<ConvolutionLayer::PtrS> mLayers;
	gpu::Packer<Input, Output>			mPacker;
};


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::ConvolutionNetwork(
	std::vector<ConvolutionLayer<Input>::PtrS> const& pLayers)
:
	mLayers(pLayers)
{

}


template<typename Input, typename Output>
ConvolutionNetwork<Input, Output>::~ConvolutionNetwork(){

}


	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_NETWORK_H_ */