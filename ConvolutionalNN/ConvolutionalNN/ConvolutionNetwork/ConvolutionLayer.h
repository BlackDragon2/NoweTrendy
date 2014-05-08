
#ifndef CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_
#define CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_

#include "../GPU/Convolution.h"
#include "../GPU/Sampler.h"


namespace cnn {
	namespace cnetwork {


template <typename T>
class ConvolutionLayer {
public:
	ConvolutionLayer(
		gpu::Convolution<T>::PtrS const&	pConvolution,
		gpu::Sampler<T>::PtrS const&		pSampler);


private:
	gpu::Convolution<T>::PtrS	mConvolution;
	gpu::Sampler<T>::PtrS		mSampler;
};


template <typename T>
ConvolutionLayer<T>::ConvolutionLayer(
	gpu::Convolution<T>::PtrS const&	pConvolution,
	gpu::Sampler<T>::PtrS const&		pSampler)
:
	mConvolution(pConvolution),
	mSampler(pSampler)
{

}



	}
}


#endif	/* CNN_CONVOLUTION_NETWORK_CONVOLUTION_LAYER_H_ */