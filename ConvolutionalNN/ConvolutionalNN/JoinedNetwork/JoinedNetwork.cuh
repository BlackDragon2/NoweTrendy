#ifndef CNN_JOINED_NETWORK_H_
#define CNN_JOINED_NETWORK_H_

#include "..\ConvolutionNetwork\ConvolutionNetwork.h"
#include "..\Network\Network.cuh"
#include "..\ImageBatch.h"
#include "..\ConvolutionNetwork\ConvolutionNetwork.h"

namespace cnn{

template<typename Input, typename Output>
class JoinedNetwork
{
public:
	JoinedNetwork(cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork, cnn::nn::Network* nnetwork, std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, float stopError, float learningRate);
	~JoinedNetwork();
	uint32 classify();
	void teach(uint32* classes);

private:
	float stopError;
	float learningRate;
	void processExample();
	cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork;
	cnn::nn::Network* nnetwork;
	std::shared_ptr<cnn::ImageBatch<uchar>> pImages;
};
	}

template<typename Input, typename Output>
cnn::JoinedNetwork<Input, Output>::JoinedNetwork(cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork, cnn::nn::Network* nnetwork, std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, float stopError, float learningRate)
{
	this->cnetwork=cnetwork;
	this->nnetwork=nnetwork;
	this->pImages=pImages;
	this->stopError=stopError;
	this->learningRate=learningRate;
}

template<typename Input, typename Output>
cnn::JoinedNetwork<Input, Output>::~JoinedNetwork()
{
}

template<typename Input, typename Output>
void cnn::JoinedNetwork<Input, Output>::teach(uint32* classes)
{
	float error=2*stopError;
	while(stopError<error)
	{
		error=0;
		nnetwork->resetWeightsUpdates();
		for(uint32 i=0;i<pImages->getImagesCount();i++)
		{
			cnetwork->run();
			//cnn::gpu::GpuBuffer::PtrS bf=cnetwork->getOutputBuffer();
			//float* x=new float[10];
			error+=nnetwork->train<float>(cnetwork->getOutputBuffer()->getDataPtr<float>(), classes[i]);
			//error+=nnetwork->train<float>(x, classes[i]);
			cnn::gpu::GpuBuffer* buffer=nnetwork->getLayer(0)->getWeightedErrorRates();//do propagacji
		}
		error/=pImages->getImagesCount();
		nnetwork->updateWeights();
		std::cout<<error<<std::endl;
	}
}

template<typename Input, typename Output>
void cnn::JoinedNetwork<Input, Output>::processExample()
{
}
#endif	/* CNN_JOINED_NETWORK_H_ */