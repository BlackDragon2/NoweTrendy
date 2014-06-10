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
	float error;
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
	while(stopError<error)
	{
		error=0;
		nnetwork->resetWeightsUpdates();
		for(int i=0;i<pImages->getImagesCount();i++)
		{
			cnetwork->run();
			nnetwork->train(cnetwork->getOutputBuffer()->getDataPtr<float>, classes[i]);
			cnn::gpu::GpuBuffer* buffer=nnetwork->getLayer[0]->getWeightedErrorRates();

			nnetwork->setWeightsUpdates(cnetwork->getOutputBuffer()->getDataPtr<float>);
		}
		nnetwork->updateWeights();
		std::cout<<error<<std::endl;
	}
}

template<typename Input, typename Output>
void cnn::JoinedNetwork<Input, Output>::processExample()
{
}
#endif	/* CNN_JOINED_NETWORK_H_ */