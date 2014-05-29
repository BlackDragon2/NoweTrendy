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
	JoinedNetwork(cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork, cnn::nn::Network* nnetwork, std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, std::string* classes, float stopError, float learningRate);
	~JoinedNetwork();
	std::string classify();
	float teach();

private:
	float error;
	float stopError;
	float learningRate;
	void processExample();
	cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork;
	cnn::nn::Network* nnetwork;
	std::shared_ptr<cnn::ImageBatch<uchar>> pImages;
	std::string* classes;
};
	}

template<typename Input, typename Output>
cnn::JoinedNetwork<Input, Output>::JoinedNetwork(cnn::cnetwork::ConvolutionNetwork<Input, Output>* cnetwork, cnn::nn::Network* nnetwork, std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, std::string* classes, float stopError, float learningRate)
{
	this->cnetwork=cnetwork;
	this->nnetwork=nnetwork;
	this->pImages=pImages;
	this->classes=classes;
	this->stopError=stopError;
	this->learningRate=learningRate;
}

template<typename Input, typename Output>
cnn::JoinedNetwork<Input, Output>::teach(std::string* classes)
{
	while(stopError<error)
	{
		error=0;
		nnetwork->resetWeightsUpdates();
		for(int i=0;i<pImages->getImagesCount();i++)
		{
			cnetwork->run();
			nnetwork->train();

		}
}

template<typename Input, typename Output>
cnn::JoinedNetwork<Input, Output>::processExample()
{
}
#endif	/* CNN_JOINED_NETWORK_H_ */