#include "JoinedNetwork.cuh"

cnn::JoinedNetwork::JoinedNetwork(mycnn::ConvNet* cnetwork, cnn::nn::Network* nnetwork, mycnn::Images im, float stopError, float learningRate)
{
	this->cnetwork=cnetwork;
	this->nnetwork=nnetwork;
	this->images=im;
	this->stopError=stopError;
	this->learningRate=learningRate;
}


cnn::JoinedNetwork::~JoinedNetwork()
{
}

void cnn::JoinedNetwork::teach()
{
	float error=2*stopError;
	while(stopError<error)
	{
		error=0;
		cnetwork->resetWeightsUpdates();
		nnetwork->resetWeightsUpdates();
		for(uint32 i=0;i<images.getImageCount();i++)
		{
			cnetwork->run(images.getImageBuffer(), images.getImageSize()*i);
			error+=nnetwork->train<float>(cnetwork->getLastLayerOutput<float>(i), images.getClass(i));
			cnn::gpu::GpuBuffer* buffer=nnetwork->getLayer(0)->getWeightedErrorRates();//do propagacji
			cnetwork->train(buffer, images.getImageBuffer(), images.getImageSize()*i);
		}
		error/=images.getImageCount();
		nnetwork->updateWeights();
		cnetwork->updateWeights();
		std::cout<<error<<std::endl;
	}
}

