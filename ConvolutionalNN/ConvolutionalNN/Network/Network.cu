#include "Network.cuh"

cnn::nn::Network::Network(uint32 batchSize, float learningRate, float stopError)
{
	this->batchSize=batchSize;
	this->learningRate=learningRate;
	this->stopError=stopError;
}

cnn::nn::Network::~Network()
{
	for(int i=0;i<layers.size();i++)
		delete layers[i];
	layers.clear();
}

void cnn::nn::Network::addLayer(uint32 neuronsNr, uint32 inputLength, activationFunction fun)
{
	layers.push_back(new Layer(neuronsNr, inputLength, fun));
}

void cnn::nn::Network::initWeights(float min, float max)
{
	for(int i=0;i<layers.size();i++)
		layers[i]->initWeights(min, max);
}

void cnn::nn::Network::setClasses(std::string* classes, uint32 classesNr)
{
	for(int i=0;i<classesNr;i++)
		this->classes.push_back(classes[i]);
}