#include "Network.cuh"

cnn::nn::Network::Network(float learningRate)
{
	this->learningRate=learningRate;
}

cnn::nn::Network::~Network()
{
	for(uint32 i=0;i<layers.size();i++)
		delete layers[i];
	layers.clear();
}

cnn::nn::Layer* cnn::nn::Network::getLayer(uint32 index)
{
	return layers[index];
}

void cnn::nn::Network::addLayer(uint32 neuronsNr, uint32 inputLength, activationFunction fun)
{
	layers.push_back(new Layer(neuronsNr, inputLength, fun));
}

void cnn::nn::Network::initWeights(float min, float max)
{
	for(uint32 i=0;i<layers.size();i++)
		layers[i]->initWeights(min, max);
}

void cnn::nn::Network::setClasses(std::string* classes, uint32 classesNr)
{
	for(uint32 i=0;i<classesNr;i++)
		this->classes.push_back(classes[i]);
}

uint32 cnn::nn::Network::findMax(float* tab, uint32 neuronsNr)
{
	uint32 indMax=0;
	float valMax=tab[0];
	for(int i=1;i<neuronsNr;i++)
		if(valMax<tab[i])
		{
			valMax=tab[i];
			indMax=i;
		}
	return indMax;
}

void cnn::nn::Network::resetWeightsUpdates()
{
	for(uint32 i=0;i<layers.size();i++)
		layers[i]->resetWeightsUpdates<float>();
}

void cnn::nn::Network::updateWeights()
{
	for(uint32 i=0;i<layers.size();i++)
		layers[i]->updateWeights();
}