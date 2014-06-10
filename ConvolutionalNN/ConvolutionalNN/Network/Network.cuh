#ifndef CNN_NETWORK_H_
#define CNN_NETWORK_H_

#include "../GPU/GpuBuffer.cuh"
#include <stdlib.h>
#include <vector>
#include "Layer.cuh"
#include "NetworkCuda.cuh"
#include "../Config.h"
#include "../Types.h"

namespace cnn{
	namespace nn{
	
class Network
{
public:
	Network(float learningRate);
	~Network();
	void addLayer(uint32 neuronsNr, uint32 inputLength, activationFunction fun);
	void initWeights(float min, float max);
	void setClasses(std::string* classes, uint32 classesNr);
	void resetWeightsUpdates();
	void updateWeights();
	Layer* getLayer(uint32 index);
	
	template<typename T>
	void setWeightsUpdates(T* input);

	template<typename T> 
	float* train(T* input, uint32 exampleClass);
	
	template<typename T> 
	uint32 classify(T* input);

	template<typename T> 
	cnn::gpu::GpuBuffer* calculateExample(T* input);

private:
	float learningRate;
	std::vector<Layer*> layers;
	std::vector<std::string> classes;
	uint32 findMax(float* tab, uint32 neuronsNr);
};
	
	}}

template<typename T>
cnn::nn::Network::setWeightsUpdates(T* input)
{
	layers[0]->setWeightsUpdates(input);
	for(int i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1]->getOutputBuffer();
		layers[i]->setWeightsUpdates(buffer->getDataPtr<float>());
	}
}

template<typename T> 
cnn::gpu::GpuBuffer* cnn::nn::Network::calculateExample(T* input)
{
	layers[0]->calculateOutput(input);
	for(int i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1]->getOutputBuffer();
		layers[i]->calculateOutput(buffer->getDataPtr<float>());
	}
	return layers[layers.size()-1]->getOutputBuffer();
}

template<typename T> 
float* cnn::nn::Network::train(T* input, int exampleClass)
{
	calculateExample(input);
	layers[layers.size()-1]->calculateError(exampleClass);
	for(int i=layers.size()-2;i>=0;i--)
	{
		layers[i]->calculateError(layers[i+1]->getWeightedErrorRates());
	}
	for(int i=0;i<layers.size();i++)
	{

	}

}

template<typename T>
std::string cnn::nn::Network::classify(T* input)
{
	uint32 index=findMax(calculateExample(input), layers[layers.size()-1]->getNeuronsNr());
	return classes[index];
}

#endif	/* CNN_NETWORK_H_ */