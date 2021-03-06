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
	void setClasses(uint32* classes, uint32 classesNr);
	void resetWeightsUpdates();
	void updateWeights();
	Layer* getLayer(uint32 index);
	
	template<typename T>
	void setWeightsUpdates(T* input);

	template<typename T> 
	float train(T* input, uint32 exampleClass);
	
	template<typename T> 
	uint32 classify(T* input);

	template<typename T> 
	cnn::gpu::GpuBuffer* calculateExample(T* input);

private:
	float learningRate;
	std::vector<Layer*> layers;
	std::vector<uint32> classes;
	uint32 findMax(float* tab, uint32 neuronsNr);
};
	
	}}

template<typename T>
void cnn::nn::Network::setWeightsUpdates(T* input)
{
	layers[0]->setWeightsUpdates(input, learningRate);
	for(int i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1]->getOutputBuffer();
		layers[i]->setWeightsUpdates(buffer->getDataPtr<float>(), learningRate);
	}
}

template<typename T> 
cnn::gpu::GpuBuffer* cnn::nn::Network::calculateExample(T* input)
{
	layers[0]->calculateOutput(input);
	for(uint32 i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1]->getOutputBuffer();
		layers[i]->calculateOutput(buffer->getDataPtr<float>());
	}
	return layers[layers.size()-1]->getOutputBuffer();
}

template<typename T> 
float cnn::nn::Network::train(T* input, uint32 exampleClass)
{
	calculateExample(input);
	float error=layers[layers.size()-1]->calculateError(exampleClass);
	for(int i=layers.size()-2;i>=0;i--)
	{
		layers[i]->calculateError(layers[i+1]->getWeightedErrorRates());
	}
	layers[0]->setWeightsUpdates(input, learningRate);
	for(uint32 i=1;i<layers.size();i++)
	{
		layers[i]->setWeightsUpdates(layers[i-1]->getOutputBuffer()->getDataPtr<float>(), learningRate);
	}
	return error;
}

template<typename T>
uint32 cnn::nn::Network::classify(T* input)
{
	uint32 index=findMax(calculateExample(input), layers[layers.size()-1]->getNeuronsNr());
	return classes[index];
}

#endif	/* CNN_NETWORK_H_ */