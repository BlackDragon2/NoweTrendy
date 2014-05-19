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
	Network(uint32 batchSize, float learningRate, float stopError);
	~Network();
	void addLayer(uint32 neuronsNr, uint32 inputLength, activationFunction fun);
	void initWeights(float min, float max);
	void setClasses(std::string* classes, uint32 classesNr);
	
	template<typename T> 
	float train(T* input);
	
	template<typename T> 
	std::string classify(T* input);

private:
	uint32 batchSize;
	float learningRate;
	float stopError;
	std::vector<Layer*> layers;
	std::vector<std::string> classes;
	uint32 findMax(float* tab, uint32 neuronsNr);
};
	
	}}

template<typename T> 
float cnn::nn::Network::train(T* input)
{
	layers[0].calculateOutput(input);
	for(int i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1].getOutputBuffer();
		layers[i].calculateOutput(buffer.getDataPtr<float>());
	}
	
}

template<typename T>
std::string cnn::nn::Network::classify(T* input)
{
	layers[0].calculateOutput(input);
	for(int i=1;i<layers.size();i++)
	{
		cnn::gpu::GpuBuffer* buffer=layers[i-1].getOutputBuffer();
		layers[i].calculateOutput(buffer.getDataPtr<float>());
	}
	uint32 index=findMax(layers[layers.size()-1].getOutput(), layers[layers.size()-1].getNeuronsNr());
	return classes[index];
}

#endif	/* CNN_NETWORK_H_ */