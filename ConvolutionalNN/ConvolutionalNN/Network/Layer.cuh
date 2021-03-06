#ifndef CNN_LAYER_H_
#define CNN_LAYER_H_

#include "../GPU/GpuBuffer.cuh"
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include "LayerCuda.cuh"
#include "../Config.h"

namespace cnn{
	namespace nn{

enum activationFunction{
	SIGMOIDAL,
	TANH
};

class Layer
{
public:
	Layer(uint32 neuronsNr, uint32 inputLength, activationFunction fun);
	~Layer(void);
	void initWeights(float min, float max);

	template<typename resetType>
	void resetWeightsUpdates();

	template<typename T> 
	void calculateOutput(T* input);
	
	float* getOutput();
	cnn::gpu::GpuBuffer* getWeightedErrorRates();
	cnn::gpu::GpuBuffer* getOutputBuffer();
	uint32 getNeuronsNr();
	float calculateError(uint32 exampleClass);
	void calculateError(cnn::gpu::GpuBuffer* errorRates);
	
	template<typename T>
	void setWeightsUpdates(T* input, float learningRate);
	
	void updateWeights();

private:
	uint32 neuronsNr;
	uint32 inputLength;
	uint32 weightsLength;
	activationFunction activationFun;
	float* weights;
	float* output;
	float* biases;
	cnn::gpu::GpuBuffer weightsDev, outputDev, biasesDev, weightsUpdateDev, errorRatesDev;
	//weightsDev - najpierw biasy potem wagi kolejnych neuronow
};
	}}

template<typename T> 
void cnn::nn::Layer::calculateOutput(T* input)
{
		weightsDev.writeToDevice(weights, weightsLength*sizeof(float));

		int blocks;
		if((weightsLength-neuronsNr)%config::Cuda::THREADS_PER_BLOCK==0)//bez biasow
			blocks=(weightsLength-neuronsNr)/config::Cuda::THREADS_PER_BLOCK;
		else
			blocks=(weightsLength-neuronsNr)/config::Cuda::THREADS_PER_BLOCK+1;
		cnn::cuda::calculatePotential<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(input, weightsDev.getDataPtr<float>()+neuronsNr, outputDev.getDataPtr<float>(), inputLength, neuronsNr);
		switch (activationFun)
		{
			SIGMOIDAL:
				cnn::cuda::calculateSigmoidalOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr, weightsDev.getDataPtr<float>(), biasesDev.getDataPtr<float>());
				break;
			TANH:
				cnn::cuda::calculateTanhOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr, weightsDev.getDataPtr<float>(), biasesDev.getDataPtr<float>());
				break;
			default:
				cnn::cuda::calculateSigmoidalOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr, weightsDev.getDataPtr<float>(), biasesDev.getDataPtr<float>());
				break;
		}

		outputDev.loadFromDevice(output, neuronsNr*sizeof(float));
}

template<typename resetType>
void cnn::nn::Layer::resetWeightsUpdates()
{
	int blocks;
	if(weightsLength%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::reset<resetType><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(weightsUpdateDev.getDataPtr<resetType>(), weightsLength);
}

template<typename T>
void cnn::nn::Layer::setWeightsUpdates(T* input, float learningRate)
{
	int blocks;
	if(weightsLength%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::setWeightsUpdates<float><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(input, weightsLength, neuronsNr, biasesDev.getDataPtr<float>(), inputLength, weightsUpdateDev.getDataPtr<float>(), errorRatesDev.getDataPtr<float>(), learningRate);
}
#endif	/* CNN_LAYER_H_ */

