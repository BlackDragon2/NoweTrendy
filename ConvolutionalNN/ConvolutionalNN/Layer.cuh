#pragma once

#include "GPU\GpuBuffer.cuh"
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include "LayerCuda.cuh"
#include "Config.h"

namespace cnn{
	namespace nn{

enum activationFunction{
	SIGMOIDAL,
	TANH,
	MAX
};

class Layer
{
public:
	Layer(uint32 neuronsNr, uint32 inputLength, activationFunction fun);
	~Layer(void);
	void initWeights(double min, double max);

	template<typename T> void calculateOutput(T* input)
	{
		cnn::gpu::GpuBuffer inputDev, weightsDev, outputDev;
		inputDev.allocate(inputLength*sizeof(T));
		weightsDev.allocate(weightsLength*sizeof(float));
		outputDev.allocate(neuronsNr*sizeof(float));
		
		inputDev.writeToDevice(input, inputLength*sizeof(T));
		weightsDev.writeToDevice(weights, weightsLength*sizeof(float));

		int blocks;
		if(weightsLength%config::Cuda::THREADS_PER_BLOCK==0)
			blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK;
		else
			blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK+1;
		cnn::cuda::calculatePotential<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(inputDev.getDataPtr<T>(), weightsDev.getDataPtr<float>(), outputDev.getDataPtr<float>(), inputLength, neuronsNr);
		switch (activationFun)
		{
			SIGMOIDAL:
				cnn::cuda::calculateSigmoidalOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr);
				break;
			TANH:
				cnn::cuda::calculateTanhOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr);
				break;
			MAX:
				cnn::cuda::calculateMaxOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr);
				break;
			default:
				cnn::cuda::calculateSigmoidalOutput<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr);
				break;
		}

		outputDev.loadFromDevice(output, neuronsNr*sizeof(float));
		outputDev.free();
		weightsDev.free();
		inputDev.free();
	};

private:
	uint32 neuronsNr;
	uint32 inputLength;
	uint32 weightsLength;
	activationFunction activationFun;
	double* weights;
	double* output;
};
	}}

