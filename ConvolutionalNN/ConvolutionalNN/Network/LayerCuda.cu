#include "LayerCuda.cuh"

__global__ void cnn::cuda::calculateSigmoidalOutput(float* output, uint32 neuronsNr, float* weights, float* biases)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=1/(1+expf(-output[idx]+(weights[idx]*biases[idx])));//aktywacja+bias
	}
}

__global__ void cnn::cuda::calculateTanhOutput(float* output, uint32 neuronsNr, float* weights, float* biases)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		output[idx]=tanhf(output[idx]+weights[idx]*biases[idx]);//aktywacja+bias
	}
}

__global__ void cnn::cuda::calculateTahnDelta(float* output, uint32 neuronsNr, float* errorRates, float* errorRatesProp)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		errorRates[idx]=(1-output[idx]*output[idx])*errorRatesProp[idx];//1-aktywacja^2*error warstwy wy¿ej
	}
}

__global__ void cnn::cuda::calculateSigmoidalDelta(float* output, uint32 neuronsNr, float* errorRates, float* errorRatesProp)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		errorRates[idx]=output[idx]*(1-output[idx])*errorRatesProp[idx];//aktywacja*(1-aktywacja)*error warstwy wy¿ej
	}
}

__global__ void cnn::cuda::calculateSigmoidalError(uint32 exampleClass, float* output, uint32 neuronsNr, float* errorRates, float* error)//oblicza b³ad na wyjsciu ostatniej wartswy
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		if(idx==exampleClass)
		{
			errorRates[idx]=(1-output[idx])*output[idx]*(1-output[idx]);//oczekiwane wyjscie - faktyczne * pochodna funcji aktywacji od wyjscia
			add(error, (1-output[idx])*(1-output[idx]));
		}
		else
		{
			errorRates[idx]=-output[idx]*output[idx]*(1-output[idx]);//oczekiwane wyjscie - faktyczne * pochodna funcji aktywacji od wyjscia
			add(error, -output[idx]*-output[idx]);
		}
	}
}

__global__ void cnn::cuda::calculateTanhError(uint32 exampleClass, float* output, uint32 neuronsNr, float* errorRates, float* error)//oblicza blad na wyjsciu ostatniej wartwy
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<neuronsNr)
	{
		if(idx==exampleClass)
		{
			errorRates[idx]=(1-output[idx])*(1-output[idx]*output[idx]);//oczekiwane wyjscie - faktyczne * pochodna funcji aktywacji od wyjscia
			add(error, (1-output[idx])*(1-output[idx]));
		}
		else
		{
			errorRates[idx]=-output[idx]*(1-output[idx]*output[idx]);//oczekiwane wyjscie - faktyczne * pochodna funcji aktywacji od wyjscia
			add(error, -output[idx]*-output[idx]);
		}
	}
}

__global__ void cnn::cuda::calculateWeightedError(float* errorRates, float* weights, float* weightedError, uint32 inputLength, uint32 neuronsNr)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<inputLength*neuronsNr)
	{
		add(&weightedError[idx%inputLength], errorRates[idx/inputLength]*weights[idx]);
	}
}

__global__ void cnn::cuda::updateWeights(float* weights, float* weigthsUpdate, uint32 weightsLength)
{
	uint32 idx		= ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx<weightsLength)
	{
		weights[idx]=weights[idx]+weigthsUpdate[idx];
	}
}
