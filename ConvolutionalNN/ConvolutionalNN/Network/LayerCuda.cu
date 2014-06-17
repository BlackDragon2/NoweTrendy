#include "LayerCuda.cuh"

__device__ float cnn::cuda::add(float* address, float value)
{
  float old = value;  
  float ret=atomicExch(address, 0.0f);
  float new_old=ret+old;
  while ((old = atomicExch(address, new_old))!=0.0f)
  {
	new_old = atomicExch(address, 0.0f);
	new_old += old;
  }
  return ret;
}

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

__global__ void cnn::cuda::convolution(float* input, float* output, float* kernels, uint32 kernelSize, uint32 inputSize, uint32 inputWidth, uint32 inputHeight, uint32 outputWidth, uint32 outputHeight)
{
	uint32 imCount=inputSize/(inputWidth*inputHeight);
	uint32 imDist=blockIdx.y*inputWidth*inputHeight;//wyznacza ktory obraz wejscia
	uint32 kerDist=blockIdx.x*kernelSize*kernelSize;//wyznacza ktory kernel
	uint32 outImDist=blockIdx.y*outputWidth*outputHeight;//przesuniecie dla calosci robionej przez jednego kernela
	uint32 outKerDist=blockIdx.x*outputWidth*outputHeight*imCount;//przesuniecie dla obrazu zrobienego przez kernel
	uint32 x=threadIdx.x*kernelSize+kernelSize/2;//wyznacza kolumne (wspolrzedna x)
	uint32 y=threadIdx.y*kernelSize+kernelSize/2;//wyznacza wiersz (wspolrzedna y)
	float val=0;
	for(uint32 i=-kernelSize/2;i<=kernelSize/2;i++)
		for(uint32 j=-kernelSize/2;j<=kernelSize/2;j++)
			val+=input[(y+j)*inputWidth+x+i+imDist]*kernels[(j+kernelSize/2)*kernelSize+i+kernelSize/2+kerDist];
		output[threadIdx.x+outputWidth*threadIdx.y+outKerDist+outImDist]=val;//na wyjsciu kolejno wiersze, kolumny, wszystkie obrazy dla 1 kernela, reszta kerneli

}

__global__ void cnn::cuda::maxPooling(float* input, float* output, uint32 kernelSize, uint32 inputWidth, uint32 inputHeight, uint32 outputWidth, uint32 outputHeight, float* errorRates)
{
	uint32 imDist=blockIdx.x*inputWidth*inputHeight;
	uint32 outImDist=blockIdx.y*outputWidth*outputHeight;
	uint32 x=threadIdx.x*kernelSize;
	uint32 y=threadIdx.y*kernelSize;
	uint32 maxId=y*inputWidth+x+imDist;
	float val=input[y*inputWidth+x+imDist];
	errorRates[maxId]=0;
	for(uint32 i=0;i<kernelSize;i++)
	{
		for(uint32 j=0;j<kernelSize;j++)
		{
			if(input[(y+j)*inputWidth+x+i+imDist]>val)
			{
				val=input[(y+j)*inputWidth+x+i+imDist];
				maxId=(y+j)*inputWidth+x+i+imDist;
			}
			errorRates[(y+j)*inputWidth+x+i+imDist]=0;
		}
	}
	errorRates[maxId]=1;
	output[threadIdx.x+outputWidth*threadIdx.y+outImDist]=val;
}

__global__ void cnn::cuda::maxError(float* errorRates, uint32 kernelSize, uint32 inputWidth, uint32 inputHeight, uint32 outputWidth, uint32 outputHeight, float* errorProp)
{
	uint32 imDist=blockIdx.x*inputWidth*inputHeight;
	uint32 imDistProp=blockIdx.x*outputWidth*outputHeight;
	uint32 x=threadIdx.x*kernelSize;
	uint32 y=threadIdx.y*kernelSize;
	for(uint32 i=0;i<kernelSize;i++)
	{
		for(uint32 j=0;j<kernelSize;j++)
			errorRates[(y+j)*inputWidth+x+i+imDist]=errorRates[(y+j)*inputWidth+x+i+imDist]*errorProp[threadIdx.x+threadIdx.y*outputWidth+imDistProp];
	}
}

__global__ void cnn::cuda::errorConv(float* input, float* errorProp, float* kernels, uint32 kernelSize, uint32 inputSize, uint32 inputWidth, uint32 inputHeight, uint32 outputWidth, uint32 outputHeight, float* errorRates, float* weightsUpdate, float learningRate)
{
	uint32 imCount=inputSize/(inputWidth*inputHeight);
	uint32 imDist=blockIdx.y*inputWidth*inputHeight;//wyznacza ktory obraz wejscia
	uint32 kerDist=blockIdx.x*kernelSize*kernelSize;//wyznacza ktory kernel
	uint32 outImDist=blockIdx.y*outputWidth*outputHeight;//przesuniecie dla calosci robionej przez jednego kernela
	uint32 outKerDist=blockIdx.x*outputWidth*outputHeight*imCount;//przesuniecie dla obrazu zrobienego przez kernel
	uint32 x=threadIdx.x*kernelSize+kernelSize/2;//wyznacza kolumne (wspolrzedna x)
	uint32 y=threadIdx.y*kernelSize+kernelSize/2;//wyznacza wiersz (wspolrzedna y)
	for(uint32 i=-kernelSize/2;i<=kernelSize/2;i++)
		for(uint32 j=-kernelSize/2;j<=kernelSize/2;j++)
		{
			add(&weightsUpdate[(j+kernelSize/2)*kernelSize+i+kernelSize/2+kerDist],learningRate*input[(y+j)*inputWidth+x+i+imDist]*errorProp[threadIdx.x+outputWidth*threadIdx.y+outKerDist+outImDist]);
			errorRates[(y+j)*inputWidth+x+i+imDist]=kernels[(j+kernelSize/2)*kernelSize+i+kernelSize/2+kerDist]*errorProp[threadIdx.x+outputWidth*threadIdx.y+outKerDist+outImDist];
		}
}

