#include "Layer.cuh"


cnn::nn::Layer::Layer(uint32 neuronsNr, uint32 inputLength, activationFunction fun)
{
	this->neuronsNr=neuronsNr;
	this->inputLength=inputLength;
	this->weightsLength=neuronsNr*(inputLength+1);
	this->weights=new float[weightsLength];
	this->output=new float[neuronsNr];
	this->activationFun=fun;
	this->biases=new float[neuronsNr];
	this->weightsDev.allocate(weightsLength*sizeof(float));
	this->weightsUpdateDev.allocate(weightsLength*sizeof(float));
	this->outputDev.allocate(neuronsNr*sizeof(float));
	this->biasesDev.allocate(neuronsNr*sizeof(float));
	this->biasesUpdateDev.allocate(neuronsNr*sizeof(float));
	this->errorRatesDev.allocate(neuronsNr*sizeof(float));
}

void cnn::nn::Layer::initWeights(float min, float max)
{
	srand((uint32)time(0));
	for(uint32 i=0;i<weightsLength;i++)
		weights[i]=min+(max-min)*(((float)rand())/RAND_MAX);
	for(uint32 i=0;i<neuronsNr;i++)
		biases[i]=min+(max-min)*(((float)rand())/RAND_MAX);
}

cnn::nn::Layer::~Layer(void)
{
	delete[] weights;
	delete[] output;
	delete[] biases;
	outputDev.free();
	weightsDev.free();
	weightsUpdateDev.free();
}

float* cnn::nn::Layer::getOutput()
{
	return output;
}

cnn::gpu::GpuBuffer* cnn::nn::Layer::getOutputBuffer()
{
	return &outputDev;
}

cnn::gpu::GpuBuffer* cnn::nn::Layer::getErrorRatesBuffer()
{
	return &errorRatesDev;
}

uint32 cnn::nn::Layer::getNeuronsNr()
{
	return neuronsNr;
}

float cnn::nn::Layer::calculateError(char* exampleClass, cnn::gpu::GpuBuffer* classes)
{
	int blocks;
	if(neuronsNr%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::calculateError<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(exampleClass, classes->getDataPtr<char*>, outputDev.getDataPtr<float>(), neuronsNr);
}

void cnn::nn::Layer::calculateError(cnn::gpu::GpuBuffer* errorRates)
{
	int blocks;
	if(neuronsNr%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK+1;
}