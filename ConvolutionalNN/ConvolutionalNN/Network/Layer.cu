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

cnn::gpu::GpuBuffer* cnn::nn::Layer::getWeightedErrorRates()
{
	cnn::gpu::GpuBuffer* weightedError=new cnn::gpu::GpuBuffer();
	weightedError->allocate(inputLength*sizeof(float));
	int blocks;
	if(weightsLength%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::calculateWeightedError<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(errorRatesDev.getDataPtr<float>(), weightsDev.getDataPtr<float>()+neuronsNr, weightedError->getDataPtr<float>(), inputLength, neuronsNr);
	return weightedError;
}

uint32 cnn::nn::Layer::getNeuronsNr()
{
	return neuronsNr;
}

float cnn::nn::Layer::calculateError(uint32 exampleClass)
{
	int blocks;
	cnn::gpu::GpuBuffer error;
	error.allocate(sizeof(float));
	if(neuronsNr%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK+1;
	if(activationFun==SIGMOIDAL)
		cnn::cuda::calculateSigmoidalError<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(exampleClass, outputDev.getDataPtr<float>(), neuronsNr, errorRatesDev.getDataPtr<float>() , error.getDataPtr<float>());
	else
		cnn::cuda::calculateTanhError<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(exampleClass, outputDev.getDataPtr<float>(), neuronsNr, errorRatesDev.getDataPtr<float>() ,error.getDataPtr<float>());
	float result;
	error.loadFromDevice(&result, sizeof(float));
	return result;
}

void cnn::nn::Layer::calculateError(cnn::gpu::GpuBuffer* errorRates)
{
	int blocks;
	if(neuronsNr%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=neuronsNr/config::Cuda::THREADS_PER_BLOCK+1;
	if(activationFun==SIGMOIDAL)
		cnn::cuda::calculateSigmoidalDelta<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr, errorRatesDev.getDataPtr<float>(), errorRates->getDataPtr<float>());
	else
		cnn::cuda::calculateSigmoidalDelta<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(outputDev.getDataPtr<float>(), neuronsNr, errorRatesDev.getDataPtr<float>() ,errorRates->getDataPtr<float>());
	errorRates->free();
}

void cnn::nn::Layer::updateWeights()
{
	int blocks;
	if(weightsLength%config::Cuda::THREADS_PER_BLOCK==0)
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=weightsLength/config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::updateWeights<<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(weightsDev.getDataPtr<float>(), weightsUpdateDev.getDataPtr<float>(), weightsLength);
}