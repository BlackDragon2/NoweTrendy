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

uint32 cnn::nn::Layer::getNeuronsNr()
{
	return neuronsNr;
}