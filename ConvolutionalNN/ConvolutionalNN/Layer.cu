#include "Layer.cuh"


cnn::nn::Layer::Layer(uint32 neuronsNr, uint32 inputLength, activationFunction fun)
{
	this->neuronsNr=neuronsNr;
	this->inputLength=inputLength;
	this->weightsLength=neuronsNr*inputLength;
	this->weights=new double[weightsLength];
	this->output=new double[neuronsNr];
	this->activationFun=fun;
}

void cnn::nn::Layer::initWeights(double min, double max)
{
	srand(time(0));
	for(int i=0;i<weightsLength;i++)
		weights[i]=min+(max-min)*(rand()/(INT_MAX+1));
}

cnn::nn::Layer::~Layer(void)
{
	delete[] weights;
	delete[] output;
}
