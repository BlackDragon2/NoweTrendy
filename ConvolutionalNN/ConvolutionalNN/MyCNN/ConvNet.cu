#include "ConvNet.h"

mycnn::ConvNet::ConvNet(float learningRate)
{
	this->learningRate=learningRate;
}

void mycnn::ConvNet::run(cnn::gpu::GpuBuffer* buffer, uint32 offset)
{
	layers[0]->run(buffer, offset);
	for(uint32 i=1;i<layers.size();i++)
		layers[i]->run(layers[i-1]->getOutput(), 0);
}
void mycnn::ConvNet::addLayer(ConvLayer layer)
{
	layers.push_back(&layer);
}

void mycnn::ConvNet::addLayer(MaxPoolLayer layer)
{
	layers.push_back(&layer);
}

void mycnn::ConvNet::train(cnn::gpu::GpuBuffer* error, cnn::gpu::GpuBuffer* input, uint32 offset)
{
	layers[layers.size()-1]->teach(error, layers[layers.size()-2]->getOutput(), learningRate, 0);
	for(int i=layers.size()-2;i>0;i--)
		layers[i]->teach(layers[i+1]->getError(), layers[i-1]->getOutput(), learningRate, 0);
	layers[0]->teach(layers[1]->getError(), input, learningRate, offset);
}

void mycnn::ConvNet::updateWeights()
{
	for(uint32 i=0;i<layers.size();i++)
		layers[i]->updateWeights();
}

void mycnn::ConvNet::resetWeightsUpdates()
{
	for(uint32 i=0;i<layers.size();i++)
		layers[i]->resetWeightsUpdates();
}