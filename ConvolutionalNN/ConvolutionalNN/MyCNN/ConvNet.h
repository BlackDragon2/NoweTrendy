#ifndef CNN_CONV_NET_H_
#define CNN_CONV_NET_H_

#include <vector>
#include "VirtualLayer.h"
#include "MaxPoolLayer.h"
#include "ConvLayer.h"

namespace mycnn{
class ConvNet
{
public:
	ConvNet(float learningRate);
	void run(cnn::gpu::GpuBuffer* buffer, uint32 offset);
	void addLayer(ConvLayer layer);
	void addLayer(MaxPoolLayer layer);
	template<typename T>
	T* getLastLayerOutput(uint32 index);
	void train(cnn::gpu::GpuBuffer* error, cnn::gpu::GpuBuffer* input, uint32 offset);
	void updateWeights();
	void resetWeightsUpdates();
private:
	std::vector<VirtualLayer*> layers;
	float learningRate;
};}

template<typename T>
T* mycnn::ConvNet::getLastLayerOutput(uint32 index)//TO DO WITH GRAYSCALE
{
	return layers[layers.size()-1]->getOutput()->getDataPtr<T>()+layers[layers.size()-1]->getImageSize()*index/3;
}

#endif	/* CNN_CONV_NET_H_ */