#ifndef CNN_MAX_POOL_LAYER_H_
#define CNN_MAX_POOL_LAYER_H_

#include "VirtualLayer.h"
#include "../Config.h"
#include "../Network/LayerCuda.cuh"

namespace mycnn{

class MaxPoolLayer : public VirtualLayer
{
public:
	MaxPoolLayer(uint32 kernelSize, uint32 inputWidth, uint32 inputHeight, uint32 inputSize);
	~MaxPoolLayer();
	uint32 getKernelSize();
	uint32 getOutputWidth();
	uint32 getOutputHeight();
	uint32 getOutputSize();
	uint32 getInputImagesCount();
	uint32 getImageSize();
	cnn::gpu::GpuBuffer* getOutput();
	void run(cnn::gpu::GpuBuffer* buffer, uint32 offset);
	void pooling(cnn::gpu::GpuBuffer* input, uint32 offset);
	void teach(cnn::gpu::GpuBuffer* errorProp, cnn::gpu::GpuBuffer* input, float learningRate, uint32 offset);
	cnn::gpu::GpuBuffer* getError();
	void updateWeights();
	void resetWeightsUpdates();
private:
	cnn::gpu::GpuBuffer* output;
	cnn::gpu::GpuBuffer* errorRates;
	uint32 kernelSize;
	uint32 inputSize;//liczba elementow (floatow) w wejsciu
	uint32 inputWidth;
	uint32 inputHeight;
};

}

#endif	/* CNN_MAX_POOL_LAYER_H_ */