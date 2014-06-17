#ifndef CNN_CONV_LAYER_H_
#define CNN_CONV_LAYER_H_

#include "VirtualLayer.h"
#include "../Config.h"
#include "../Network/LayerCuda.cuh"

namespace mycnn{

class ConvLayer : public VirtualLayer
{
public:
	ConvLayer(uint32 kernelNr, uint32 kernelSize, uint32 inputWidth, uint32 inputHeight, uint32 inputSize);
	~ConvLayer();
	uint32 getKernelSize();
	uint32 getKernelsNr();
	uint32 getOutputWidth();
	uint32 getOutputHeight();
	uint32 getOutputSize();
	cnn::gpu::GpuBuffer* getKernels();
	cnn::gpu::GpuBuffer* getOutput();
	void initKernels();
	void convolution(cnn::gpu::GpuBuffer* input, uint32 offset);
	uint32 getInputImagesCount();
	void run(cnn::gpu::GpuBuffer* buffer, uint32 offset);
	uint32 getImageSize();
	void teach(cnn::gpu::GpuBuffer* errorProp, cnn::gpu::GpuBuffer* input, float learningRate, uint32 offset);
	cnn::gpu::GpuBuffer* getError();
	void updateWeights();
	void resetWeightsUpdates();
private:
	cnn::gpu::GpuBuffer* kernels;
	cnn::gpu::GpuBuffer* output;
	cnn::gpu::GpuBuffer* weightsUpdate;
	cnn::gpu::GpuBuffer* errorRates;
	uint32 kernelSize;
	uint32 kernelsNr;
	uint32 inputSize;//liczba elementow (floatow) w wejsciu
	uint32 inputWidth;
	uint32 inputHeight;
};}



#endif	/* CNN_CONV_LAYER_H_ */