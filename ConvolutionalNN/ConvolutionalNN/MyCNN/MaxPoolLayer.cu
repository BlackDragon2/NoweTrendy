#include "MaxPoolLayer.cuh"

uint32 mycnn::MaxPoolLayer::getKernelSize()
{
	return kernelSize;
}

cnn::gpu::GpuBuffer* mycnn::MaxPoolLayer::getOutput()
{
	return output;
}

mycnn::MaxPoolLayer::MaxPoolLayer(uint32 kernelSize, uint32 inputWidht, uint32 inputHeight, uint32 inputSize)
{
	output=new cnn::gpu::GpuBuffer();
	errorRates=new cnn::gpu::GpuBuffer();
	this->kernelSize=kernelSize;
	this->inputSize=inputSize;
	this->inputWidth=inputWidht;
	this->inputHeight=inputHeight;
	output->allocate(getOutputSize()*sizeof(float));
	errorRates->allocate(inputSize*sizeof(float));
}

mycnn::MaxPoolLayer::~MaxPoolLayer()
{
	output->free();
	errorRates->free();
	delete output;
	delete errorRates;
}

uint32 mycnn::MaxPoolLayer::getOutputWidth()
{
	return inputWidth/kernelSize;
}

uint32 mycnn::MaxPoolLayer::getInputImagesCount()
{
	return inputSize/(inputWidth*inputHeight);
}

uint32 mycnn::MaxPoolLayer::getOutputHeight()
{
	return inputHeight/kernelSize;
}

uint32 mycnn::MaxPoolLayer::getOutputSize()
{
	return getOutputWidth()*getOutputHeight()*getInputImagesCount();
}

void mycnn::MaxPoolLayer::pooling(cnn::gpu::GpuBuffer* input, uint32 offset)
{
	int blockSize=getOutputHeight()*getOutputWidth();
	dim3 block(getOutputWidth(), getOutputHeight());//blok - jeden kernel z jednym obrazem
	cnn::cuda::maxPooling<<<getInputImagesCount(), block>>>(input->getDataPtr<float>(), output->getDataPtr<float>(), kernelSize, inputWidth, inputHeight, getOutputWidth(), getOutputHeight(), errorRates->getDataPtr<float>());
}

uint32 mycnn::MaxPoolLayer::getImageSize()
{
	return getOutputHeight()*getOutputWidth();
}

void mycnn::MaxPoolLayer::run(cnn::gpu::GpuBuffer* buffer, uint32 offset)
{
	pooling(buffer, offset);
}

void mycnn::MaxPoolLayer::teach(cnn::gpu::GpuBuffer* errorProp, cnn::gpu::GpuBuffer* input, float learningRate, uint32 offset)
{
	int blockSize=getOutputHeight()*getOutputWidth();
	dim3 block(getOutputWidth(), getOutputHeight());//blok - jeden kernel z jednym obrazem
	cnn::cuda::maxError<<<getInputImagesCount(), block>>>(errorRates->getDataPtr<float>(), output->getDataPtr<float>(), kernelSize, inputWidth, inputHeight, getOutputWidth(), getOutputHeight(), errorProp->getDataPtr<float>());
}

cnn::gpu::GpuBuffer* mycnn::MaxPoolLayer::getError()
{
	return errorRates;
}

void mycnn::MaxPoolLayer::updateWeights()
{}

void mycnn::MaxPoolLayer::resetWeightsUpdates()
{}