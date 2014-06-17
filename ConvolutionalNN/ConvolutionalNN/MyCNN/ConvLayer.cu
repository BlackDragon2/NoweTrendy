#include "ConvLayer.cuh"

uint32 mycnn::ConvLayer::getKernelsNr()
{
	return kernelsNr;
}

uint32 mycnn::ConvLayer::getKernelSize()
{
	return kernelSize;
}

cnn::gpu::GpuBuffer* mycnn::ConvLayer::getKernels()
{
	return kernels;
}

cnn::gpu::GpuBuffer* mycnn::ConvLayer::getOutput()
{
	return output;
}

void mycnn::ConvLayer::initKernels()
{
	float* temp=new float[kernelSize*kernelSize*kernelsNr];
	for(uint32 i=0;i<kernelSize*kernelSize*kernelsNr;i++)
		temp[i]=((float)rand())/RAND_MAX;
	kernels->writeToDevice(temp, kernelSize*kernelSize*kernelsNr*sizeof(float));
	delete[] temp;
}

mycnn::ConvLayer::ConvLayer(uint32 kernelNr, uint32 kernelSize, uint32 inputWidht, uint32 inputHeight, uint32 inputSize)
{
	this->kernelsNr=kernelsNr;
	this->kernelSize=kernelSize;
	this->inputSize=inputSize;
	this->inputWidth=inputWidht;
	this->inputHeight=inputHeight;
	output=new cnn::gpu::GpuBuffer();
	kernels=new cnn::gpu::GpuBuffer();
	weightsUpdate=new cnn::gpu::GpuBuffer();
	errorRates=new cnn::gpu::GpuBuffer();
	output->allocate(getOutputSize()*sizeof(float));
	kernels->allocate(kernelNr*kernelSize*kernelSize*sizeof(float));
	weightsUpdate->allocate(kernelNr*kernelSize*kernelSize*sizeof(float));
	errorRates->allocate(inputSize*sizeof(float));
	initKernels();
}

mycnn::ConvLayer::~ConvLayer()
{
	output->free();
	kernels->free();
	weightsUpdate->free();
	errorRates->free();
	delete output;
	delete kernels;
	delete weightsUpdate;
	delete errorRates;
}

uint32 mycnn::ConvLayer::getOutputWidth()
{
	return inputWidth/kernelSize;
}

uint32 mycnn::ConvLayer::getInputImagesCount()
{
	return inputSize/(inputWidth*inputHeight);
}

uint32 mycnn::ConvLayer::getOutputHeight()
{
	return inputHeight/kernelSize;
}

uint32 mycnn::ConvLayer::getOutputSize()
{
	return getOutputWidth()*getOutputHeight()*kernelsNr*getInputImagesCount();
}

void mycnn::ConvLayer::convolution(cnn::gpu::GpuBuffer* input, uint32 offset)
{
	int blockSize=getOutputHeight()*getOutputWidth();
	dim3 block(getOutputWidth(), getOutputHeight());//blok - jeden kernel z jednym obrazem
	dim3 grid(getInputImagesCount(), kernelsNr);//grid kernele x obrazy
	cnn::cuda::convolution<<<grid, block>>>(input->getDataPtr<float>()+offset, output->getDataPtr<float>(), kernels->getDataPtr<float>(), kernelSize, inputSize, inputWidth, inputHeight, getOutputWidth(), getOutputHeight());
}

void mycnn::ConvLayer::teach(cnn::gpu::GpuBuffer* errorProp, cnn::gpu::GpuBuffer* input, float learningRate, uint32 offset)
{
	int blockSize=getOutputHeight()*getOutputWidth();
	dim3 block(getOutputWidth(), getOutputHeight());//blok - jeden kernel z jednym obrazem
	dim3 grid(getInputImagesCount(), kernelsNr);//grid kernele x obrazy
	cnn::cuda::errorConv<<<grid, block>>>(input->getDataPtr<float>()+offset, errorProp->getDataPtr<float>(), kernels->getDataPtr<float>(), kernelSize, inputSize, inputWidth, inputHeight, getOutputWidth(), getOutputHeight(), errorRates->getDataPtr<float>(), weightsUpdate->getDataPtr<float>(), learningRate);
}

void mycnn::ConvLayer::run(cnn::gpu::GpuBuffer* buffer, uint32 offset)
{
	convolution(buffer, offset);
}

uint32 mycnn::ConvLayer::getImageSize()
{
	return getOutputHeight()*getOutputWidth();
}

cnn::gpu::GpuBuffer* mycnn::ConvLayer::getError()
{
	return errorRates;
}

void mycnn::ConvLayer::updateWeights()
{
	int blocks;
	uint32 kernelsCap=getKernelsNr()*getKernelSize()*getKernelSize();
	if(kernelsCap%cnn::config::Cuda::THREADS_PER_BLOCK==0)
		blocks=kernelsCap/cnn::config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=kernelsCap/cnn::config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::updateWeights<<<blocks, cnn::config::Cuda::THREADS_PER_BLOCK>>>(kernels->getDataPtr<float>(), weightsUpdate->getDataPtr<float>(), kernelsCap);
}

void mycnn::ConvLayer::resetWeightsUpdates()
{
	int blocks;
	uint32 kernelsCap=getKernelsNr()*getKernelSize()*getKernelSize();
	if(kernelsCap%cnn::config::Cuda::THREADS_PER_BLOCK==0)
		blocks=kernelsCap/cnn::config::Cuda::THREADS_PER_BLOCK;
	else
		blocks=kernelsCap/cnn::config::Cuda::THREADS_PER_BLOCK+1;
	cnn::cuda::reset<float><<<blocks, cnn::config::Cuda::THREADS_PER_BLOCK>>>(weightsUpdate->getDataPtr<float>(), kernelsCap);
}
