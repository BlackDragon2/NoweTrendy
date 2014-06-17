#ifndef CNN_VIRTUAL_LAYER_H_
#define CNN_VIRTUAL_LAYER_H_

#include "../Types.h"
#include "../GPU/GpuBuffer.cuh"
#include "../Config.h"

namespace mycnn{
class VirtualLayer
{
public:
	virtual void run(cnn::gpu::GpuBuffer* buffer, uint32 offset)=0;
	virtual cnn::gpu::GpuBuffer* getOutput()=0;
	virtual uint32 getImageSize()=0;
	virtual void teach(cnn::gpu::GpuBuffer* errorProp, cnn::gpu::GpuBuffer* input, float learningRate, uint32 offset)=0;
	virtual cnn::gpu::GpuBuffer* getError()=0;
	virtual void updateWeights()=0;
	virtual void resetWeightsUpdates()=0;
};}

#endif	/* CNN_VIRTUAL_LAYER_H_ */