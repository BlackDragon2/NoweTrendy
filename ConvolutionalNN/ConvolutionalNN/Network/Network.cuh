#ifndef CNN_NETWORK_H_
#define CNN_NETWORK_H_

#include "../GPU/GpuBuffer.cuh"
#include <stdlib.h>
#include <vector>
#include "Layer.cuh"
#include "../Config.h"
#include "../Types.h"

namespace cnn{
	namespace nn{
	
class Network
{
public:
	Network(uint32 batchSize, float learningRate, float stopError);
	~Network();
	void addLayer(uint32 neuronsNr, uint32 inputLength, activationFunction fun);
	void initWeights(float min, float max);
	void setClasses(std::string* classes, uint32 classesNr);
	void train();
	void classify();

private:
	uint32 batchSize;
	float learningRate;
	float stopError;
	std::vector<Layer*> layers;
	std::vector<std::string> classes;
};
	
	}}

#endif	/* CNN_NETWORK_H_ */