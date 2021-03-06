
#ifndef CNN_UTILS_H_
#define CNN_UTILS_H_

#include "../Types.h"
#include "CudaUtils.h"
#include "CvUtils.h"


namespace cnn {
	namespace utils {


size_t align(size_t pValue, size_t pAlign);

uint32 randU();
uint32 bigRand32();
uint64 bigRand64();


	}
}


#endif	/* CNN_UTILS_H_ */