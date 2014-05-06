
#include "CudaUtils.h"

namespace cnn {
	namespace utils {


size_t blocksCount(size_t pThreadsCount, size_t pBlockSize){
	return static_cast<size_t>(std::ceil(static_cast<double>(pThreadsCount) / pBlockSize));
}


	}
}