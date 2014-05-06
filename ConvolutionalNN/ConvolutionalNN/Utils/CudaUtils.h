

#ifndef CNN_CUDA_UTILS_H_
#define CNN_CUDA_UTILS_H_

#include <opencv2/highgui/highgui.hpp>


namespace cnn {
	namespace utils {


size_t blocksCount(size_t pThreadsCount, size_t pBlockSize);


	}
}


#endif	/* CNN_CUDA_UTILS_H_ */