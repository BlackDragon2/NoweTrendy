
#ifndef CNN_UTILS_H_
#define CNN_UTILS_H_


#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>

#include "../Types.h"


namespace cnn {
	namespace utils {


size_t align(size_t pValue, size_t pAlign);

size_t getCvMatBytesCount(cv::Mat const& pMat);

uint	bigRand32();
size_t	bigRand64();


	}
}


#endif	/* CNN_UTILS_H_ */