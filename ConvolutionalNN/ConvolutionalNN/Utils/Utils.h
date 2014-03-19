
#ifndef CNN_UTILS_H_
#define CNN_UTILS_H_


#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>


namespace cnn {
	namespace utils {


template<typename T>
T align(T pValue, T pAlign){
	return static_cast<T>((std::ceil(static_cast<double>(pValue) / pAlign)) * pAlign);
}


size_t getCvMatBytesCount(cv::Mat const& pMat){
	return static_cast<size_t>(pMat.size().area() * pMat.channels() * sizeof(uchar));
}


	}
}


#endif	/* CNN_UTILS_H_ */