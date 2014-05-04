
#ifndef CNN_CV_UTILS_H_
#define CNN_CV_UTILS_H_

#include <opencv2/highgui/highgui.hpp>


namespace cnn {
	namespace utils {


size_t getCvMatBytesCount(cv::Mat const& pMat);

int decodeCvChannelsCount(int pType);


template <typename T> 
int createCvImageType(int pChannelsCount){
	return (sizeof(T) == 4UL ? CV_32FC(pChannelsCount) : CV_8UC(pChannelsCount));
}


	}
}


#endif	/* CNN_CV_UTILS_H_ */