
#include "CvUtils.h"


namespace cnn {
	namespace utils {


size_t getCvMatBytesCount(cv::Mat const& pMat){
	return static_cast<size_t>(pMat.size().area() * pMat.channels() * sizeof(uchar));
}


int decodeCvChannelsCount(int pType){
	return (pType / 8) + 1;
}


	}
}