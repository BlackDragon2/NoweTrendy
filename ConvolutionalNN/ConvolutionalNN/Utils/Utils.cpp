
#include "Utils.h"


namespace cnn {
	namespace utils {


size_t align(size_t pValue, size_t pAlign){
	return static_cast<size_t>((std::ceil(static_cast<double>(pValue) / pAlign)) * pAlign);
}


size_t getCvMatBytesCount(cv::Mat const& pMat){
	return static_cast<size_t>(pMat.size().area() * pMat.channels() * sizeof(uchar));
}


int bigRand(){
	return
		((rand() & 0x00FF)			) |
		((rand() & 0x00FF) << 8		) |
		((rand() & 0x00FF) << 16	) |
		((rand() & 0x7F00) << 24	);
}


	}
}