
#include <algorithm>

#include "Utils.h"


namespace cnn {
	namespace utils {


size_t align(size_t pValue, size_t pAlign){
	return static_cast<size_t>((std::ceil(static_cast<double>(pValue) / pAlign)) * pAlign);
}


// returns [0 : 0x7FFF]
// returns [0 : 2^15 - 1]
uint32 randU(){
	return static_cast<uint32>(rand());
}


// returns [0 : 0xFFFFFFFF]
// returns [0 : 2^32]
uint32 bigRand32(){
	return
		((randU() & 0xFFU)		) |
		((randU() & 0xFFU) << 8	) |
		((randU() & 0xFFU) << 16	) |
		((randU() & 0xFFU) << 24	);
}


// returns [0 : 0xFFFFFFFFFFFFFFFF]
// returns [0 : 2^64]
uint64 bigRand64(){
	return (static_cast<uint64>(bigRand32()) << 32) | static_cast<uint64>(bigRand32());
}


	}
}