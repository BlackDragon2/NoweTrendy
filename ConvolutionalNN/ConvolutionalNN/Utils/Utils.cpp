
#include <algorithm>

#include "Utils.h"


namespace cnn {
	namespace utils {


size_t align(size_t pValue, size_t pAlign){
	return static_cast<size_t>((std::ceil(static_cast<double>(pValue) / pAlign)) * pAlign);
}


uint32 bigRand32(){
	return
		((rand() & 0x00FF)			) |
		((rand() & 0x00FF) << 8		) |
		((rand() & 0x00FF) << 16	) |
		((rand() & 0xFF00) << 24	);
}

uint64 bigRand64(){
	return (static_cast<uint64>(bigRand32()) << 32) | static_cast<uint64>(bigRand32());
}


	}
}