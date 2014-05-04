
#ifndef CONFIG_H
#define CONFIG_H


#include <string>

#include "Types.h"


namespace cnn {
	namespace config {


struct Author {
	static const std::string NAME;
	static const std::string EMAIL;
};
const std::string Author::NAME	= "Michal";
const std::string Author::EMAIL	= "michal.wendel@gmail.com";


struct Cuda {
	static const uint32 THREADS_PER_BLOCK	= 1024U;
	static const uint32 CUDA_DEVICE_ID		= 0U;
};


	}
}


#endif	/* CONFIG_H */