
#ifndef CNN_NORMALIZER_H_
#define CNN_NORMALIZER_H_

#include "GpuBuffer.cuh"
#include "../ImageBatch.h"


namespace cnn {
	namespace gpu {


template <typename T>
class Normalizer {
public:
	typedef std::shared_ptr<Normalizer<T>> PtrS;


public:
	Normalizer();
	virtual ~Normalizer();
	
	virtual void build(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pOutputBuffer) = 0;

	virtual void normalize(
		ImageBatch<T> const&	pImageBatch, 
		GpuBuffer&				pInputBuffer,
		GpuBuffer&				pBuilderBuffer,
		GpuBuffer&				pOutputBuffer) = 0;
};


template <typename T>
Normalizer<T>::Normalizer(){

}


template <typename T>
Normalizer<T>::~Normalizer(){

}


	}
}

#endif	/* CNN_NORMALIZER_H_ */