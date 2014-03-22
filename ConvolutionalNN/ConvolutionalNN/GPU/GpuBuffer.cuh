
#ifndef CNN_GPU_BUFFER_H_
#define CNN_GPU_BUFFER_H_


#include <assert.h>

#include "cuda_runtime.h"

#include "../Types.h"


namespace cnn {
	namespace gpu {


class GpuBuffer {
public:
	GpuBuffer();
	GpuBuffer(size_t pBytesCount, size_t pByteAlignment = 32UL);
	template <typename T> GpuBuffer(size_t pBytesCount, T* pData, size_t pByteAlignment = 32UL);
	virtual ~GpuBuffer();


	void allocate(size_t pBytesCount, size_t pByteAlignment = 32UL);
	void free();
	void reallocate(size_t pBytesCount, size_t pByteAlignment = 32UL);

	template <typename T> void writeToDevice(T* pData, size_t pElementsCount);
	template <typename T> void loadFromDevice(T* pData, size_t pElementsCount);


	template <typename T> T*		dataPtr();
	template <typename T> T const*	dataPtr() const;

	template <typename T> T*		operator&();
	template <typename T> T const*	operator&() const;

	size_t getByteSize()	const;
	size_t getAlignment()	const;


private:
	GpuBuffer(GpuBuffer const& pBuffer);
	GpuBuffer& operator=(GpuBuffer const& pBuffer);


private:
	uchar*	mAddress;
	size_t	mByteSize;
	size_t	mByteAlignment;
};


template <typename T> 
GpuBuffer::GpuBuffer(size_t pBufferUnitsCount, T* pData, size_t pByteAlignment)
:
	mAddress(nullptr),
	mByteSize(0UL),
	mByteAlignment(0UL)
{
	allocate(pBufferUnitsCount, pByteAlignment);
	writeToDevice(pData, pBufferUnitsCount);
}


template <typename T>
void GpuBuffer::writeToDevice(T* pData, size_t pElementsCount){
	size_t amountBytes = pElementsCount * sizeof(T);
	assert(amountBytes <= mByteSize);
	cudaError_t result = cudaMemcpy(mAddress, pData, amountBytes, cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
}
	

template <typename T>
void GpuBuffer::loadFromDevice(T* pData, size_t pElementsCount){
	size_t amountBytes = pElementsCount * sizeof(T);
	cudaError_t result = cudaMemcpy(pData, mAddress, amountBytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
}


template <typename T>
T* GpuBuffer::dataPtr(){
	return reinterpret_cast<T*>(mAddress);
}


template <typename T>
T const* GpuBuffer::dataPtr() const {
	return reinterpret_cast<T const*>(mAddress);
}


template <typename T>
T* GpuBuffer::operator&(){
	return dataPtr<T>();
}


template <typename T>
T const* GpuBuffer::operator&() const {
	return dataPtr<T>();
}


	}
}


#endif	/* CNN_GPU_BUFFER_H_ */