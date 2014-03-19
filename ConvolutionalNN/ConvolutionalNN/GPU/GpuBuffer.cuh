
#ifndef CNN_GPU_BUFFER_H_
#define CNN_GPU_BUFFER_H_


#include <assert.h>

#include "cuda_runtime.h"


namespace cnn {
	namespace gpu {


template <typename T>
class GpuBuffer {
public:
	GpuBuffer();
	GpuBuffer(size_t pBufferUnitsCount);
	GpuBuffer(size_t pBufferUnitsCount, T* pData);
	virtual ~GpuBuffer();


	void allocateBytes(size_t pBufferBytesCount);
	void allocateUnits(size_t pBufferUnitsCount);
	void free();
	void reallocateBytes(size_t pBufferBytesCount);
	void reallocateUnits(size_t pBufferUnitsCount);

	void writeToDevice(T* pData, size_t pAmout);
	void loadFromDevice(T* pData, size_t pAmout);

	T*			operator&();
	T const*	operator&() const;

	size_t getBufferByteSize() const;
	size_t getBufferUnitSize() const;


private:
	T*		mAddress;
	size_t	mBufferByteSize;
};


template <typename T>
GpuBuffer<T>::GpuBuffer()
:
	mAddress(nullptr),
	mBufferByteSize(0UL)
{

}


template <typename T>
GpuBuffer<T>::GpuBuffer(size_t pBufferUnitsCount)
:
	mAddress(nullptr),
	mBufferByteSize(0UL)
{
	allocateUnits(pBufferUnitsCount);
}


template <typename T>
GpuBuffer<T>::GpuBuffer(size_t pBufferUnitsCount, T* pData)
:
	mAddress(nullptr),
	mBufferByteSize(0UL)
{
	allocateUnits(pBufferUnitsCount);
	writeToDevice(pData, pBufferUnitsCount);
}


template <typename T>
GpuBuffer<T>::~GpuBuffer(){
	free();
}


template <typename T>
void GpuBuffer<T>::allocateBytes(size_t pBufferBytesCount){
	cudaError_t result = cudaMalloc<T>(&mAddress, pBufferBytesCount);
	assert(result == cudaSuccess);
	mBufferByteSize = pBufferBytesCount;
}


template <typename T>
void GpuBuffer<T>::allocateUnits(size_t pBufferUnitsCount){
	allocateBytes(pBufferUnitsCount * sizeof(T));
}


template <typename T>
void GpuBuffer<T>::free(){
	if(mAddress != nullptr){
		cudaFree(reinterpret_cast<void*>(mAddress));
		mAddress		= nullptr;
		mBufferByteSize = 0UL;
	}
}


template <typename T>
void GpuBuffer<T>::reallocateBytes(size_t pBufferBytesCount){
	free();
	allocateBytes(pBufferBytesCount);
}


template <typename T>
void GpuBuffer<T>::reallocateUnits(size_t pBufferUnitsCount){
	reallocateBytes(pBufferUnitsCount * sizeof(T));
}


template <typename T>
void GpuBuffer<T>::writeToDevice(T* pData, size_t pAmout){
	size_t amountBytes = pAmout * sizeof(T);
	if(amountBytes > mBufferByteSize)
		reallocateBytes(amountBytes);
	cudaError_t result = cudaMemcpy(mAddress, pData, amountBytes, cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
}
	

template <typename T>
void GpuBuffer<T>::loadFromDevice(T* pData, size_t pAmout){
	size_t amountBytes = pAmout * sizeof(T);
	cudaError_t result = cudaMemcpy(pData, mAddress, amountBytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
}


template <typename T>
T* GpuBuffer<T>::operator&(){
	return mAddress;
}


template <typename T>
T const* GpuBuffer<T>::operator&() const {
	return mAddress;
}


template <typename T>
size_t GpuBuffer<T>::getBufferByteSize() const {
	return mBufferByteSize;
}


template <typename T>
size_t GpuBuffer<T>::getBufferUnitSize() const {
	return getBufferByteSize() / sizeof(T);
}


	}
}


#endif	/* CNN_GPU_BUFFER_H_ */