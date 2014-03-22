
#include "GpuBuffer.cuh"

#include "../Utils/Utils.h"


namespace cnn {
	namespace gpu {


GpuBuffer::GpuBuffer()
:
	mAddress(nullptr),
	mByteSize(0UL),
	mByteAlignment(0UL)
{

}


GpuBuffer::GpuBuffer(size_t pBytesCount, size_t pByteAlignment)
:
	mAddress(nullptr),
	mByteSize(0UL),
	mByteAlignment(0UL)
{
	allocate(pBytesCount, pByteAlignment);
}


GpuBuffer::~GpuBuffer(){
	free();
}


void GpuBuffer::allocate(size_t pBytesCount, size_t pByteAlignment){
	cudaError_t result = cudaMalloc<uchar>(&mAddress, pBytesCount);
	assert(result == cudaSuccess);
	mByteSize		= pBytesCount;
	mByteAlignment	= pByteAlignment;
}


void GpuBuffer::free(){
	if(mAddress != nullptr){
		cudaError_t result = cudaFree(reinterpret_cast<void*>(mAddress));
		assert(result == cudaSuccess);
		mAddress		= nullptr;
		mByteSize		= 0UL;
		mByteAlignment	= 0UL;
	}
}


void GpuBuffer::reallocate(size_t pBytesCount, size_t pByteAlignment){
	free();
	allocate(pBytesCount, pByteAlignment);
}


size_t GpuBuffer::getByteSize() const {
	return utils::align(mByteSize, mByteAlignment);
}


size_t GpuBuffer::getAlignment() const {
	return mByteAlignment;
}


	}
}