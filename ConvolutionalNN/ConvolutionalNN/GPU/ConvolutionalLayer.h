
#include "GpuBuffer.cuh"
#include "../Types.h"
#include "../Utils/Utils.h"


namespace cnn {
	namespace gpu {


template <typename T>
class ConvolutionalLayer {
public:
	struct MapsData {
		size_t count;
		size_t width;
		size_t height;
	};

	struct FiltersData {
		size_t width;
		size_t height;
		size_t shiftX;
		size_t shiftY;
	};


public:
	ConvolutionalLayer(
		size_t				pDepth,
		MapsData const&		pMapsData, 
		FiltersData const&	pFiltersData,
		bool				pAllocateOnGpu = true);

	virtual ~ConvolutionalLayer();

	
	size_t getLayerUnitSize()			const;
	size_t getLayerByteSize()			const;
	size_t getLayerAlignedByteSize()	const;
	size_t getLayerAlignedUnitSize()	const;


private:
	ConvolutionalLayer(ConvolutionalLayer<T> const& pLayer);
	ConvolutionalLayer<T>& operator=(ConvolutionalLayer<T> const& pLayer);


private:
	GpuBuffer<T>	mGpuBuffer;
	MapsData		mMapsData;
	FiltersData		mFiltersData;
	size_t			mDepth;
};


template <typename T>
ConvolutionalLayer<T>::ConvolutionalLayer(
	size_t				pDepth,
	MapsData const&		pMapsData, 
	FiltersData const&	pFiltersData,
	bool				pAllocateOnGpu = true)
:
	mDepth(pDepth),
	mMapsData(pMapsData),
	mFiltersData(pFiltersData)
{
	if(pAllocateOnGpu)
		mGpuBuffer.reallocateUnits(getLayerAlignedUnitSize() * mMapsData.count);
}


template <typename T>
ConvolutionalLayer<T>::~ConvolutionalLayer(){

}


template <typename T>
size_t ConvolutionalLayer<T>::getLayerUnitSize() const {
	return mDepth * mMapsData.width * mMapsData.height;
}


template <typename T>
size_t ConvolutionalLayer<T>::getLayerByteSize() const {
	return getLayerUnitSize() * sizeof(T);
}


template <typename T>
size_t ConvolutionalLayer<T>::getLayerAlignedByteSize()	const {
	return utils::align<size_t>(getLayerByteSize(), sizeof(T));
}


template <typename T>
size_t ConvolutionalLayer<T>::getLayerUnitSize() const {
	return utils::align<size_t>(getLayerByteSize(), sizeof(T));
}


	}
}