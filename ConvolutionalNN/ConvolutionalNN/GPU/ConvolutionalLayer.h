
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

	
	size_t getMapUnitSize()			const;
	size_t getMapByteSize()			const;
	size_t getMapAlignedByteSize()	const;
	size_t getMapAlignedUnitSize()	const;


private:
	ConvolutionalLayer(ConvolutionalLayer<T> const& pLayer);
	ConvolutionalLayer<T>& operator=(ConvolutionalLayer<T> const& pLayer);


private:
	GpuBuffer		mGpuBuffer;
	MapsData		mMapsData;
	FiltersData		mFiltersData;
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
		mGpuBuffer.allocateUnits(getMapAlignedUnitSize() * mMapsData.count);
}


template <typename T>
ConvolutionalLayer<T>::~ConvolutionalLayer(){

}


template <typename T>
size_t ConvolutionalLayer<T>::getMapUnitSize() const {
	return mDepth * mMapsData.width * mMapsData.height;
}


template <typename T>
size_t ConvolutionalLayer<T>::getMapByteSize() const {
	return getLayerUnitSize() * sizeof(T);
}


template <typename T>
size_t ConvolutionalLayer<T>::getMapAlignedByteSize()	const {
	return utils::align<size_t>(getLayerByteSize(), sizeof(T));
}


template <typename T>
size_t ConvolutionalLayer<T>::getMapUnitSize() const {
	return utils::align<size_t>(getLayerByteSize(), sizeof(T));
}


	}
}