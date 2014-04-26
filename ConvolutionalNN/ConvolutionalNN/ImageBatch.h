
#ifndef CNN_IMAGE_BATCH_H_
#define CNN_IMAGE_BATCH_H_

#include <memory>
#include <vector>
#include <string>
#include <assert.h>
#include <limits>

#include "Utils/Utils.h"
#include "Utils/CvUtils.h"


namespace cnn {



template <typename T>
class ImageBatch {
public:
	static std::shared_ptr<ImageBatch<uchar>> fromFiles(
		std::vector<std::string> const& pFiles,
		bool							pLoadInColor			= true,
		size_t							pImageRowByteAlignment	= 32UL);


public:
	ImageBatch(	
		size_t		pImageWidth,
		size_t		pImageHeight,
		size_t		pImageChannels,
		size_t		pImageRowByteAlignment = 32UL);
	virtual ~ImageBatch();


	void validateImage(cv::Mat const& pImage) const;

	void allocateSpaceForImages(size_t pCount, bool pUpdateImageCounter = false);
	
	void copyMatToBatch(cv::Mat const& pImage, size_t pUnderIndex);
	void addImage(cv::Mat const& pImage);

	void	copyFromBatchToMat(cv::Mat& pImage, size_t pFromIndex)	const;
	cv::Mat retriveImageAsMat(size_t pImageIndex)					const;
	cv::Mat retriveAllImagesAsMat(size_t pImagesPerRow)				const;
	

	size_t getImageWidth()	const; 
	size_t getImageHeight()	const;

	bool isGray()	const;
	bool isColor()	const;

	size_t getImageChannelsCount() const;

	size_t getImageRowByteAligment() const;

	size_t getImageRowByteSize()		const;
	size_t getAlignedImageRowByteSize()	const;
	size_t getImageByteSize()			const;
	size_t getAlignedImageByteSize()	const;
	
	size_t getBatchByteCapacity()	const;
	size_t getBatchByteSize()		const;
	size_t getImagesCapacity()		const;
	size_t getImagesCount()			const;

	T*			getBatchDataPtr();
	T const*	getBatchDataPtr() const;

	T*			getImageDataPtr(size_t pIndex);
	T const*	getImageDataPtr(size_t pIndex) const;
	
	T*			getImageRowDataPtr(size_t pIndex, size_t pRow);
	T const*	getImageRowDataPtr(size_t pIndex, size_t pRow) const;

	template <typename R> R*		getBatchDataPtrAs();
	template <typename R> R const*	getBatchDataPtrAs() const;
	
	template <typename R> R*		getImageDataPtrAs(size_t pIndex);
	template <typename R> R const*	getImageDataPtrAs(size_t pIndex) const;
	
	template <typename R> R*		getImageRowDataPtrAs(size_t pIndex, size_t pRow);
	template <typename R> R const*	getImageRowDataPtrAs(size_t pIndex, size_t pRow) const;


private:
	std::shared_ptr<std::vector<T>> mData;
	size_t							mImagesCount;

	size_t mImageChannels;
	size_t mImageWidth;
	size_t mImageHeight;
	size_t mImageRowByteAlignment;
};


template <typename T>
std::shared_ptr<ImageBatch<uchar>> ImageBatch<T>::fromFiles(
	std::vector<std::string> const& pFiles,
	bool							pLoadInColor,
	size_t							pImageRowByteAlignment)
{
	assert(pFiles.size() > 0UL);
	
	int readType	= pLoadInColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;	
	cv::Mat mat		= cv::imread(pFiles[0], readType);

	assert(mat.data != nullptr);
	std::shared_ptr<ImageBatch<uchar>> batch(new ImageBatch<uchar>(
		mat.size().width, 
		mat.size().height, 
		mat.channels(), 
		pImageRowByteAlignment));

	batch->allocateSpaceForImages(pFiles.size());

	batch->addImage(mat);
	for(size_t i=1UL; i<pFiles.size(); ++i)
		batch->addImage(cv::imread(pFiles[i], readType));

	return batch;
}


template <typename T>
ImageBatch<T>::ImageBatch(	
	size_t		pImageWidth,
	size_t		pImageHeight,
	size_t		pImageChannels,
	size_t		pImageRowByteAlignment)
:
	mImageWidth(pImageWidth),
	mImageHeight(pImageHeight),
	mImageChannels(pImageChannels),
	mImageRowByteAlignment(pImageRowByteAlignment),
	mImagesCount(0UL),
	mData(new std::vector<T>(0))
{

}


template <typename T>
ImageBatch<T>::~ImageBatch(){

}


template <typename T>
void ImageBatch<T>::validateImage(cv::Mat const& pImage) const {
	assert(
		pImage.size().width		== mImageWidth		&&
		pImage.size().height	== mImageHeight		&&
		pImage.channels()		== mImageChannels	&&
		pImage.elemSize1()		== sizeof(T));	
}

	
template <typename T>
void ImageBatch<T>::allocateSpaceForImages(size_t pCount, bool pUpdateImageCounter){
	mData->resize((mImagesCount + pCount) * getAlignedImageByteSize() / sizeof(T));
	if(pUpdateImageCounter)
		mImagesCount += pCount;
}


template <typename T>
void ImageBatch<T>::copyMatToBatch(cv::Mat const& pImage, size_t pUnderIndex){
	validateImage(pImage);
	size_t bytes = getImageRowByteSize();
	for (size_t i=0UL; i<mImageHeight; ++i)
		memcpy(getImageRowDataPtr(pUnderIndex, i), pImage.row(i).data, bytes); 
}


template <typename T>
void ImageBatch<T>::addImage(cv::Mat const& pImage){
	if(getImagesCapacity() == getImagesCount())
		allocateSpaceForImages(static_cast<size_t>(std::sqrt(getImagesCount())) + 1UL);
	++mImagesCount;
	copyMatToBatch(pImage, mImagesCount - 1UL);
}


template <typename T>
void ImageBatch<T>::copyFromBatchToMat(cv::Mat& pImage, size_t pFromIndex) const {
	validateImage(pImage);
	size_t bytes = getImageRowByteSize();
	for (size_t i=0UL; i<mImageHeight; ++i){ 
		memcpy(pImage.row(i).data, getImageRowDataPtr(pFromIndex, i), bytes); 
	}
}


template <typename T>
cv::Mat ImageBatch<T>::retriveImageAsMat(size_t pImageIndex) const {
	cv::Mat mtx(mImageHeight, mImageWidth, utils::createCvImageType<T>(mImageChannels));
	copyFromBatchToMat(mtx, pImageIndex);
	return mtx;
}


template <typename T>
cv::Mat ImageBatch<T>::retriveAllImagesAsMat(size_t pImagesPerRow) const {
	size_t width	= mImageWidth * pImagesPerRow;
	size_t height	= mImageHeight * static_cast<size_t>(std::ceil(
		static_cast<float>(mImagesCount) / pImagesPerRow));
	cv::Mat mtx(height, width, utils::createCvImageType<T>(mImageChannels));
	for(size_t i=0UL; i<mImagesCount; ++i){
		cv::Rect roi((i % pImagesPerRow) * mImageWidth, (i / pImagesPerRow) * mImageHeight, mImageWidth, mImageHeight);
		cv::Mat part = cv::Mat(mtx, roi);
		copyFromBatchToMat(part, i);
	}
	return mtx;
}


template <typename T>
size_t ImageBatch<T>::ImageBatch::getImageWidth() const {
	return mImageWidth;
}


template <typename T>
size_t ImageBatch<T>::ImageBatch::getImageHeight() const {
	return mImageHeight;
}


template <typename T>
bool ImageBatch<T>::isGray() const {
	return getImageChannelsCount() == 1UL;
}


template <typename T>
bool ImageBatch<T>::isColor() const {
	return getImageChannelsCount() == 3UL;
}


template <typename T>
size_t ImageBatch<T>::getImageChannelsCount() const {
	return mImageChannels;
}


template <typename T>
size_t ImageBatch<T>::getImageRowByteAligment() const {
	return mImageRowByteAlignment;
}


template <typename T>
size_t ImageBatch<T>::getImageRowByteSize() const {
	return mImageWidth * getImageChannelsCount() * sizeof(T);
}


template <typename T>
size_t ImageBatch<T>::getAlignedImageRowByteSize() const {
	return utils::align(getImageRowByteSize(), mImageRowByteAlignment);
}


template <typename T>
size_t ImageBatch<T>::getImageByteSize() const {
	return getImageRowByteSize() * mImageHeight;
}


template <typename T>
size_t ImageBatch<T>::getAlignedImageByteSize() const {
	return getAlignedImageRowByteSize() * mImageHeight;
}


template <typename T>
size_t ImageBatch<T>::getBatchByteCapacity() const {
	return mData->size() * sizeof(T);
}


template <typename T>
size_t ImageBatch<T>::getBatchByteSize() const {
	return getAlignedImageByteSize() * mImagesCount;
}


template <typename T>
size_t ImageBatch<T>::getImagesCapacity() const {
	return getBatchByteCapacity() / getAlignedImageByteSize();
}


template <typename T>
size_t ImageBatch<T>::getImagesCount() const {
	return mImagesCount;
}


template <typename T>
T* ImageBatch<T>::getBatchDataPtr(){
	return mData->data();
}


template <typename T>
T const* ImageBatch<T>::getBatchDataPtr() const {
	return mData->data();
}


template <typename T>
T* ImageBatch<T>::getImageDataPtr(size_t pIndex){
	assert(pIndex < getImagesCount());
	return getBatchDataPtr() + pIndex * getAlignedImageByteSize();
}


template <typename T>
T const* ImageBatch<T>::getImageDataPtr(size_t pIndex) const {
	assert(pIndex < getImagesCount());
	return getBatchDataPtr() + pIndex * getAlignedImageByteSize();
}
	

template <typename T>
T* ImageBatch<T>::getImageRowDataPtr(size_t pIndex, size_t pRow){
	assert(pRow < mImageHeight);
	return getImageDataPtr(pIndex) + pRow * getAlignedImageRowByteSize();
}
	

template <typename T>
T const* ImageBatch<T>::getImageRowDataPtr(size_t pIndex, size_t pRow) const {
	assert(pRow < mImageHeight);
	return getImageDataPtr(pIndex) + pRow * getAlignedImageRowByteSize();
}


template <typename T> template<typename R> 
R* ImageBatch<T>::getBatchDataPtrAs(){
	return reinterpret_cast<R*>(getBatchDataPtr());
}


template <typename T> template<typename R>
R const* ImageBatch<T>::getBatchDataPtrAs() const {
	return reinterpret_cast<R*>(getBatchDataPtr());
}


template <typename T> template<typename R>
R* ImageBatch<T>::getImageDataPtrAs(size_t pIndex){
	return reinterpret_cast<R*>(getImageDataPtr(pIndex));
}


template <typename T> template<typename R>
R const* ImageBatch<T>::getImageDataPtrAs(size_t pIndex) const {
	return reinterpret_cast<R*>(getImageDataPtr(pIndex));
}
	

template <typename T> template<typename R>
R* ImageBatch<T>::getImageRowDataPtrAs(size_t pIndex, size_t pRow){
	return reinterpret_cast<R*>(getImageRowDataPtr(pIndex, pRow));
}


template <typename T> template<typename R>
R const* ImageBatch<T>::getImageRowDataPtrAs(size_t pIndex, size_t pRow) const {
	return reinterpret_cast<R*>(getImageRowDataPtr(pIndex, pRow));
}


}


#endif	/* CNN_IMAGE_BATCH_H_ */