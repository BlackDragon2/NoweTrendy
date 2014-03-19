
#ifndef CNN_IMAGES_BATCH_H_
#define CNN_IMAGES_BATCH_H_


#include <string>
#include <vector>
#include <memory>

#include <opencv2/highgui/highgui.hpp>

#include "Utils/Utils.h"


namespace cnn {


template <typename T>
class ImagesBatch {
public:
	typedef std::shared_ptr<ImagesBatch<T>> PtrS;
	typedef std::unique_ptr<ImagesBatch<T>> PtrU;

	enum ImageType {
		UNCHANGED	= CV_LOAD_IMAGE_UNCHANGED,
		COLOR		= CV_LOAD_IMAGE_COLOR,
		GRAY		= CV_LOAD_IMAGE_GRAYSCALE
	};


public:
	static std::shared_ptr<ImagesBatch<T>> fromFiles(
		std::vector<std::string> const& pFiles,
		ImageType						pImageType = ImageType::COLOR);


public:
	ImagesBatch(
		size_t		pImageWidth, 
		size_t		pImageHeight, 
		size_t		pImageChannels, 
		ImageType	pImageType);
	virtual ~ImagesBatch();


	cv::Mat getImageAsMat(size_t pIndex);
	
	void allocateSpaceForImages(size_t pCount);

	void addImage(cv::Mat const& pMat);
	void addImageFromFile(std::string const& pPath);


	size_t getWidth()		const;
	size_t getHeight()		const;
	size_t getChannels()	const;

	size_t getImageByteSize()			const;
	size_t getImageUnitSize()			const;
	size_t getAlignedImageByteSize()	const;
	size_t getAlignedImageUnitSize()	const;

	T*			getImagesData();
	T const*	getImagesData() const;
	
	T*			getImageData(size_t pIndex);
	T const* 	getImageData(size_t pIndex) const;

	size_t getBatchUnitSize()	const;
	size_t getBatchByteSize()	const;
	size_t getImagesCount()		const;


private:
	void validateImage(cv::Mat const& pMat) const;

	ImagesBatch(ImagesBatch<T> const& pBatch);
	ImagesBatch<T>& operator=(ImagesBatch<T> const& pBatch);


private:
	std::shared_ptr<std::vector<T>>	mImagesData;
	size_t							mImagesCount;

	size_t mImageWidth;
	size_t mImageHeight;
	size_t mImageChannels;

	ImageType mImageType;
};


template <typename T>
std::shared_ptr<ImagesBatch<T>> ImagesBatch<T>::fromFiles(
	std::vector<std::string> const& pFiles,
	ImageType						pImageType)
{
	assert(pFiles.size() > 0UL);
	
	// setup metadata
	cv::Mat mat	= cv::imread(pFiles[0], pImageType);
	assert(mat.data);
	
	ImagesBatch<T>::PtrS batch(new ImagesBatch<T>(
		static_cast<size_t>(mat.size().width),
		static_cast<size_t>(mat.size().height),
		static_cast<size_t>(mat.channels()),
		pImageType));

	// alloc space
	batch->allocateSpaceForImages(pFiles.size());

	// copy images
	batch->addImage(mat);
	for(size_t i=1UL; i<pFiles.size(); ++i)
		batch->addImageFromFile(pFiles[i]);

	return batch;
}


template <typename T>
ImagesBatch<T>::ImagesBatch(
	size_t		pImageWidth, 
	size_t		pImageHeight, 
	size_t		pImageChannels, 
	ImageType	pImageType)
:
	mImageWidth(pImageWidth), 
	mImageHeight(pImageHeight),
	mImageChannels(pImageChannels),
	mImageType(pImageType),
	mImagesCount(0UL)
{
	mImagesData.reset(new std::vector<T>());
}


template <typename T>
ImagesBatch<T>::~ImagesBatch(){

}


template <typename T>
cv::Mat ImagesBatch<T>::getImageAsMat(size_t pIndex){
	return cv::Mat(mImageHeight, mImageWidth, CV_8UC(mImageChannels), getImageData(pIndex));
}


template <typename T>
void ImagesBatch<T>::allocateSpaceForImages(size_t pCount){
	mImagesData->resize((mImagesCount + pCount) * getAlignedImageUnitSize());
}


template <typename T>
void ImagesBatch<T>::addImage(cv::Mat const& pMat){
	validateImage(pMat);
	size_t imgUnitSize = getAlignedImageUnitSize();
	if(mImagesData->size() < (mImagesCount + 1) * imgUnitSize)
		allocateSpaceForImages(1UL);
	std::memcpy(mImagesData->data() + mImagesCount * imgUnitSize, pMat.data, utils::getCvMatBytesCount(pMat));
	++mImagesCount;
}


template <typename T>
void ImagesBatch<T>::addImageFromFile(std::string const& pPath){
	addImage(cv::imread(pPath, mImageType));
}


template <typename T>
size_t ImagesBatch<T>::getWidth() const {
	return mImageWidth;
}


template <typename T>
size_t ImagesBatch<T>::getHeight() const {
	return mImageHeight;
}


template <typename T>
size_t ImagesBatch<T>::getChannels() const {
	return mImageChannels;
}


template <typename T>
size_t ImagesBatch<T>::getImageByteSize() const {
	return mImageWidth * mImageHeight * mImageChannels;
}


template <typename T>
size_t ImagesBatch<T>::getImageUnitSize() const {
	return getImageByteSize() / sizeof(T);
}


template <typename T>
size_t ImagesBatch<T>::getAlignedImageByteSize() const {
	return utils::align<size_t>(getImageByteSize(), sizeof(T));
}


template <typename T>
size_t ImagesBatch<T>::getAlignedImageUnitSize() const {
	return getAlignedImageByteSize() / sizeof(T);
}


template <typename T>
T* ImagesBatch<T>::getImagesData(){
	return mImagesData->data();
}


template <typename T>
T const* ImagesBatch<T>::getImagesData() const {
	return mImagesData->data();
}


template <typename T>
T* ImagesBatch<T>::getImageData(size_t pIndex){
	assert(pIndex < mImagesCount);
	return mImagesData->data() + pIndex * getAlignedImageUnitSize();
}


template <typename T>
T const* ImagesBatch<T>::getImageData(size_t pIndex) const {
	assert(pIndex < mImagesCount);
	return mImagesData->data() + pIndex * getAlignedImageUnitSize();
}


template <typename T>
size_t ImagesBatch<T>::getBatchUnitSize() const {
	return mImagesData->size();
}


template <typename T>
size_t ImagesBatch<T>::getBatchByteSize() const {
	return mImagesData->size() * sizeof(T);
}


template <typename T>
size_t ImagesBatch<T>::getImagesCount() const {
	return mImagesCount;
}


template <typename T>
void ImagesBatch<T>::validateImage(cv::Mat const& pMat) const {
	assert(	pMat.data								&& 
			pMat.size().width	== mImageWidth		&& 
			pMat.size().height	== mImageHeight		&&
			pMat.channels()		== mImageChannels);
}



}


#endif	/* CNN_IMAGES_BATCH_H_ */