
#include "ImagesBatch.h"

#include "Utils/Utils.h"


namespace cnn {


std::shared_ptr<ImagesBatch> ImagesBatch::fromFiles(
	std::vector<std::string> const& pFiles,
	ImageType						pImageType,
	size_t							pByteAlignment)
{
	assert(pFiles.size() > 0UL);
	
	// setup metadata
	cv::Mat mat	= cv::imread(pFiles[0], pImageType);
	assert(mat.data != nullptr);
	
	ImagesBatch::PtrS batch(new ImagesBatch(
		static_cast<size_t>(mat.size().width),
		static_cast<size_t>(mat.size().height),
		static_cast<size_t>(mat.channels()),
		pImageType,
		pByteAlignment));

	// alloc space
	batch->allocateSpaceForImages(pFiles.size());

	// copy images
	batch->addImage(mat);
	for(size_t i=1UL; i<pFiles.size(); ++i)
		batch->addImageFromFile(pFiles[i]);

	return batch;
}


ImagesBatch::ImagesBatch(
	size_t		pImageWidth, 
	size_t		pImageHeight, 
	size_t		pImageChannels, 
	ImageType	pImageType,
	size_t		pByteAlignment)
:
	mByteAlignment(pByteAlignment),
	mImageWidth(pImageWidth), 
	mImageHeight(pImageHeight),
	mImageChannels(pImageChannels),
	mImageType(pImageType),
	mImagesCount(0UL)
{
	mImagesData.reset(new std::vector<uchar>());
}


ImagesBatch::~ImagesBatch(){

}


cv::Mat ImagesBatch::getImageAsMat(size_t pIndex){
	return cv::Mat(mImageHeight, mImageWidth, CV_8UC(mImageChannels), getImageDataPtr(pIndex));
}


void ImagesBatch::allocateSpaceForImages(size_t pCount){
	mImagesData->resize((mImagesCount + pCount) * getAlignedImageByteSize());
}


void ImagesBatch::addImage(uchar const* pData){
	size_t imgUnitSize = getAlignedImageByteSize();
	if(mImagesData->size() < (mImagesCount + 1) * imgUnitSize)
		allocateSpaceForImages(static_cast<size_t>(sqrt(mImagesCount)) + 1UL);
	std::memcpy(mImagesData->data() + mImagesCount * imgUnitSize, pData, getImageByteSize());
	++mImagesCount;
}


void ImagesBatch::addImage(cv::Mat const& pMat){
	validateImage(pMat);
	addImage(pMat.data);
}


void ImagesBatch::addImageFromFile(std::string const& pPath){
	addImage(cv::imread(pPath, mImageType));
}


size_t ImagesBatch::getWidth() const {
	return mImageWidth;
}


size_t ImagesBatch::getHeight() const {
	return mImageHeight;
}


size_t ImagesBatch::getChannels() const {
	return mImageChannels;
}

	
ImagesBatch::ImageType ImagesBatch::getImageType() const {
	return mImageType;
}


size_t ImagesBatch::getByteAlignment() const {
	return mByteAlignment;
}


size_t ImagesBatch::getImageByteSize() const {
	return mImageWidth * mImageHeight * mImageChannels;
}


size_t ImagesBatch::getAlignedImageByteSize() const {
	return utils::align(getImageByteSize(), mByteAlignment);
}


uchar* ImagesBatch::getImagesDataPtr(){
	return mImagesData->data();
}


uchar const* ImagesBatch::getImagesDataPtr() const {
	return mImagesData->data();
}


uchar* ImagesBatch::getImageDataPtr(size_t pIndex){
	assert(pIndex < mImagesCount);
	return mImagesData->data() + pIndex * getAlignedImageByteSize();
}


uchar const* ImagesBatch::getImageDataPtr(size_t pIndex) const {
	assert(pIndex < mImagesCount);
	return mImagesData->data() + pIndex * getAlignedImageByteSize();
}


size_t ImagesBatch::getBatchByteSize() const {
	return mImagesData->size() * sizeof(uchar);
}


size_t ImagesBatch::getImagesCount() const {
	return mImagesCount;
}


void ImagesBatch::validateImage(cv::Mat const& pMat) const {
	assert(	pMat.data								&& 
			pMat.size().width	== mImageWidth		&& 
			pMat.size().height	== mImageHeight		&&
			pMat.channels()		== mImageChannels);
}


}