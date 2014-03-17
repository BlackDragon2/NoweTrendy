
#include <assert.h>

#include "ImagesBatch.h"
#include "Utils/Utils.h"


namespace cnn {


ImagesBatch ImagesBatch::fromFiles(
	std::vector<std::string> const& pFiles,
	int								pCvReadFlag,
	size_t							pAlign)
{
	assert(pFiles.size() > 0UL);
	
	// setup metadata
	cv::Mat mat	= cv::imread(pFiles[0], pCvReadFlag);
	assert(mat.data);
	
	ImagesBatch batch(
		static_cast<size_t>(mat.size().width),
		static_cast<size_t>(mat.size().height),
		static_cast<size_t>(mat.channels()),
		pAlign,
		pCvReadFlag);

	// alloc space
	batch.allocateSpaceForImages(pFiles.size());

	// copy images
	batch.addImage(mat);
	for(size_t i=1UL; i<pFiles.size(); ++i)
		batch.addImageFromFile(pFiles[i]);

	return batch;
}


ImagesBatch::ImagesBatch(
	size_t	pImageWidth, 
	size_t	pImageHeight, 
	size_t	pImageChannels, 
	int		pAlignBytes, 
	size_t	pCvReadFlags)
:
	mImageWidth(pImageWidth), 
	mImageHeight(pImageHeight),
	mImageChannels(pImageChannels),
	mAlignBytes(pAlignBytes),
	mCvReadFlags(pCvReadFlags),
	mImagesCount(0)
{
	mImageByteSize = utils::align<size_t>(pImageWidth * pImageHeight * pImageChannels, pAlignBytes);
	mImagesData.reset(new std::vector<uchar>());
}


ImagesBatch::~ImagesBatch(){

}


cv::Mat ImagesBatch::getImageAsMat(size_t pIndex){
	return cv::Mat(mImageHeight, mImageWidth, CV_8UC(mImageChannels), getImage(pIndex));
}


void ImagesBatch::allocateSpaceForImages(size_t pCount){
	mImagesData->resize((mImagesCount + pCount) * mImageByteSize);
}


void ImagesBatch::addImage(cv::Mat const& pMat){
	validateImage(pMat);
	if(mImagesData->size() < (mImagesCount + 1) * mImageByteSize)
		allocateSpaceForImages(1UL);
	std::memcpy(mImagesData->data() + mImagesCount * mImageByteSize, pMat.data, utils::getCvMatBytesCount(pMat));
	++mImagesCount;
}


void ImagesBatch::addImageFromFile(std::string const& pPath){
	addImage(cv::imread(pPath, mCvReadFlags));
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


size_t ImagesBatch::getImageByteSize() const {
	return mImageByteSize;
}


uchar* ImagesBatch::getImagesData(){
	return mImagesData->data();
}


uchar const* ImagesBatch::getImagesData() const {
	return mImagesData->data();
}


uchar* ImagesBatch::getImage(size_t pIndex){
	return mImagesData->data() + pIndex * mImageByteSize;
}


uchar const* ImagesBatch::getImage(size_t pIndex) const {
	return mImagesData->data() + pIndex * mImageByteSize;
}


size_t ImagesBatch::getBatchSize() const {
	return mImagesData->size();
}


size_t ImagesBatch::getImagesCount() const {
	return mImagesCount;
}


void ImagesBatch::validateImage(cv::Mat const& pMat) const {
	assert(pMat.data && 
			pMat.size().width == mImageWidth && 
			pMat.size().height == mImageHeight);
}


}