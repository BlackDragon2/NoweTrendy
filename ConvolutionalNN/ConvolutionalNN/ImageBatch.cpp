
#include "ImageBatch.h"


namespace cnn {


std::shared_ptr<ImageBatch> ImageBatch::fromFiles(
	std::vector<std::string> const& pFiles,
	ImageType						pImageType,
	size_t							pImageRowByteAlignment)
{
	
}


ImageBatch::ImageBatch(	
	size_t		pImageWidth,
	size_t		pImageHeight,
	ImageType	pImageType,
	size_t		pImageRowByteAlignment = 32UL)
:
	mImageWidth(pImageWidth),
	mImageHeight(pImageHeight),
	mImageType(pImageType),
	mImageRowByteAlignment(pImageRowByteAlignment)
{

}


ImageBatch::~ImageBatch(){

}


void ImageBatch::validateImage(cv::Mat const& pImage) const {
	assert(
		pImage.size().width		== mImageWidth,
		pImage.size().height	== mImageHeight,
		pImage.type()			== mImageType);	
}

	
void ImageBatch::allocateSpaceForImages(size_t pCount){
	mData->resize((mImagesCount + pCount) * getAlignmentImageByteSize());
}


void ImageBatch::copyMatToBatch(cv::Mat const& pImage, size_t pUnderIndex){
	validateImage(pImage);
	for (size_t i=0UL; i<mImageHeight; ++i)
		memcpy(getImageRowDataPtr(pUnderIndex, i), pImage.row(i).data, getImageRowByteSize()); 
}


void ImageBatch::addImage(cv::Mat const& pImage){
	if(getBatchCapacity() == getBatchSize())
		allocateSpaceForImages(std::sqrt(getBatchSize()) + 1UL);
	copyMatToBatch(pImage, mImagesCount);
	++mImagesCount;
}


void ImageBatch::copyFromBatchToMat(cv::Mat& pImage, size_t pFromIndex) const {
	validateImage(pImage);
	for (size_t i=0UL; i<mImageHeight; ++i)
		memcpy(pImage.row(i).data, getImageRowDataPtr(pFromIndex, i), getImageRowByteSize()); 
}


cv::Mat ImageBatch::retriveImageAsMat(size_t pImageIndex) const {
	cv::Mat mtx(mImageHeight, mImageWidth, mImageType);
	copyFromBatchToMat(mtx, pImageIndex);
	return mtx;
}


size_t ImageBatch::getImageWidth() const {
	return mImageWidth;
}


size_t ImageBatch::getImageHeight() const {
	return mImageHeight;
}


ImageBatch::ImageType ImageBatch::getImageType() const {
	return mImageType;
}


bool ImageBatch::isFloatType() const {
	return (mImageType & 0xF == 5);
}


bool ImageBatch::isByteType() const {
	return (mImageType & 0xF == 0);
}


bool ImageBatch::isGray() const {
	return getImageChannelsCount() == 1UL;
}


bool ImageBatch::isColor() const {
	return getImageChannelsCount() == 3UL;
}


int	ImageBatch::getImageChannelsCount() const {
	return static_cast<int>(mImageType / 8) + 1;
}


size_t ImageBatch::getImageRowByteSize() const {
	return mImageWidth * getImageChannelsCount() * (isFloatType() ? 4UL : 1UL);
}


size_t ImageBatch::getAlignmentImageRowByteSize() const {
	return utils::align(getImageRowByteSize(), mImageRowByteAlignment);
}


size_t ImageBatch::getImageByteSize() const {
	return getImageRowByteSize() * mImageHeight;
}


size_t ImageBatch::getAlignmentImageByteSize() const {
	return getAlignmentImageRowByteSize() * mImageHeight;
}


size_t ImageBatch::getBatchCapacity() const {
	return mData->size() / mImagesCount;
}


size_t ImageBatch::getBatchSize() const {
	return getAlignmentImageByteSize() * mImagesCount;
}


uchar* ImageBatch::getBatchDataPtr(){
	return mData->data();
}


uchar const* ImageBatch::getBatchDataPtr() const {
	return mData->data();
}


uchar* ImageBatch::getImageDataPtr(size_t pIndex){
	assert(pIndex < mImagesCount);
	return getBatchDataPtr() + pIndex * getAlignmentImageByteSize();
}


uchar const* ImageBatch::getImageDataPtr(size_t pIndex) const {
	assert(pIndex < mImagesCount);
	return getBatchDataPtr() + pIndex * getAlignmentImageByteSize();
}
	

uchar* ImageBatch::getImageRowDataPtr(size_t pIndex, size_t pRow){
	assert(pRow < mImageHeight);
	return getImageDataPtr(pIndex) + pRow * getAlignmentImageRowByteSize();
}
	

uchar const* ImageBatch::getImageRowDataPtr(size_t pIndex, size_t pRow) const {
	assert(pRow < mImageHeight);
	return getImageDataPtr(pIndex) + pRow * getAlignmentImageRowByteSize();
}


}