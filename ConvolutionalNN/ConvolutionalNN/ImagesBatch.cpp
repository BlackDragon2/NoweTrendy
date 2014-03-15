
#include <assert.h>

#include "ImagesBatch.h"
#include "Utils/Utils.h"


namespace cnn {


ImagesBatch ImagesBatch::fromFiles(
	std::vector<std::string> const& pFiles,
	int								pCvReadFlag,
	size_t							pAlign)
{
	ImagesBatch batch;
	if (pFiles.size() == 0UL)
		return batch;
	
	// setup metadata
	size_t matImageSize;
	{
		cv::Mat mat			= cv::imread(pFiles[0], pCvReadFlag);
		assert(mat.data);

		batch.mWidth		= static_cast<size_t>(mat.size().width);
		batch.mHeight		= static_cast<size_t>(mat.size().height);
		batch.mChannels		= static_cast<size_t>(mat.channels());

		matImageSize		= static_cast<size_t>(mat.size().area() * mat.channels() * sizeof(uchar));
		batch.mImageSize	= utils::align<size_t>(matImageSize, pAlign);

		batch.mImages.resize(batch.mImageSize * pFiles.size());
		std::memcpy(batch.mImages.data(), mat.data, matImageSize);

		batch.mImagesCount = pFiles.size();
	}

	// copy images
	for(size_t i=1UL; i<pFiles.size(); ++i){
		cv::Mat mat	= cv::imread(pFiles[i], pCvReadFlag);
		assert(mat.data && 
			mat.size().width == batch.mWidth && 
			mat.size().height == batch.mHeight);
		std::memcpy(batch.mImages.data() + i * batch.mImageSize, mat.data, matImageSize);
	}

	return batch;
}


ImagesBatch::~ImagesBatch(){

}


cv::Mat ImagesBatch::getImageAsMat(size_t pIndex){
	return cv::Mat(mHeight, mWidth, CV_8UC(mChannels), getImage(pIndex));
}


size_t ImagesBatch::getWidth() const {
	return mWidth;
}


size_t ImagesBatch::getHeight() const {
	return mHeight;
}


size_t ImagesBatch::getChannels() const {
	return mChannels;
}


size_t ImagesBatch::getImageSize() const {
	return mImageSize;
}


uchar* ImagesBatch::getImagesData(){
	return mImages.data();
}


uchar const* ImagesBatch::getImagesData() const {
	return mImages.data();
}


uchar* ImagesBatch::getImage(size_t pIndex){
	return mImages.data() + pIndex * mImageSize;
}


uchar const* ImagesBatch::getImage(size_t pIndex) const {
	return mImages.data() + pIndex * mImageSize;
}


size_t ImagesBatch::getBatchSize() const {
	return mImages.size();
}


size_t ImagesBatch::getImagesCount() const {
	return mImagesCount;
}


ImagesBatch::ImagesBatch(){

}


}