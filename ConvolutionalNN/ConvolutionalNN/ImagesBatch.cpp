
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
		cv::Mat mat	= cv::imread(pFiles[0], pCvReadFlag);
		assert(mat.data);

		batch.mImageWidth		= static_cast<size_t>(mat.size().width);
		batch.mImageHeight		= static_cast<size_t>(mat.size().height);
		batch.mImageChannels	= static_cast<size_t>(mat.channels());

		matImageSize		= static_cast<size_t>(mat.size().area() * mat.channels() * sizeof(uchar));
		batch.mImageSize	= utils::align<size_t>(matImageSize, pAlign);

		batch.mImagesData.reset(new std::vector<uchar>(batch.mImageSize * pFiles.size()));
		std::memcpy(batch.mImagesData->data(), mat.data, matImageSize);

		batch.mImagesCount = pFiles.size();
	}

	// copy images
	for(size_t i=1UL; i<pFiles.size(); ++i){
		cv::Mat mat	= cv::imread(pFiles[i], pCvReadFlag);
		assert(mat.data && 
			mat.size().width == batch.mImageWidth && 
			mat.size().height == batch.mImageHeight);
		std::memcpy(batch.mImagesData->data() + i * batch.mImageSize, mat.data, matImageSize);
	}

	return batch;
}


ImagesBatch::~ImagesBatch(){

}


cv::Mat ImagesBatch::getImageAsMat(size_t pIndex){
	return cv::Mat(mImageHeight, mImageWidth, CV_8UC(mImageChannels), getImage(pIndex));
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


size_t ImagesBatch::getImageSize() const {
	return mImageSize;
}


uchar* ImagesBatch::getImagesData(){
	return mImagesData->data();
}


uchar const* ImagesBatch::getImagesData() const {
	return mImagesData->data();
}


uchar* ImagesBatch::getImage(size_t pIndex){
	return mImagesData->data() + pIndex * mImageSize;
}


uchar const* ImagesBatch::getImage(size_t pIndex) const {
	return mImagesData->data() + pIndex * mImageSize;
}


size_t ImagesBatch::getBatchSize() const {
	return mImagesData->size();
}


size_t ImagesBatch::getImagesCount() const {
	return mImagesCount;
}


ImagesBatch::ImagesBatch(){

}


}