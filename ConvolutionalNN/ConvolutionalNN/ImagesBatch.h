
#pragma once

#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>


namespace cnn {


class ImagesBatch {
public:
	static ImagesBatch fromFiles(
		std::vector<std::string> const& pFiles,
		int								pCvReadFlag	= CV_LOAD_IMAGE_COLOR,
		size_t							pAlign		= 16UL);


public:
	virtual ~ImagesBatch();


	cv::Mat getImageAsMat(size_t pIndex);


	size_t getWidth()		const;
	size_t getHeight()		const;
	size_t getChannels()	const;
	size_t getImageSize()	const;

	uchar*			getImagesData();
	uchar const*	getImagesData() const;
	
	uchar*			getImage(size_t pIndex);
	uchar const* 	getImage(size_t pIndex) const;

	size_t getBatchSize()	const;
	size_t getImagesCount() const;


private:
	ImagesBatch();


private:
	std::vector<uchar>	mImages;
	size_t				mWidth;
	size_t				mHeight;
	size_t				mChannels;
	size_t				mImageSize;
	size_t				mImagesCount;
};


}