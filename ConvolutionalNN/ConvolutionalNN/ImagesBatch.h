
#ifndef CNN_IMAGES_BATCH_H_
#define CNN_IMAGES_BATCH_H_


#include <string>
#include <vector>
#include <memory>

#include <opencv2/highgui/highgui.hpp>


namespace cnn {


class ImagesBatch {
public:
	typedef std::shared_ptr<ImagesBatch> PtrS;
	typedef std::unique_ptr<ImagesBatch> PtrU;

	enum ImageType {
		UNCHANGED	= CV_LOAD_IMAGE_UNCHANGED,
		GRAY		= CV_LOAD_IMAGE_GRAYSCALE,
		COLOR		= CV_LOAD_IMAGE_COLOR
	};


public:
	static std::shared_ptr<ImagesBatch> fromFiles(
		std::vector<std::string> const& pFiles,
		ImageType						pImageType		= ImageType::COLOR,
		size_t							pByteAlignment	= 32UL);


public:
	ImagesBatch(
		size_t		pImageWidth, 
		size_t		pImageHeight, 
		size_t		pImageChannels, 
		ImageType	pImageType,
		size_t		pByteAlignment = 32UL);
	virtual ~ImagesBatch();


	cv::Mat getImageAsMat(size_t pIndex, int pCvFormat);
	
	void allocateSpaceForImages(size_t pCount);
	
	void addImage(uchar const* pData);
	void addImage(cv::Mat const& pMat);
	void addImageFromFile(std::string const& pPath);


	size_t		getWidth()			const;
	size_t		getHeight()			const;
	size_t		getChannels()		const;
	ImageType	getImageType()		const;
	size_t		getByteAlignment()	const;

	size_t getImageByteSize()			const;
	size_t getAlignedImageByteSize()	const;

	uchar*			getImagesDataPtr();
	uchar const*	getImagesDataPtr() const;
	
	uchar*			getImageDataPtr(size_t pIndex);
	uchar const* 	getImageDataPtr(size_t pIndex) const;

	size_t getBatchByteSize()	const;
	size_t getImagesCount()		const;


private:
	void validateImage(cv::Mat const& pMat) const;

	ImagesBatch(ImagesBatch const& pBatch);
	ImagesBatch& operator=(ImagesBatch const& pBatch);


private:
	std::shared_ptr<std::vector<uchar>>	mImagesData;
	size_t								mImagesCount;
	size_t								mByteAlignment;

	size_t mImageWidth;
	size_t mImageHeight;
	size_t mImageChannels;

	ImageType mImageType;
};


}


#endif	/* CNN_IMAGES_BATCH_H_ */