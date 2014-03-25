
#ifndef IMAGE_BATCH_H_
#define IMAGE_BATCH_H_

#include <memory>
#include <vector>
#include <string>

#include "Utils/Utils.h"
#include "Types.h"


namespace cnn {


class ImageBatch {
public:
	enum ImageType {
		BYTE_GRAY	= CV_8UC1,
		BYTE_BGR	= CV_8UC3,
		FLOAT_GRAY	= CV_32FC1,
		FLOAT_BGR	= CV_32FC3
	};


public:
	static std::shared_ptr<ImageBatch> fromFiles(
		std::vector<std::string> const& pFiles,
		ImageType						pImageType,
		size_t							pImageRowByteAlignment = 32UL);


public:
	ImageBatch(	
		size_t		pImageWidth,
		size_t		pImageHeight,
		ImageType	pImageType,
		size_t		pImageRowByteAlignment = 32UL);
	virtual ~ImageBatch();
	

	void validateImage(cv::Mat const& pImage) const;

	void allocateSpaceForImages(size_t pCount);
	
	void copyMatToBatch(cv::Mat const& pImage, size_t pUnderIndex);
	void addImage(cv::Mat const& pImage);

	void	copyFromBatchToMat(cv::Mat& pImage, size_t pFromIndex)	const;
	cv::Mat retriveImageAsMat(size_t pImageIndex)					const;


	size_t getImageWidth()	const;
	size_t getImageHeight() const;

	ImageType	getImageType()	const;
	bool		isFloatType()	const;
	bool		isByteType()	const;
	bool		isGray()		const;
	bool		isColor()		const;

	int	getImageChannelsCount() const;

	size_t getImageRowByteSize()			const;
	size_t getAlignmentImageRowByteSize()	const;
	size_t getImageByteSize()				const;
	size_t getAlignmentImageByteSize()		const;

	size_t getBatchCapacity()	const;
	size_t getBatchSize()		const;

	uchar*			getBatchDataPtr();
	uchar const*	getBatchDataPtr() const;

	uchar*			getImageDataPtr(size_t pIndex);
	uchar const*	getImageDataPtr(size_t pIndex) const;
	
	uchar*			getImageRowDataPtr(size_t pIndex, size_t pRow);
	uchar const*	getImageRowDataPtr(size_t pIndex, size_t pRow) const;

	template <typename T> T*		getBatchDataPtrAs();
	template <typename T> T const*	getBatchDataPtrAs() const;
	
	template <typename T> T*		getImageDataPtrAs(size_t pIndex);
	template <typename T> T const*	getImageDataPtrAs(size_t pIndex) const;
	
	template <typename T> T*		getImageRowDataPtrAs(size_t pIndex, size_t pRow);
	template <typename T> T const*	getImageRowDataPtrAs(size_t pIndex, size_t pRow) const;


private:
	std::shared_ptr<std::vector<uchar>> mData;
	size_t								mImagesCount;
	ImageType							mImageType;

	size_t mImageWidth;
	size_t mImageHeight;
	size_t mImageRowByteAlignment;
};


template <typename T> 
T* ImageBatch::getBatchDataPtrAs(){
	return reinterpret_cast<T*>(getBatchDataPtr());
}


template <typename T> 
T const* ImageBatch::getBatchDataPtrAs() const {
	return reinterpret_cast<T*>(getBatchDataPtr());
}


template <typename T> 
T* ImageBatch::getImageDataPtrAs(size_t pIndex){
	return reinterpret_cast<T*>(getImageDataPtr(pIndex));
}


template <typename T>
T const* ImageBatch::getImageDataPtrAs(size_t pIndex) const {
	return reinterpret_cast<T*>(getImageDataPtr(pIndex));
}
	

template <typename T> 
T* ImageBatch::getImageRowDataPtrAs(size_t pIndex, size_t pRow){
	return reinterpret_cast<T*>(getImageRowDataPtr(pIndex, pRow));
}


template <typename T> 
T const* ImageBatch::getImageRowDataPtrAs(size_t pIndex, size_t pRow) const {
	return reinterpret_cast<T*>(getImageRowDataPtr(pIndex, pRow));
}


}


#endif	/* IMAGE_BATCH_H_ */