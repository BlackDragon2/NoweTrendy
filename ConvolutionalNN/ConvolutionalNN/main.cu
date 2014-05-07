
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>


#include "ImageBatch.h"

#include "GPU/Converter.cuh"
#include "GPU/GpuBuffer.cuh"
#include "GPU/ImageConvolution.cuh"
#include "GPU/VarianceCenterizer.cuh"
#include "GPU/Sharpener.cuh"
#include "GPU/MaxPooling.cuh"

#include "Types.h"
#include "Utils/FoldsFactory.h"


//#define MEASURE_SEPARATE


void doUchar(
	std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, 
	std::shared_ptr<cnn::ImageBatch<uchar>>& pKernels);


void doFloat(
	std::shared_ptr<cnn::ImageBatch<uchar>>& b, 
	std::shared_ptr<cnn::ImageBatch<uchar>>& filtersUchar);

int main()
{
	srand((uint32)time(0));
	
	__int64 freq;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	double spc = 1.0 / freq;

	/*
	std::string names[] = {
		"9336923", "9338535", "anpage", "asamma", "asewil",
		"astefa", "drbost", "ekavaz", "elduns", "kaknig", 
		"klclar", "ksunth", "lfso", "mbutle", "phughe", 
		"sbains", "slbirc", "vstros", "yfhsie"};
	*/
	std::string names[] = {"slbirc"};
	size_t nsize = ARRAYSIZE(names);

	std::vector<std::string> files;
	for(size_t a=0UL; a<1UL; ++a){
		for(size_t n=0UL; n<nsize; ++n){
			for(size_t i=1UL; i<=8; ++i){
				std::stringstream path;
				path << "data/" << names[n] << "/" << names[n] << "." << (i * 2) << ".jpg";
				files.push_back(path.str());
			}
		}
	}

	std::vector<std::string> filtersFiles;
	filtersFiles.push_back("data/test/none.png");
	filtersFiles.push_back("data/test/blur1.png");
	filtersFiles.push_back("data/test/sharp1.png");

	bool color = true;

	// Load images
	std::shared_ptr<cnn::ImageBatch<uchar>> b = cnn::ImageBatch<uchar>::fromFiles(files, color);
	std::shared_ptr<cnn::ImageBatch<uchar>> filtersUchar = cnn::ImageBatch<uchar>::fromFiles(filtersFiles, color);

	cudaDeviceProp prop0;
	cudaGetDeviceProperties(&prop0, 0);

	cudaSetDevice(cnn::config::Cuda::CUDA_DEVICE_ID);

	bool dof = false;
	if (dof){
		doFloat(b, filtersUchar);
	} else {
		doUchar(b, filtersUchar);
	}

    return 0;
}



void doUchar(
	std::shared_ptr<cnn::ImageBatch<uchar>>& pImages, 
	std::shared_ptr<cnn::ImageBatch<uchar>>& pKernels)
{
	{
		// space for uchars and floats
		cnn::gpu::GpuBuffer bImages;
		bImages.allocate(pImages->getBatchByteSize());
		bImages.writeToDevice(pImages->getBatchDataPtr(), pImages->getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// centering
		cnn::gpu::GpuBuffer bCenterImage;
		bCenterImage.allocate(pImages->getAlignedImageByteSize() * sizeof(float));

		cnn::gpu::AverageCenterizer<uchar> cent;
		cent.build(*pImages, bImages, bCenterImage);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cnn::gpu::GpuBuffer bCenterized;
		bCenterized.allocate(bImages.getByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cent.normalize(*pImages, bImages, bCenterImage, bCenterized);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// sharp
		cnn::gpu::Sharpener<uchar> shrp;
		shrp.build(*pImages, bCenterized, bCenterImage);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		shrp.normalize(*pImages, bCenterized, bCenterImage, bCenterized);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// convolution
		cnn::gpu::GpuBuffer bKernels;
		bKernels.allocate(pKernels->getBatchByteSize());
		bKernels.writeToDevice(pKernels->getBatchDataPtr(), pKernels->getBatchByteSize());

		cnn::ImageBatch<uchar> filtered(178, 198, pImages->getImageChannelsCount());
		filtered.allocateSpaceForImages(pImages->getImagesCount() * pKernels->getImagesCount(), true);

		cnn::gpu::GpuBuffer bFilteredBuffer;
		bFilteredBuffer.allocate(filtered.getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cnn::gpu::ImageConvolution<uchar> sc;
		sc.compute(*pImages, bCenterized, *pKernels, bKernels, filtered, bFilteredBuffer, 1, 1);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// load
		cnn::ImageBatch<uchar> centerized(pImages->getImageWidth(), pImages->getImageHeight(), pImages->getImageChannelsCount());
		centerized.allocateSpaceForImages(pImages->getImagesCount(), true);
		bCenterized.loadFromDevice(centerized.getBatchDataPtr(), centerized.getBatchByteSize());
		bFilteredBuffer.loadFromDevice(filtered.getBatchDataPtr(), filtered.getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// show
		cv::namedWindow("raw images");
		cv::namedWindow("centered images");
		cv::namedWindow("kerneled images");
		cv::namedWindow("kernels");

		cv::imshow("raw images", pImages->retriveAllImagesAsMat(5));
		cv::imshow("centered images", centerized.retriveAllImagesAsMat(5));
		cv::imshow("kerneled images", filtered.retriveAllImagesAsMat(pKernels->getImagesCount()));
		cv::imshow("kernels", pKernels->retriveAllImagesAsMat(pKernels->getImagesCount()));
	}
	cv::waitKey(0);
}


void doFloat(
	std::shared_ptr<cnn::ImageBatch<uchar>>& b, 
	std::shared_ptr<cnn::ImageBatch<uchar>>& filtersUchar)
{
	cnn::ImageBatch<float> fb(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount(), 32 * sizeof(float));
	fb.allocateSpaceForImages(b->getImagesCount(), true);

	{
		// space for uchars and floats
		cnn::gpu::GpuBuffer uchars;
		uchars.allocate(b->getBatchByteSize());
		uchars.writeToDevice(b->getBatchDataPtr(), b->getBatchByteSize());

		cnn::gpu::GpuBuffer floats;
		floats.allocate(b->getBatchByteSize() * sizeof(float));
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// convert
		cnn::gpu::Converter<uchar, float> converter;
		converter.convert(*b, uchars, floats);
		assert(cudaDeviceSynchronize() == cudaSuccess); 

		// convert filters
		uchars.writeToDevice(filtersUchar->getBatchDataPtr(), filtersUchar->getBatchByteSize());
		
		cnn::gpu::GpuBuffer kernels;
		kernels.allocate(filtersUchar->getBatchByteSize() * sizeof(float));
		converter.convert(*filtersUchar, uchars, kernels);
		assert(cudaDeviceSynchronize() == cudaSuccess);  

		// centering
		cnn::gpu::GpuBuffer centerImage;
		centerImage.allocate(b->getAlignedImageByteSize() * sizeof(float));

		cnn::gpu::AverageCenterizer<float> cent;
		cent.build(fb, floats, centerImage);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cnn::gpu::GpuBuffer centerized;
		centerized.allocate(floats.getByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cent.normalize(fb, floats, centerImage, centerized);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// sharp
		cnn::gpu::Sharpener<float> shrp;
		shrp.build(fb, centerized, centerImage);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		shrp.normalize(fb, centerized, centerImage, centerized);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// convolution
		cnn::ImageBatch<float> filtered(35, 39, b->getImageChannelsCount(), 32 * sizeof(float));
		filtered.allocateSpaceForImages(b->getImagesCount() * filtersUchar->getImagesCount(), true);

		cnn::gpu::GpuBuffer filteredBuffer;
		filteredBuffer.allocate(filtered.getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		
		cnn::ImageBatch<float> filters(filtersUchar->getImageWidth(), filtersUchar->getImageHeight(), filtersUchar->getImageChannelsCount(), 32 * sizeof(float));
		filters.allocateSpaceForImages(filtersUchar->getImagesCount(), true);

		cnn::gpu::ImageConvolution<float> sc;
		sc.compute(fb, centerized, filters, kernels, filtered, filteredBuffer, 5, 5);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// unconvert centerized
		cnn::gpu::Converter<float, uchar> converter2;
		converter2.convert(fb, centerized, uchars);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		uchars.loadFromDevice(b->getBatchDataPtr(), b->getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// unconvert filtered
		converter2.convert(filtered, filteredBuffer, uchars);
		assert(cudaDeviceSynchronize() == cudaSuccess);

		cnn::ImageBatch<uchar> filterResult(filtered.getImageWidth(), filtered.getImageHeight(), filtered.getImageChannelsCount());
		filterResult.allocateSpaceForImages(b->getImagesCount() * filters.getImagesCount(), true);
		uchars.loadFromDevice(filterResult.getBatchDataPtr(), filterResult.getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);

		// show
		cv::namedWindow("centered images");
		cv::namedWindow("kerneled images");
		cv::namedWindow("kernels");

		cv::imshow("centered images", b->retriveAllImagesAsMat(5));
		cv::imshow("kerneled images", filterResult.retriveAllImagesAsMat(filters.getImagesCount()));
		cv::imshow("kernels", filtersUchar->retriveAllImagesAsMat(filters.getImagesCount()));
	}
	cv::waitKey(0);
}