
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "ImageBatch.h"

#include "GPU/Tasks.cuh"
#include "GPU/GpuBuffer.cuh"

#include "Types.h"
#include "Utils/FoldsFactory.h"


int main()
{
	srand((uint32)time(0));

	std::shared_ptr<std::vector<size_t>> folds = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::DEFAULT);
	std::shared_ptr<std::vector<size_t>> folds2 = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::CUT);
	std::shared_ptr<std::vector<size_t>> folds3 = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::EXTEND_WITH_COPIES);

	std::vector<std::string> files;
	for(size_t a=0UL; a<10UL; ++a){
		for(size_t i=1UL; i<=20; ++i){
			std::stringstream path;
			path << "data/phughe/phughe." << i << ".jpg";
			files.push_back(path.str());
		}
	}

	__int64 freq, s1, s12, s2, e1, e2, pre1;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	double spc = 1.0 / freq;

	std::shared_ptr<cnn::ImageBatch<uchar>> b = cnn::ImageBatch<uchar>::fromFiles(files);

	cnn::ImageBatch<uchar> center(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	center.allocateSpaceForImages(1, true);
	
	cnn::ImageBatch<uchar> centerized(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	centerized.allocateSpaceForImages(b->getImagesCount(), true);
	
	cnn::ImageBatch<uchar> eroded(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	eroded.allocateSpaceForImages(b->getImagesCount(), true);
	{
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&pre1));
		std::vector<std::pair<uchar, uchar> > res = b->findImagesColorsBoundaries();

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s1));

		cnn::gpu::GpuBuffer devbuffer(b->getBatchByteSize());
		cnn::gpu::GpuBuffer devbuffersingle(b->getAlignedImageByteSize());
		cnn::gpu::GpuBuffer devbuffer2(b->getBatchByteSize());
		cnn::gpu::GpuBuffer devbound(b->getImagesCount() * b->getImageChannelsCount() * 2);
		cnn::gpu::GpuBuffer devbuffer3(b->getBatchByteSize());

		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s12));
		
		devbuffer.writeToDevice(b->getBatchDataPtr(), b->getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s2));
		
		cnn::gpu::Tasks::buildCenterMap<uchar>(*b, devbuffer, devbuffersingle);
		cnn::gpu::Tasks::centerizeWithMap<uchar>(*b, devbuffer, devbuffersingle, devbuffer2);
		cnn::gpu::Tasks::findEachImageBoundaries<uchar>(*b, devbuffer2, devbound);
		cnn::gpu::Tasks::erodeEachImageUsingBoundaries<uchar>(*b, devbuffer2, devbound, devbuffer3, 255);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e1));
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		devbuffersingle.loadFromDevice(center.getBatchDataPtr(), center.getAlignedImageByteSize());

		devbuffer2.loadFromDevice(centerized.getBatchDataPtr(), b->getBatchByteSize());
		devbuffer3.loadFromDevice(eroded.getBatchDataPtr(), b->getBatchByteSize());

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e2));
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		std::cout << "find            "	<< double(s1 - pre1) * spc << std::endl;
		std::cout << "allocation:     " << double(s12 - s1) * spc << std::endl;
		std::cout << "send:           " << double(s2 - s12) * spc << std::endl;
		std::cout << "comp:           " << double(e1 - s2) * spc << std::endl;
		std::cout << "recv & dealloc: " << double(e2 - e1) * spc << std::endl;
		std::cout << "all:            " << double(e2 - pre1) * spc << std::endl;
		
		cv::namedWindow("some name1", CV_WINDOW_AUTOSIZE);
		cv::namedWindow("some name2", CV_WINDOW_AUTOSIZE);
		cv::namedWindow("some name3", CV_WINDOW_AUTOSIZE);
		cv::namedWindow("some name4", CV_WINDOW_AUTOSIZE);

		cv::imshow("some name1", b->retriveImageAsMat(10));
		cv::imshow("some name2", center.retriveImageAsMat(0));
		cv::imshow("some name3", centerized.retriveImageAsMat(10));
		cv::imshow("some name4", eroded.retriveImageAsMat(10));
	}
	cv::waitKey(0);

    return 0;
}


/*
int main()
{
	srand((uint)time(0));

	std::shared_ptr<std::vector<size_t>> folds = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::DEFAULT);
	std::shared_ptr<std::vector<size_t>> folds2 = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::CUT);
	std::shared_ptr<std::vector<size_t>> folds3 = cnn::utils::FoldsFactory::prepareFoldVector(117, 7, cnn::utils::FoldsFactory::FitTactic::EXTEND_WITH_COPIES);

	std::vector<std::string> files;
	//for(size_t a=0UL; a<10UL; ++a){
		for(size_t i=1UL; i<=20; ++i){
			std::stringstream path;
			path << "data/phughe/phughe." << i << ".jpg";
			files.push_back(path.str());
		}
	//}

	__int64 freq, s1, s12, s2, e1, e2;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	double spc = 1.0 / freq;

	std::shared_ptr<cnn::ImagesBatch> b = cnn::ImagesBatch::fromFiles(files);
	{
		cv::namedWindow("some name23", CV_WINDOW_AUTOSIZE);
		cv::namedWindow("some name24", CV_WINDOW_AUTOSIZE);

		cv::imshow("some name23", b->getImageAsMat(19));

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s1));

		cnn::gpu::GpuBuffer devbuffer(b->getBatchByteSize());

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s12));
		
		devbuffer.writeToDevice(b->getImagesDataPtr(), b->getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s2));
		
		cnn::gpu::Normalizations::centerize<uint>(b, devbuffer);
		
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e1));
		
		devbuffer.loadFromDevice(b->getImagesDataPtr(), b->getBatchByteSize());

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e2));
		
		std::cout << "allocation:     " << double(s12 - s1) * spc << std::endl;
		std::cout << "send:           " << double(s2 - s12) * spc << std::endl;
		std::cout << "comp:           " << double(e1 - s2) * spc << std::endl;
		std::cout << "recv & dealloc: " << double(e2 - e1) * spc << std::endl;
		std::cout << "all:            " << double(e2 - s1) * spc << std::endl;

		cv::imshow("some name24", b->getImageAsMat(19));
	}
	cv::waitKey(0);

    return 0;
}*/