
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
#include "GPU/Normalizations.cuh"
#include "GPU/GpuBuffer.cuh"
#include "Types.h"
#include "Utils/FoldsFactory.h"


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

	std::shared_ptr<cnn::ImageBatch> b = cnn::ImageBatch::fromFiles(files);
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