
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


#define MEASURE_SEPARATE



int main()
{


	srand((uint32)time(0));

	std::shared_ptr<std::vector<size_t>> folds = cnn::utils::FoldsFactory::prepareSequence(117, 7, cnn::utils::FoldsFactory::FitTactic::DEFAULT);
	std::shared_ptr<std::vector<size_t>> folds2 = cnn::utils::FoldsFactory::prepareSequence(117, 7, cnn::utils::FoldsFactory::FitTactic::CUT);
	std::shared_ptr<std::vector<size_t>> folds3 = cnn::utils::FoldsFactory::prepareSequence(117, 7, cnn::utils::FoldsFactory::FitTactic::EXTEND_WITH_COPIES);

	cnn::utils::FoldsFactory::FoldsPtrS f1 = cnn::utils::FoldsFactory::prepareFolds(folds, 7);
	cnn::utils::FoldsFactory::FoldsPtrS f2 = cnn::utils::FoldsFactory::prepareFolds(folds2, 7);
	cnn::utils::FoldsFactory::FoldsPtrS f3 = cnn::utils::FoldsFactory::prepareFolds(folds3, 7);

	std::vector<std::string> files;
	std::string names[] = {
		"9336923", "9338535", "anpage", "asamma", "asewil",
		"astefa", "drbost", "ekavaz", "elduns", "kaknig", 
		"klclar", "ksunth", "lfso", "mbutle", "phughe", 
		"sbains", "slbirc", "vstros", "yfhsie"};
	

	size_t nsize = ARRAYSIZE(names);

	for(size_t a=0UL; a<1UL; ++a){
		for(size_t n=0UL; n<nsize; ++n){
			for(size_t i=1UL; i<=20; ++i){
				std::stringstream path;
				path << "data/" << names[n] << "/" << names[n] << "." << i << ".jpg";
				files.push_back(path.str());
			}
		}
		/*for(size_t i=1UL; i<=2; ++i){
			std::stringstream path;
			path << "data/test/test." << i << ".jpg";
			files.push_back(path.str());
		}*/
	}

	__int64 freq, s1, s12, s2, e1, e2, pre1;
	__int64 p0, p1, p2, p3;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	double spc = 1.0 / freq;

	std::shared_ptr<cnn::ImageBatch<uchar>> b = cnn::ImageBatch<uchar>::fromFiles(files, true);

	cnn::ImageBatch<uchar> center(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	center.allocateSpaceForImages(1, true);
	
	cnn::ImageBatch<uchar> centerized(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	centerized.allocateSpaceForImages(b->getImagesCount(), true);
	
	cnn::ImageBatch<uchar> eroded(b->getImageWidth(), b->getImageHeight(), b->getImageChannelsCount());
	eroded.allocateSpaceForImages(b->getImagesCount(), true);
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&pre1));
		//std::vector<std::pair<uchar, uchar> > res = b->findImagesColorsBoundaries();
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s1));

		cnn::gpu::GpuBuffer devbuffer(b->getBatchByteSize());
		cnn::gpu::GpuBuffer devbuffersingle(b->getAlignedImageByteSize());
		cnn::gpu::GpuBuffer devbuffer2(b->getBatchByteSize());
		cnn::gpu::GpuBuffer devbound(b->getImagesCount() * b->getImageChannelsCount() * 2);
		cnn::gpu::GpuBuffer devbuffer3(b->getBatchByteSize());
		//cnn::gpu::GpuBuffer devbufferfloat(b->getBatchByteSize() * sizeof(float));

		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s12));
		
		devbuffer.writeToDevice(b->getBatchDataPtr(), b->getBatchByteSize());
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s2));
	
		cnn::gpu::Tasks::buildCenterMap<uchar>(*b, devbuffer, devbuffersingle);
#ifdef MEASURE_SEPARATE
		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&p0));
#endif		

		cnn::gpu::Tasks::centerizeWithMap<uchar>(*b, devbuffer, devbuffersingle, devbuffer2);
#ifdef MEASURE_SEPARATE
		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&p1));
#endif

		cnn::gpu::Tasks::findEachImageBoundaries<uchar>(*b, devbuffer2, devbound);
#ifdef MEASURE_SEPARATE
		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&p2));
#endif		

		cnn::gpu::Tasks::erodeEachImageUsingBoundaries<uchar>(*b, devbuffer2, devbound, devbuffer3, 255);
#ifdef MEASURE_SEPARATE
		assert(cudaDeviceSynchronize() == cudaSuccess);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&p3));

		std::cout 
			<< "center map: " << (double(p0 - s2) * spc)
			<< "\n centering: " << (double(p1 - p0) * spc)
			<< "\nboundaries: " << (double(p2 - p1) * spc)
			<< "\n   eroding: " << (double(p3 - p2) * spc)
			<< "\n\n";
#endif

		//cnn::gpu::Tasks::convert<uchar, float>(*b, devbuffer3, devbufferfloat);
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e1));
		assert(cudaDeviceSynchronize() == cudaSuccess);
		
		std::vector<uchar> bounds(b->getImagesCount() * b->getImageChannelsCount() * 2);
		devbound.loadFromDevice(bounds.data(), b->getImagesCount() * b->getImageChannelsCount() * 2);

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

		cv::imshow("some name1", b->retriveImageAsMat(130));
		cv::imshow("some name2", center.retriveImageAsMat(0));
		cv::imshow("some name3", centerized.retriveImageAsMat(130));
		cv::imshow("some name4", eroded.retriveImageAsMat(130));
	}
	cv::waitKey(0);

    return 0;
}
