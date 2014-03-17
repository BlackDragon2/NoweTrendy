
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Windows.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ImagesBatch.h"
#include "Kernels.h"


int main()
{
	std::vector<std::string> files;
	for(size_t i=1UL; i<=20; ++i){
		std::stringstream path;
		path << "data/phughe/phughe." << i << ".jpg";
		files.push_back(path.str());
	}

	__int64 freq, s1, s12, s2, e1, e2;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	double spc = 1.0 / freq;

	cnn::ImagesBatch b = cnn::ImagesBatch::fromFiles(files);
	{
		cv::namedWindow("some name23", CV_WINDOW_AUTOSIZE);

		// <<<216, 500>>>
		// 216 blocks per 500 threads
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s1));

		uchar* imgsOnDev;
		cudaMalloc<uchar>(&imgsOnDev, b.getBatchSize());

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s12));
		
		cudaMemcpy(imgsOnDev, b.getImagesData(), b.getBatchSize(), cudaMemcpyHostToDevice);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&s2));
		
		centerImages<<<216, 500>>>(imgsOnDev, b.getImageByteSize(), b.getImagesCount());
		cudaDeviceSynchronize();
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e1));
		
		cudaMemcpy(b.getImagesData(), imgsOnDev, b.getBatchSize(), cudaMemcpyDeviceToHost);
		cudaFree(imgsOnDev);
		
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&e2));
		
		std::cout << "allocation:     " << double(s12 - s1) * spc << std::endl;
		std::cout << "send:           " << double(s2 - s12) * spc << std::endl;
		std::cout << "comp:           " << double(e1 - s2) * spc << std::endl;
		std::cout << "recv & dealloc: " << double(e2 - e1) * spc << std::endl;
		std::cout << "all:            " << double(e2 - s1) * spc << std::endl;

		cv::imshow("some name23", b.getImageAsMat(1));
	}
	cv::waitKey(0);

    return 0;
}