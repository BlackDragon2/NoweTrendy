
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ImagesBatch.h"
#include "Kernels.h"


int main()
{
	std::vector<std::string> files;
	files.push_back("data/test.png");
	files.push_back("data/test2.png");

	cnn::ImagesBatch b = cnn::ImagesBatch::fromFiles(files);
	{
		cv::namedWindow("some name", CV_WINDOW_AUTOSIZE);
		cv::namedWindow("some name2", CV_WINDOW_AUTOSIZE);
		cv::imshow("some name", b.getImageAsMat(0));
		cv::imshow("some name2", b.getImageAsMat(1));
	}
	cv::waitKey(0);

    return 0;
}