
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Utils/Utils.h"
#include "Kernels.h"


int main()
{
	cv::Mat rgb = cv::imread("data/test.png", CV_LOAD_IMAGE_COLOR);
	if(!rgb.data){
		std::cout << "empty?" << std::endl;
		return -1;
	}
	
	cv::namedWindow("some name", CV_WINDOW_AUTOSIZE);
	cv::imshow("some name", rgb);

	cv::waitKey(0);

    return 0;
}