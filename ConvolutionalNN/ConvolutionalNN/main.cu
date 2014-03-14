
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
	cv::namedWindow("some name", CV_WINDOW_AUTOSIZE);
	cv::imshow("some name", rgb);

	cvWaitKey(1000);
	system("pause");

    return 0;
}