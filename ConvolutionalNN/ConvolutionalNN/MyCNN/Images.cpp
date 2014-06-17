#include "Images.h"

mycnn::Images::Images()
{
	buffer=new cnn::gpu::GpuBuffer();
}

uint32 mycnn::Images::getImagesByteSize()
{
	if(inColor)
		return 3*imageHeight*imageWidth*sizeof(float)*imageCount;
	else
		return imageHeight*imageWidth*sizeof(float)*imageCount;
}

uint32 mycnn::Images::getImageDist()
{
	if(inColor)
		return 3*imageHeight*imageWidth;
	else
		return imageHeight*imageWidth;
}

uint32 mycnn::Images::getImageHeight()
{
	return imageHeight;
}

uint32 mycnn::Images::getImageWidth()
{
	return imageWidth;
}

uint32 mycnn::Images::getImageCount()
{
	return imageCount;
}

uint32 mycnn::Images::getClass(uint32 index)
{
	return index/examplesPerClass;
}

mycnn::Images::~Images()
{
	buffer->free();
	delete buffer;
	delete[] classesInBuffer;
}

void mycnn::Images::load(std::string classes[], uint32 classesCount, uint32 examplesPerClass, std::string source, bool inColor)
{
	imageCount=examplesPerClass*classesCount;
	this->examplesPerClass=examplesPerClass;
	this->inColor=inColor;
	classesInBuffer=new std::string[classesCount];
	memcpy(classesInBuffer, classes, sizeof(classes));

	std::vector<std::string> files;
	for(size_t n=0; n<classesCount; n++)
	{
		for(size_t i=0; i<examplesPerClass; i++)
		{
				std::stringstream path;
				path << source<<"/" << classes[n] << "/" << classes[n] << "." << i << ".jpg";
				files.push_back(path.str());
		}
	}
	int readType	= inColor ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;	
	cv::Mat mat		= cv::imread(files[0], readType);

	imageWidth=mat.size().width;
	imageHeight=mat.size().height;

	float* toCopy;
	if(inColor)
	{
		buffer->allocate(3*imageWidth*imageHeight*imageCount*sizeof(float));
		toCopy=new float[3*imageWidth*imageHeight*imageCount*sizeof(float)];
	}
	else
	{
		buffer->allocate(imageWidth*imageHeight*imageCount*sizeof(float));
		toCopy=new float[imageWidth*imageHeight*imageCount*sizeof(float)];
	}
	
	for(int k=0;k<imageCount;k++)
	{
		cv::Mat mat		= cv::imread(files[k], readType);
		unsigned char *input = (unsigned char*)(mat.data);
		for(int i = 0;i < mat.rows ;i++)
		{
			if(inColor)
			{
				for(int j = 0;j < mat.cols ;j++)
				{
					toCopy[k*getImageDist()+mat.step * j + i] = input[mat.step * j + i ]/255;
					toCopy[k*getImageDist()+mat.step * j + i+imageHeight*imageWidth] = input[mat.step * j + i+1]/255;
					toCopy[k*getImageDist()+mat.step * j + i+2*imageHeight*imageWidth] = input[mat.step * j + i+2]/255;
				}
			}
			else
			{
				for(int j = 0;j < mat.cols ;j++)
					toCopy[k*getImageDist()+mat.step * j + i] = input[mat.step * j + i ]/255;
			}
        }
	}
	buffer->writeToDevice(toCopy, getImagesByteSize());
	delete[] toCopy;
}

uint32 mycnn::Images::getImagesSize()
{
	if(inColor)
		return 3*imageHeight*imageWidth*imageCount;
	else
		return imageHeight*imageWidth*imageCount;
}

uint32 mycnn::Images::getImageSize()
{
	if(inColor)
		return 3*imageHeight*imageWidth;
	else
		return imageHeight*imageWidth;
}

cnn::gpu::GpuBuffer* mycnn::Images::getImageBuffer()
{
	return buffer;
}