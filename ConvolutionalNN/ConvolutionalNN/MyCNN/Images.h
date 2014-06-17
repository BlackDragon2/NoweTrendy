#ifndef CNN_LOADER_H_
#define CNN_LOADER_H_

#include <Windows.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <string.h>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include "../Types.h"
#include "../GPU/GpuBuffer.cuh"

namespace mycnn{

class Images
{
public: 
	Images();
	~Images();
	void load(std::string classes[], uint32 classesCount, uint32 examplesNr, std::string source, bool color);
	uint32 getClass(uint32 exampleIndex);
	uint32 getImageWidth();
	uint32 getImageHeight();
	uint32 getImageCount();
	uint32 getImageDist();//roznica we floatach miedzy obrazkami
	uint32 getImagesByteSize();
	uint32 getImagesSize();
	uint32 getImageSize();
	cnn::gpu::GpuBuffer* getImageBuffer();

private:
	uint32 imageWidth;//szerokosc=liczba kolumn
	uint32 imageHeight;//wysokosc=liczba wierszy
	uint32 imageCount;
	uint32 examplesPerClass;//liczba roznych klas
	std::string* classesInBuffer;//inty odpowiadajace klasa
	bool inColor;
	cnn::gpu::GpuBuffer* buffer;//bgr pikseli, wierszami
};}


#endif	/* CNN_LOADER_H_ */