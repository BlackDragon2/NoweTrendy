
#ifndef CNN_SIGNAL_CONVOLUTION_H_
#define CNN_SIGNAL_CONVOLUTION_H_

#include "Convolution.h"
#include "../Config.h"
#include "../Utils/CudaUtils.h"


namespace cnn {
	namespace gpu {


//////////////
// TODO: Shared memory?
template <typename T>
class ImageConvolution : public Convolution<T> {
public:
	struct InputParams {
		T*		data;
		size_t	widthUnit; 
		size_t	alignedWidthUnit; 
		size_t	rowsCount;
	};

	struct KernelsParams {
		T*		data;
		size_t	widthUnit; 
		size_t	alignedWidthUnit; 
		size_t	rowsCount;
		size_t	offsetX;
		size_t	offsetY;
	};

	struct OutputParams {
		T*		data;
		size_t	widthUnit; 
		size_t	alignedWidthUnit; 
		size_t	rowsCount;
	};

	struct GeneralParams {
		size_t	imagesCount;
		size_t	channelsCount;
		size_t	kernelsCount;
		size_t	kernelRunsPerRow;
		size_t	kernelRunsPerImage;
	};


public:
	ImageConvolution(
		uint32 pOffsetX,
		uint32 pOffsetY);
	virtual ~ImageConvolution();

	
	virtual void compute(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer);

	
	virtual uint32 convolvedImageWidth(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const;
	virtual uint32 convolvedImageHeight(
		ImageBatch<T> const& pInputBatch,
		ImageBatch<T> const& pKernelsBatch) const;
};


template <typename T>
ImageConvolution<T>::ImageConvolution(
	uint32 pOffsetX,
	uint32 pOffsetY)
:
	Convolution<T>(pOffsetX, pOffsetY)
{

}


template <typename T>
ImageConvolution<T>::~ImageConvolution(){

}

/*
template <typename T>
__global__ void convolutionImage_old(
	typename ImageConvolution<T>::GeneralParams	pGeneralParams,
	typename ImageConvolution<T>::InputParams		pInputParams, 
	typename ImageConvolution<T>::KernelsParams	pKernelsParams,
	typename ImageConvolution<T>::OutputParams		pOutputParams)
{
	typename ImageConvolution<T>::GeneralParams&	gp = pGeneralParams;
	typename ImageConvolution<T>::InputParams&		ip = pInputParams;
	typename ImageConvolution<T>::KernelsParams&	kp = pKernelsParams;
	typename ImageConvolution<T>::OutputParams&	op = pOutputParams;


	// correct
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= gp.imagesCount * gp.kernelsCount * gp.kernelRunsPerImage)
		return;

	// correct
	uint32 myImageIdx		= idx / (gp.kernelsCount * gp.kernelRunsPerImage);
	uint32 myKernelRunIdx	= (idx / gp.kernelsCount) % gp.kernelRunsPerImage;
	uint32 myKernelIdx		= idx % gp.kernelsCount;

	// correct row
	uint32 myRowIdx			= (myKernelRunIdx / gp.kernelRunsPerRow);
	uint32 myColIdx			= (myKernelRunIdx % gp.kernelRunsPerRow) * gp.channelsCount;

	// correct
	T* myImageRect		= ip.data + myImageIdx * ip.alignedWidthUnit * ip.rowsCount + 
		myRowIdx * ip.alignedWidthUnit * kp.offsetY + myColIdx * kp.offsetX;
	T* myKernel			= kp.data + myKernelIdx * kp.alignedWidthUnit * kp.rowsCount;
	T* myOutputField	= op.data +	myKernelIdx * op.alignedWidthUnit * op.rowsCount + 
		myImageIdx * gp.kernelsCount * op.alignedWidthUnit * op.rowsCount + myRowIdx * op.alignedWidthUnit + myColIdx;

	float	sums[3]		= {0.0F, 0.0F, 0.0F};
	float	maximes[3]	= {0.0F, 0.0F, 0.0F};

	// seems ok
	for(uint32 imgr=0UL; imgr<kp.rowsCount; ++imgr){
		for(uint32 imgc=0UL; imgc<kp.widthUnit; imgc+=gp.channelsCount){
			// convolution for single pixel (sum of sums)
			for(uint32 kr=0UL; kr<=imgr; ++kr){
				for(uint32 kc=0UL; kc<=imgc; kc+=gp.channelsCount){
					// sum += h[kr, kc] * a[imgr - kr, imgc - kc]
					for(uint32 ch=0UL; ch<gp.channelsCount; ++ch){
						float first		= static_cast<float>(*(myImageRect + kr * ip.alignedWidthUnit + kc + ch)) / 255.0F;
						float second	= static_cast<float>(*(myKernel + (imgr - kr) * kp.alignedWidthUnit + (imgc - kc) + ch));
						sums[ch]		+= (first * second);
					}
				}
			}
			// what to do with convolution value for one pixel?
			for(uint32 ch=0ULL; ch<gp.channelsCount; ++ch){
				if(sums[ch] > maximes[ch])
					maximes[ch] = sums[ch];
				sums[ch] = 0.0F;
			}
		}
	}

	for(uint32 ch=0UL; ch<gp.channelsCount; ++ch)
		*(myOutputField + ch) = static_cast<T>(tanh(maximes[ch] / 16.0F) * 255.0F);
}
*/

template <typename T>
__global__ void convolutionImage(
	typename ImageConvolution<T>::GeneralParams	pGeneralParams,
	typename ImageConvolution<T>::InputParams	pInputParams, 
	typename ImageConvolution<T>::KernelsParams	pKernelsParams,
	typename ImageConvolution<T>::OutputParams	pOutputParams)
{
	typename ImageConvolution<T>::GeneralParams&	gp = pGeneralParams;
	typename ImageConvolution<T>::InputParams&		ip = pInputParams;
	typename ImageConvolution<T>::KernelsParams&	kp = pKernelsParams;
	typename ImageConvolution<T>::OutputParams&		op = pOutputParams;


	// correct
	uint32 idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if(idx >= gp.imagesCount * gp.kernelsCount * gp.kernelRunsPerImage)
		return;

	// correct
	uint32 myImageIdx		= idx / (gp.kernelsCount * gp.kernelRunsPerImage);
	uint32 myKernelRunIdx	= (idx / gp.kernelsCount) % gp.kernelRunsPerImage;
	uint32 myKernelIdx		= idx % gp.kernelsCount;

	// correct row
	uint32 myRowIdx			= (myKernelRunIdx / gp.kernelRunsPerRow);
	uint32 myColIdx			= (myKernelRunIdx % gp.kernelRunsPerRow) * gp.channelsCount;

	// correct
	T* myImageRect		= ip.data + myImageIdx * ip.alignedWidthUnit * ip.rowsCount + 
		myRowIdx * ip.alignedWidthUnit * kp.offsetY + myColIdx * kp.offsetX;
	T* myKernel			= kp.data + myKernelIdx * kp.alignedWidthUnit * kp.rowsCount;
	T* myOutputField	= op.data +	myKernelIdx * op.alignedWidthUnit * op.rowsCount + 
		myImageIdx * gp.kernelsCount * op.alignedWidthUnit * op.rowsCount + myRowIdx * op.alignedWidthUnit + myColIdx;

	float sums[3] = {0.0F, 0.0F, 0.0F};
	float norm[3] = {0.0F, 0.0F, 0.0F};

	// convolution for single pixel (sum of sums)
	for(uint32 kr=0UL; kr<kp.rowsCount; ++kr){
		for(uint32 kc=0UL; kc<kp.widthUnit; kc+=gp.channelsCount){
			// sum += h[kr, kc] * a[imgr - kr, imgc - kc]
			for(uint32 ch=0UL; ch<gp.channelsCount; ++ch){
				float first		= static_cast<float>(*(myImageRect + kr * ip.alignedWidthUnit + kc + ch));
				float second	= static_cast<float>(*(myKernel + kr * kp.alignedWidthUnit + kc + ch)) - 128.0F;
				norm[ch] += abs(second);
				sums[ch] += (first * second);
			}
		}
	}

	for(uint32 ch=0UL; ch<gp.channelsCount; ++ch)
		*(myOutputField + ch) = sums[ch] / norm[ch];
}


template <typename T>
void ImageConvolution<T>::compute(
	ImageBatch<T> const&	pInputImageBatch,
	GpuBuffer&				pInputImageBuffer,
	ImageBatch<T> const&	pKernelsImageBatch,
	GpuBuffer&				pKernelsImageBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputImageBuffer)
{
	size_t kernelRunsPerRow		= convolvedImageWidth(pInputImageBatch, pKernelsImageBatch); 
	size_t kernelRunsPerCol		= convolvedImageHeight(pInputImageBatch, pKernelsImageBatch);
	size_t kernelRunsPerImage	= kernelRunsPerRow * kernelRunsPerCol;

	size_t blocks = utils::blocksCount(kernelRunsPerImage * pInputImageBatch.getImagesCount() * pKernelsImageBatch.getImagesCount(), config::Cuda::THREADS_PER_BLOCK);

	GeneralParams gp;
	gp.imagesCount			= pInputImageBatch.getImagesCount();
	gp.channelsCount		= pInputImageBatch.getImageChannelsCount();
	gp.kernelsCount			= pKernelsImageBatch.getImagesCount();
	gp.kernelRunsPerRow		= kernelRunsPerRow;
	gp.kernelRunsPerImage	= kernelRunsPerImage;

	InputParams ip;
	ip.data				= pInputImageBuffer.getDataPtr<T>();
	ip.widthUnit		= pInputImageBatch.getImageRowByteSize() / sizeof(T);
	ip.alignedWidthUnit = pInputImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	ip.rowsCount		= pInputImageBatch.getImageHeight();
	
	KernelsParams kp;
	kp.data				= pKernelsImageBuffer.getDataPtr<T>();
	kp.widthUnit		= pKernelsImageBatch.getImageRowByteSize() / sizeof(T);
	kp.alignedWidthUnit	= pKernelsImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	kp.rowsCount		= pKernelsImageBatch.getImageHeight();
	kp.offsetX			= getOffsetX();
	kp.offsetY			= getOffsetY();

	OutputParams op;
	op.data				= pOutputImageBuffer.getDataPtr<T>();
	op.widthUnit		= pOutputImageBatch.getImageRowByteSize() / sizeof(T);
	op.alignedWidthUnit	= pOutputImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	op.rowsCount		= pOutputImageBatch.getImageHeight();

	convolutionImage<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(gp, ip, kp, op);
}


template <typename T>
uint32 ImageConvolution<T>::convolvedImageWidth(
	ImageBatch<T> const& pInputBatch,
	ImageBatch<T> const& pKernelsBatch) const 
{
	return (pInputBatch.getImageWidth() - pKernelsBatch.getImageWidth()) / getOffsetX() + 1; 
}


template <typename T>
uint32 ImageConvolution<T>::convolvedImageHeight(
	ImageBatch<T> const& pInputBatch,
	ImageBatch<T> const& pKernelsBatch) const 
{
	return (pInputBatch.getImageHeight() - pKernelsBatch.getImageHeight()) / getOffsetY() + 1;
}


	}
}


#endif	/* CNN_SIGNAL_CONVOLUTION_H_ */