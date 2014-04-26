
#ifndef CNN_SIGNAL_CONVOLUTION_H_
#define CNN_SIGNAL_CONVOLUTION_H_

#include "Convolution.h"
#include "../Config.h"


namespace cnn {
	namespace gpu {


//////////////
// TODO: Shared memory?
template <typename T>
class SignalConvolution : public Convolution<T> {
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
	SignalConvolution();
	virtual ~SignalConvolution();
	
	virtual void compute(
		ImageBatch<T> const&	pInputImageBatch,
		GpuBuffer&				pInputImageBuffer,
		ImageBatch<T> const&	pKernelsImageBatch,
		GpuBuffer&				pKernelsImageBuffer,
		ImageBatch<T> const&	pOutputImageBatch,
		GpuBuffer&				pOutputImageBuffer,
		uint32					pKernelOffsetX,
		uint32					pKernelOffsetY);
};


template <typename T>
SignalConvolution<T>::SignalConvolution(){

}


template <typename T>
SignalConvolution<T>::~SignalConvolution(){

}


template <typename T>
__global__ void signalConvolution(
	typename SignalConvolution<T>::GeneralParams	pGeneralParams,
	typename SignalConvolution<T>::InputParams		pInputParams, 
	typename SignalConvolution<T>::KernelsParams	pKernelsParams,
	typename SignalConvolution<T>::OutputParams		pOutputParams)
{
	typename SignalConvolution<T>::GeneralParams&	gp = pGeneralParams;
	typename SignalConvolution<T>::InputParams&		ip = pInputParams;
	typename SignalConvolution<T>::KernelsParams&	kp = pKernelsParams;
	typename SignalConvolution<T>::OutputParams&	op = pOutputParams;


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
			// convolution for single pixel
			for(uint32 kr=0UL; kr<=imgr; ++kr){
				for(uint32 kc=0UL; kc<=imgc; kc+=gp.channelsCount){
					// sum += h[kr, kc] * a[imgr - kr, imgc - kc]
					for(uint32 ch=0UL; ch<gp.channelsCount; ++ch){
						float first		= static_cast<float>(*(myImageRect + kr * ip.alignedWidthUnit + kc + ch));
						float second	= static_cast<float>(*(myKernel + (imgr - kr) * kp.alignedWidthUnit + (imgc - kc) + ch));
						sums[ch]		+= (first * second);
					}
				}
			}
			for(uint32 ch=0ULL; ch<gp.channelsCount; ++ch){
				if(sums[ch] > maximes[ch])
					maximes[ch] = MIN(255.0F, sums[ch] / 512.0F);
				sums[ch] = 0.0F;
			}
		}
	}

	for(uint32 ch=0UL; ch<gp.channelsCount; ++ch)
		*(myOutputField + ch) = maximes[ch];
}

template <typename T>
void SignalConvolution<T>::compute(
	ImageBatch<T> const&	pInputImageBatch,
	GpuBuffer&				pInputImageBuffer,
	ImageBatch<T> const&	pKernelsImageBatch,
	GpuBuffer&				pKernelsImageBuffer,
	ImageBatch<T> const&	pOutputImageBatch,
	GpuBuffer&				pOutputImageBuffer,
	uint32					pKernelOffsetX,
	uint32					pKernelOffsetY)
{
	size_t kernelRunsPerRow		= (pInputImageBatch.getImageWidth() - pKernelsImageBatch.getImageWidth()) / pKernelOffsetX + 1; 
	size_t kernelRunsPerCol		= (pInputImageBatch.getImageHeight() - pKernelsImageBatch.getImageHeight()) / pKernelOffsetY + 1; 
	size_t kernelRunsPerImage	= kernelRunsPerRow * kernelRunsPerCol;

	size_t blocks	= static_cast<size_t>(std::ceil(static_cast<double>(
		kernelRunsPerImage * pInputImageBatch.getImagesCount() * pKernelsImageBatch.getImagesCount()) / config::Cuda::THREADS_PER_BLOCK));

	SignalConvolution<T>::GeneralParams gp;
	gp.imagesCount			= pInputImageBatch.getImagesCount();
	gp.channelsCount		= pInputImageBatch.getImageChannelsCount();
	gp.kernelsCount			= pKernelsImageBatch.getImagesCount();
	gp.kernelRunsPerRow		= kernelRunsPerRow;
	gp.kernelRunsPerImage	= kernelRunsPerImage;

	SignalConvolution<T>::InputParams ip;
	ip.data				= pInputImageBuffer.getDataPtr<T>();
	ip.widthUnit		= pInputImageBatch.getImageRowByteSize() / sizeof(T);
	ip.alignedWidthUnit = pInputImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	ip.rowsCount		= pInputImageBatch.getImageHeight();
	
	SignalConvolution<T>::KernelsParams kp;
	kp.data				= pKernelsImageBuffer.getDataPtr<T>();
	kp.widthUnit		= pKernelsImageBatch.getImageRowByteSize() / sizeof(T);
	kp.alignedWidthUnit	= pKernelsImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	kp.rowsCount		= pKernelsImageBatch.getImageHeight();
	kp.offsetX			= pKernelOffsetX;
	kp.offsetY			= pKernelOffsetY;

	SignalConvolution<T>::OutputParams op;
	op.data				= pOutputImageBuffer.getDataPtr<T>();
	op.widthUnit		= pOutputImageBatch.getImageRowByteSize() / sizeof(T);
	op.alignedWidthUnit	= pOutputImageBatch.getAlignedImageRowByteSize() / sizeof(T);
	op.rowsCount		= pOutputImageBatch.getImageHeight();

	signalConvolution<T><<<blocks, config::Cuda::THREADS_PER_BLOCK>>>(gp, ip, kp, op);
}


	}
}


#endif	/* CNN_SIGNAL_CONVOLUTION_H_ */