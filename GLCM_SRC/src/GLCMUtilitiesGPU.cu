#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include "GLCMCalculationGPU.cuh"

void PrintGLCM(const float* const glcm, const int depth)
{
	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < depth; j++)
		{
			printf("%20.9f ", glcm[i * depth + j]);
		}
		printf("\n");
	}
}

void CompareGLCMs(const float* const glcm1, const float* const glcm2, const int depth)
{
	printf("\nComparing GLCMs...\n");
	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < depth; j++)
		{
			float diff = glcm1[i * depth + j] - glcm2[i * depth + j];
			if (diff != 0)
				printf("(%d,%d) DOES NOT MATCH\n", i, j);
		}
	}
	printf("...End of Comparison\n");
}

__global__ void BitmapToIntensity_kernel(const unsigned char* pixels,
	unsigned char* intensity, int size)
{
	int offset = 4;
	int uid = blockIdx.x * blockDim.x + threadIdx.x;

	if (uid < size)
	{
		unsigned char rgbBlue = pixels[uid * offset];
		unsigned char rgbGreen = pixels[uid * offset + 1];
		unsigned char rgbRed = pixels[uid * offset + 2];

		// LUMA Coding standard conversion: Y' = 0.299R' + 0.587G' + 0.114B'
		intensity[uid] = (0.299 * rgbRed) + (0.587 * rgbGreen) + (0.114 * rgbBlue);
	}
}

unsigned char* BitmapToIntensityGPU(BitmapData* bmdata)
{
	const float size = bmdata->bmi->bmiHeader.biHeight * bmdata->bmi->bmiHeader.biWidth;
	const int rgbQuadSize = 4;
	const float K = 16;

	unsigned char* d_intensity = 0;
	unsigned char* d_pixels = 0;

	dim3 blocksize(ceil(size / K), 1, 1);
	dim3 threadsize(K, 1, 1);

	checkCudaErrors(cudaMalloc(&d_intensity, sizeof(unsigned char) * size));
	checkCudaErrors(cudaMalloc(&d_pixels, sizeof(unsigned char) * size * rgbQuadSize));

	checkCudaErrors(cudaMemcpy(d_pixels, bmdata->pixels, sizeof(unsigned char) * size * rgbQuadSize, cudaMemcpyHostToDevice));

	BitmapToIntensity_kernel<<<blocksize, threadsize>>>(d_pixels, d_intensity, size);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	cudaFree(d_pixels);

	return d_intensity;
}

__global__ void IntensityScaleLevels_kernel(const unsigned char* intensity,
	unsigned char* intensityNew, int size, float scaleFactor)
{
	int uid = blockIdx.x * blockDim.x + threadIdx.x;

	if (uid < size)
	{
		intensityNew[uid] = (int)((float)intensity[uid] / scaleFactor);
	}
}

unsigned char* IntensityScaleDepthGPU(const unsigned char* d_intensity, const int rows,
	const int cols, const int currentDepth, const int newDepth)
{
	const float size = rows * cols;
	const float scaleFactor = currentDepth / newDepth;
	const float K = 16;

	unsigned char* d_intensityNew = 0;

	dim3 blocksize((size + K - 1) / K, 1, 1);
	dim3 threadsize(K, 1, 1);

	checkCudaErrors(cudaMalloc(&d_intensityNew, sizeof(unsigned char) * size));

	IntensityScaleLevels_kernel<<<blocksize, threadsize>>>(d_intensity, d_intensityNew, size, scaleFactor);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	return d_intensityNew;
}

void GLCMgpu_PrintFeatures(GLCMInfo &gi)
{
	printf("\nPhase 1 Features Parallel\n\n");
	printf("%30s: %.9f\n", "Mean", gi.h_ux);
	printf("%30s: %.9f\n", "Dissimilarity", gi.h_dis);
	printf("%30s: %.9f\n", "Contrast", gi.h_con);
	printf("%30s: %.9f\n", "Inverse Difference Momentum", gi.h_idm);
	printf("%30s: %.9f\n", "Entropy", gi.h_ent);
	printf("%30s: %.9f\n", "Angular Second Momentum", gi.h_asm);
	printf("%30s: %.9f\n", "Maximum Probability", gi.h_map);
	printf("%30s: %.9f\n", "Minimum Probability", gi.h_mip);
	printf("\nPhase 2 Features Parallel\n\n");
	printf("%30s: %.9f\n", "Sum Entropy", gi.h_sen);
	printf("%30s: %.9f\n", "Difference Entropy", gi.h_den);
	printf("%30s: %.9f\n", "Sum Average", gi.h_sav);
	printf("%30s: %.9f\n", "Sum Variance", gi.h_sva);
	printf("%30s: %.9f\n", "Difference Variance", gi.h_dva);
	printf("%30s: %.9f\n", "Variance", gi.h_var);
	printf("%30s: %.9f\n", "Standard Devation", gi.h_sdx);
	printf("\nPhase 3 Features Parallel\n\n");
	printf("%30s: %.9f\n", "Correlation", gi.h_cor);
	printf("%30s: %.9f\n", "Cluster Shade", gi.h_cls);
	printf("%30s: %.9f\n", "Cluster Prominance", gi.h_clp);
}

void GLCMgpu_FreeMemory(GLCMInfo &gi)
{
	cudaFree(gi.d_intensity);
	cudaFree(gi.d_glcm);
}

void GLCMgpu_InitializeMemory(GLCMInfo &gi, unsigned char* in_intensity)
{
	int intensitysize = sizeof(unsigned char) * gi.rows * gi.cols;
	checkCudaErrors(cudaMalloc(&(gi.d_intensity), intensitysize));
	checkCudaErrors(cudaMemcpy(gi.d_intensity, in_intensity, intensitysize, cudaMemcpyHostToDevice));

	int glcmsize = sizeof(float) * gi.depth * gi.depth;
	checkCudaErrors(cudaMalloc(&(gi.d_glcm), glcmsize));
	checkCudaErrors(cudaMemset(gi.d_glcm, 0, glcmsize));
}

void GLCMgpu_GetGLCMMatrixCPU(GLCMInfo &gi, float*& glcm)
{
	checkCudaErrors(cudaMemcpy(glcm, gi.d_glcm, sizeof(float) * gi.depth * gi.depth, cudaMemcpyDeviceToHost));
}