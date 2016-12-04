#include <malloc.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include "GLCMCalculationGPU.cuh"

__global__ void GLCMNormalize_kernel(float* glcm, const int R)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	glcm[id] /= R;
}

void GLCMgpu_NormalizeGLCM(GLCMInfo &gi)
{
	GLCMNormalize_kernel<<<gi.depth, gi.depth>>>(gi.d_glcm, gi.R);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

__global__ void GLCMPerGrid_kernel(const unsigned char* intensity, float* glcm, const int rows, const int cols,
	const int depth, const int xmin, const int xmax, const int ymin, const int copixel)
{
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int iid = x + y * cols;

	if (x >= xmin && x < xmax && y >= ymin && y < rows)
	{
		unsigned char i = intensity[iid];
		unsigned char j = intensity[iid + copixel];
		atomicAdd(&(glcm[i * depth + j]), 1);
		atomicAdd(&(glcm[j * depth + i]), 1);
	}
}

void GLCMPerGrid(GLCMInfo &gi, float* time)
{
	const int K = 16;

	int xmin = 0, xmax = 0, ymin = 0, copixel = 0;

	if		(gi.angle == 0)   { xmin = 0;           xmax = gi.cols - gi.distance; ymin = 0;			  copixel = 1;			    }
	else if (gi.angle == 45)  { xmin = 0;		    xmax = gi.cols - gi.distance; ymin = gi.distance; copixel = 1 - gi.cols;  }
	else if (gi.angle == 90)  { xmin = 0;			xmax = gi.cols;               ymin = gi.distance; copixel = -(gi.cols);   }
	else if (gi.angle == 135) { xmin = gi.distance; xmax = gi.cols;               ymin = gi.distance; copixel = -1 - gi.cols; }
	else {} // invalid angle -- handle?

	dim3 blocks((gi.cols + K - 1) / K, (gi.rows + K - 1) / K);
	dim3 threads(K, K);
	cudaEvent_t start = 0, stop = 0;

	checkCudaErrors(cudaEventCreate(&start, 0));
	checkCudaErrors(cudaEventCreate(&stop, 0));

	checkCudaErrors(cudaEventRecord(start, 0));
	GLCMPerGrid_kernel<<<blocks, threads>>>(gi.d_intensity, gi.d_glcm, gi.rows, gi.cols, gi.depth, xmin, xmax, ymin, copixel);
	checkCudaErrors(cudaEventRecord(stop, 0));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventElapsedTime(time, start, stop));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void GLCMPerBlock_kernel(const unsigned char* intensity, float* glcm, const int rows, const int cols,
	const int depth, const int xmin, const int xmax, const int ymin, const int copixel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int iid = x + y * cols;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int blocksize = blockDim.x * blockDim.y;
	int glcmsize = depth * depth;

	extern __shared__ int glcmshared[];
	int k = tid;
	while (k < glcmsize)
	{
		glcmshared[k] = 0;
		k += blocksize;
	}

	__syncthreads();

	if (x >= xmin && x < xmax && y >= ymin && y < rows)
	{
		unsigned char i = intensity[iid];
		unsigned char j = intensity[iid + copixel];
		atomicAdd(&(glcmshared[i * depth + j]), 1);
		atomicAdd(&(glcmshared[j * depth + i]), 1);
	}

	__syncthreads();

	k = tid;
	while (k < glcmsize)
	{
		atomicAdd(&(glcm[k]), glcmshared[k]);
		k += blocksize;
	}
}

void GLCMPerBlock(GLCMInfo &gi, float* time)
{
	const int glcmsize = gi.depth * gi.depth * sizeof(float);
	const int K = 16;

	int xmin = 0, xmax = 0, ymin = 0, copixel = 0;

	if		(gi.angle == 0)   { xmin = 0;            xmax = gi.cols - gi.distance; ymin = 0;			   copixel = 1;             }
	else if (gi.angle == 45)  { xmin = 0;		      xmax = gi.cols - gi.distance; ymin = gi.distance; copixel = 1 - gi.cols;  }
	else if (gi.angle == 90)  { xmin = 0;			  xmax = gi.cols;                ymin = gi.distance; copixel = -(gi.cols);   }
	else if (gi.angle == 135) { xmin = gi.distance; xmax = gi.cols;                ymin = gi.distance; copixel = -1 - gi.cols; }
	else {} // invalid angle -- handle?

	dim3 blocks((gi.cols + K - 1) / K, (gi.rows + K - 1) / K);
	dim3 threads(K, K);
	cudaEvent_t start = 0, stop = 0;

	checkCudaErrors(cudaEventCreate(&start, 0));
	checkCudaErrors(cudaEventCreate(&stop, 0));

	checkCudaErrors(cudaEventRecord(start, 0));
	GLCMPerBlock_kernel<<<blocks, threads, glcmsize>>>(gi.d_intensity, gi.d_glcm, gi.rows, gi.cols, gi.depth, xmin, xmax, ymin, copixel);
	checkCudaErrors(cudaEventRecord(stop, 0));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventElapsedTime(time, start, stop));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void GLCMgpu_CalculateGLCM(GLCMInfo &gi)
{
	float timeElapsed;

	if (gi.depth > 64)
		GLCMPerGrid(gi, &timeElapsed);
	else
		GLCMPerBlock(gi, &timeElapsed);
}