#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include "GLCMCalculationGPU.cuh"

template <unsigned int blocksize>
__device__ void MinWarpReduce(volatile float* shared, unsigned int tid) {
	if (blocksize >= 64) shared[tid] = fminf(shared[tid], shared[tid + 32]);
	if (blocksize >= 32) shared[tid] = fminf(shared[tid], shared[tid + 16]);
	if (blocksize >= 16) shared[tid] = fminf(shared[tid], shared[tid + 8]);
	if (blocksize >= 8)  shared[tid] = fminf(shared[tid], shared[tid + 4]);
	if (blocksize >= 4)  shared[tid] = fminf(shared[tid], shared[tid + 2]);
	if (blocksize >= 2)  shared[tid] = fminf(shared[tid], shared[tid + 1]);
}

template<unsigned int blocksize>
__device__ void MinReduction_dev_kernel(float* shared)
{
	int tid = threadIdx.x;

	if (blocksize >= 256)
	{
		if (tid < 128)
			shared[tid] = fminf(shared[tid], shared[tid + 128]);
		__syncthreads();
	}

	if (blocksize >= 128)
	{
		if (tid < 64)
			shared[tid] = fminf(shared[tid], shared[tid + 64]);
		__syncthreads();
	}

	if (tid < 32) MinWarpReduce<blocksize>(shared, tid);

	//if (tid < 32)
	//{
	//	if (blocksize >= 64)
	//		shared[tid] = fminf(shared[tid], shared[tid + 32]);
	//	if (blocksize >= 32)
	//		shared[tid] = fminf(shared[tid], shared[tid + 16]);
	//	if (blocksize >= 16)
	//		shared[tid] = fminf(shared[tid], shared[tid + 8]);
	//	if (blocksize >= 8)
	//		shared[tid] = fminf(shared[tid], shared[tid + 4]);
	//	if (blocksize >= 4)
	//		shared[tid] = fminf(shared[tid], shared[tid + 2]);
	//	if (blocksize >= 2)
	//		shared[tid] = fminf(shared[tid], shared[tid + 1]);
	//}
}

template <unsigned int blocksize>
__device__ void MaxWarpReduce(volatile float* shared, unsigned int tid) {
	if (blocksize >= 64) shared[tid] = fmaxf(shared[tid], shared[tid + 32]);
	if (blocksize >= 32) shared[tid] = fmaxf(shared[tid], shared[tid + 16]);
	if (blocksize >= 16) shared[tid] = fmaxf(shared[tid], shared[tid + 8]);
	if (blocksize >= 8)  shared[tid] = fmaxf(shared[tid], shared[tid + 4]);
	if (blocksize >= 4)  shared[tid] = fmaxf(shared[tid], shared[tid + 2]);
	if (blocksize >= 2)  shared[tid] = fmaxf(shared[tid], shared[tid + 1]);
} 

template<unsigned int blocksize>
__device__ void MaxReduction_dev_kernel(float* shared)
{
	int tid = threadIdx.x;

	if (blocksize >= 256)
	{
		if (tid < 128)
			shared[tid] = fmaxf(shared[tid], shared[tid + 128]);
		__syncthreads();
	}

	if (blocksize >= 128)
	{
		if (tid < 64)
			shared[tid] = fmaxf(shared[tid], shared[tid + 64]);
		__syncthreads();
	}

	if (tid < 32) MaxWarpReduce<blocksize>(shared, tid);

	//if (tid < 32)
	//{
	//	if (blocksize >= 64)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 32]);
	//	if (blocksize >= 32)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 16]);
	//	if (blocksize >= 16)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 8]);
	//	if (blocksize >= 8)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 4]);
	//	if (blocksize >= 4)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 2]);
	//	if (blocksize >= 2)
	//		shared[tid] = fmaxf(shared[tid], shared[tid + 1]);
	//}
}

template <unsigned int blocksize>
__device__ void SumWarpReduce(volatile float* shared, unsigned int tid) {
	if (blocksize >= 64) shared[tid] += shared[tid + 32];
	if (blocksize >= 32) shared[tid] += shared[tid + 16];
	if (blocksize >= 16) shared[tid] += shared[tid + 8];
	if (blocksize >= 8)  shared[tid] += shared[tid + 4];
	if (blocksize >= 4)  shared[tid] += shared[tid + 2];
	if (blocksize >= 2)  shared[tid] += shared[tid + 1];
}

template<unsigned int blocksize>
__device__ void SumReduction_dev_kernel(float* shared)
{
	int tid = threadIdx.x;

	if (blocksize >= 256)
	{
		if (tid < 128)
			shared[tid] += shared[tid + 128];
		__syncthreads();
	}

	if (blocksize >= 128)
	{
		if (tid < 64)
			shared[tid] += shared[tid + 64];
		__syncthreads();
	}

	if (tid < 32) SumWarpReduce<blocksize>(shared, tid);

	//if (tid < 32)
	//{
	//	if (blocksize >= 64)
	//		shared[tid] += shared[tid + 32];
	//	if (blocksize >= 32)
	//		shared[tid] += shared[tid + 16];
	//	if (blocksize >= 16)
	//		shared[tid] += shared[tid + 8];
	//	if (blocksize >= 8)
	//		shared[tid] += shared[tid + 4];
	//	if (blocksize >= 4)
	//		shared[tid] += shared[tid + 2];
	//	if (blocksize >= 2)
	//		shared[tid] += shared[tid + 1];
	//}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP1A_kernel(float* g_glcm, float* g_ux, float* g_dis,
	float* g_con, float* g_idm, float* g_ent, float* g_asm, float* g_map, float* g_mip)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;
	int i = blockIdx.x * 2;
	int j = threadIdx.x;

	float a = g_glcm[uid];
	float b = g_glcm[uid + blocksize];

	// mean-x
	{
		shared[tid] = (i * a) + ((i + 1) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_ux[blockIdx.x] = shared[0];
	}

	// dissimilarity
	{
		shared[tid] = (a * abs(i - j)) + (b * abs((i + 1) - j));
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_dis[blockIdx.x] = shared[0];
	}

	// contrast
	{
		int at = i - j;
		int bt = i + 1 - j;
		shared[tid] = (a * (at * at)) + (b * (bt * bt));
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_con[blockIdx.x] = shared[0];
	}

	// inverse difference momentum
	{
		float at = (float)(i - j);
		float bt = (float)(i + 1 - j);
		shared[tid] = (a * (1.0f / (1.0f + (at * at)))) + (b * (1.0f / (1.0f + (bt * bt))));
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_idm[blockIdx.x] = shared[0];
	}

	// entropy
	{
		float loga = (a == 0) ? (0.0f) : (log(a));
		float logb = (b == 0) ? (0.0f) : (log(b));
		shared[tid] = (a * loga) + (b * logb);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_ent[blockIdx.x] = shared[0];
	}

	// angular second momentum
	{
		shared[tid] = (a * a) + (b * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_asm[blockIdx.x] = shared[0];
	}

	// max probability
	{
		shared[tid] = fmaxf(a, b);
		__syncthreads();
		MaxReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_map[blockIdx.x] = shared[0];
	}

	// min probability
	{
		shared[tid] = fminf(a, b);
		__syncthreads();
		MinReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_mip[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP1B_kernel(float* g_ux, float* g_dis,
	float* g_con, float* g_idm, float* g_ent, float* g_asm, float* g_map, float* g_mip)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;

	// mean-x
	{
		shared[tid] = g_ux[uid] + g_ux[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_ux[blockIdx.x] = shared[0];
	}

	// dissimilarity
	{
		shared[tid] = g_dis[uid] + g_dis[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_dis[blockIdx.x] = shared[0];
	}

	// contrast
	{
		shared[tid] = g_con[uid] + g_con[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_con[blockIdx.x] = shared[0];
	}

	// inverse difference momentum
	{
		shared[tid] = g_idm[uid] + g_idm[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_idm[blockIdx.x] = shared[0];
	}

	// entropy
	{
		shared[tid] = g_ent[uid] + g_ent[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_ent[blockIdx.x] = -shared[0];
	}

	// angular second momentum
	{
		shared[tid] = g_asm[uid] + g_asm[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_asm[blockIdx.x] = shared[0];
	}

	// max probability
	{
		shared[tid] = fmaxf(g_map[uid], g_map[uid + blocksize]);
		__syncthreads();
		MaxReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_map[blockIdx.x] = shared[0];
	}

	// min probability
	{
		shared[tid] = fminf(g_mip[uid], g_mip[uid + blocksize]);
		__syncthreads();
		MinReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_mip[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP1C_kernel(float* g_glcm, float* g_pxpy)
{
	extern __shared__ float shared[];

	int k = blockIdx.x;
	int i = threadIdx.x;

	// anti diagonal probability
	{
		int j = k - i;
		shared[i] = (j < 0 || j >= blocksize) ? 0.0f : g_glcm[i * blocksize + j];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (i == 0) g_pxpy[k] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP1D_kernel(float* g_glcm, float* g_pxmy)
{
	extern __shared__ float shared[];

	int k = blockIdx.x;
	int i = threadIdx.x;

	// main diagonal probability
	{
		int j = k + i;
		shared[i] = (j >= blocksize) ? 0.0f : g_glcm[i * blocksize + j];
		if (i != j) shared[i] += shared[i];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (i == 0) g_pxmy[k] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP2A_kernel(float* g_pxpy, float* g_sen, float* g_sav, float* g_sva)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int k = threadIdx.x * 2;

	float a = g_pxpy[k];
	float b = g_pxpy[k + 1];
	float sen;

	// sum entropy
	{
		float loga = (a == 0) ? (0.0f) : (log(a));
		float logb = (b == 0) ? (0.0f) : (log(b));
		shared[tid] = (a * loga) + (b * logb);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_sen[blockIdx.x] = -shared[0];
	}

	__syncthreads();

	// sum variance
	{
		sen = shared[0];
		float kdif = (float)k - sen;
		float k2dif = (float)(k + 1) - sen;
		shared[tid] = ((kdif * kdif) * a) + ((k2dif * k2dif) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_sva[blockIdx.x] = shared[0];
	}

	// sum average
	{
		shared[tid] = ((float)k * a) + ((float)(k + 1) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_sav[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP2B_kernel(float* g_pxmy, float* g_den, float* g_dva)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int k = threadIdx.x * 2;

	float a = g_pxmy[k];
	float b = g_pxmy[k + 1];

	// difference entropy
	{
		float loga = (a == 0) ? (0.0f) : (log(a));
		float logb = (b == 0) ? (0.0f) : (log(b));
		shared[tid] = (a * loga) + (b * logb);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_den[blockIdx.x] = -shared[0];
	}

	// difference variance
	{
		shared[tid] = ((float)(k * k) * a) + ((float)((k + 1) * (k + 1)) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_dva[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP2C_kernel(float* g_glcm, float* g_ux, float* g_var)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;
	int i = blockIdx.x * 2;

	float a = g_glcm[uid];
	float b = g_glcm[uid + blocksize];

	// variance
	{
		float ux = *g_ux;
		float adif = ((float)i - ux);
		float bdif = ((float)(i + 1) - ux);
		shared[tid] = ((adif * adif) * a) + ((bdif * bdif) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_var[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP2D_kernel(float* g_var, float* g_sdx)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;

	// variance & standard deviation
	{
		shared[tid] = g_var[uid] + g_var[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0)
		{
			g_var[blockIdx.x] = shared[0];
			g_sdx[blockIdx.x] = sqrtf(shared[0]);
		}
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP3A_kernel(float* g_glcm, float* g_ux, float* g_cor, float* g_cls, float* g_clp)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;
	float i = blockIdx.x * 2;
	float j = threadIdx.x;

	float a = g_glcm[uid];
	float b = g_glcm[uid + blocksize];
	float ux = *g_ux;

	// correlation
	{
		float jminus_ux = j - ux;
		shared[tid] = (((i - ux) * jminus_ux) * a) + ((((i + 1.0f) - ux) * jminus_ux) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_cor[blockIdx.x] = shared[0];
	}

	float ijdif = (i + j) - ux - ux;
	float ijdif_cubed = ijdif * ijdif * ijdif;
	float i2jdif = ((i + 1.0f) + j) - ux - ux;
	float i2jdif_cubed = i2jdif * i2jdif * i2jdif;

	// cluster shade
	{
		shared[tid] = (ijdif_cubed * a) + (i2jdif_cubed * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_cls[blockIdx.x] = shared[0];
	}

	// cluster prominence
	{
		shared[tid] = ((ijdif_cubed * ijdif) * a) + ((i2jdif_cubed * i2jdif) * b);
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_clp[blockIdx.x] = shared[0];
	}
}

template<unsigned int blocksize>
__global__ void GLCMFeaturesP3B_kernel(float* g_cor, float* g_var, float* g_cls, float* g_clp)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int uid = blockIdx.x * (blocksize * 2) + tid;

	// correlation
	{
		float var = *g_var;
		if (var == 0)
		{
			g_cor[0] = 1.0f;
		}
		else
		{
			shared[tid] = g_cor[uid] + g_cor[uid + blocksize];
			__syncthreads();
			SumReduction_dev_kernel<blocksize>(shared);
			if (tid == 0) g_cor[blockIdx.x] = shared[0] / sqrtf(var * var);
		}
	}

	// cluster shade
	{
		shared[tid] = g_cls[uid] + g_cls[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_cls[blockIdx.x] = shared[0];
	}

	// cluster prominence
	{
		shared[tid] = g_clp[uid] + g_clp[uid + blocksize];
		__syncthreads();
		SumReduction_dev_kernel<blocksize>(shared);
		if (tid == 0) g_clp[blockIdx.x] = shared[0];
	}
}


void GLCMgpu_CalculateFeatures(GLCMInfo &gi, float* time)
{
	int depth = gi.depth;
	float* d_glcm = gi.d_glcm;

	int diagRange = 2 * depth - 1;

	float* d_ux = 0;
	float* d_dis = 0;
	float* d_con = 0;
	float* d_idm = 0;
	float* d_ent = 0;
	float* d_asm = 0;
	float* d_map = 0;
	float* d_mip = 0;
	float* d_pxpy = 0;
	float* d_pxmy = 0;

	float* d_sen = 0;
	float* d_sav = 0;
	float* d_sva = 0;
	float* d_den = 0;
	float* d_dva = 0;
	float* d_var = 0;
	float* d_sdx = 0;

	float* d_cor = 0;
	float* d_cls = 0;
	float* d_clp = 0;

	float time1, time2, time3;

	cudaEvent_t start = 0, stop = 0;

	checkCudaErrors(cudaEventCreate(&start, 0));
	checkCudaErrors(cudaEventCreate(&stop, 0));

	checkCudaErrors(cudaMalloc(&d_ux, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_dis, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_con, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_idm, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_ent, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_asm, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_map, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_mip, sizeof(float) * (depth/2)));
	checkCudaErrors(cudaMalloc(&d_pxpy, sizeof(float) * (depth*2)));
	checkCudaErrors(cudaMemset(d_pxpy, 0.0f, (depth*2)));
	checkCudaErrors(cudaMalloc(&d_pxmy, sizeof(float) * depth));

	checkCudaErrors(cudaMalloc(&d_sen, sizeof(float)* diagRange));
	checkCudaErrors(cudaMalloc(&d_sav, sizeof(float)* depth));
	checkCudaErrors(cudaMalloc(&d_sva, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_den, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_dva, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_var, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_sdx, sizeof(float)));

	checkCudaErrors(cudaMalloc(&d_cor, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_cls, sizeof(float)* (depth/2)));
	checkCudaErrors(cudaMalloc(&d_clp, sizeof(float)* (depth/2)));


	switch (depth)
	{
	case 256:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<256><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<64><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<256><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<256><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<256><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<128><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<256><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<64><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<256><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<64><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	case 128:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<128><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<32><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<128><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<128><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<128><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<64><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<128><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<32><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<128><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<32><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	case 64:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<64><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<16><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<64><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<64><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<64><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<32><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<64><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<16><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<64><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<16><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	case 32:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<32><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<8><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<32><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<32><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<32><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<16><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<32><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<8><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<32><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<8><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	case 16:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<16><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<4><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<16><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<16><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<16><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<8><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<16><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<4><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<16><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<4><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	case 8:
		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP1A_kernel<8><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1B_kernel<2><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_ux, d_dis, d_con, d_idm, d_ent, d_asm, d_map, d_mip);
		GLCMFeaturesP1C_kernel<8><<<diagRange, depth, sizeof(float) * depth>>>(d_glcm, d_pxpy);
		GLCMFeaturesP1D_kernel<8><<<depth, depth, sizeof(float) * depth>>>(d_glcm, d_pxmy);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time1, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP2A_kernel<8><<<1, depth, sizeof(float) * depth>>>(d_pxpy, d_sen, d_sav, d_sva);
		GLCMFeaturesP2B_kernel<4><<<1, depth/2, sizeof(float) * (depth/2)>>>(d_pxmy, d_den, d_dva);
		GLCMFeaturesP2C_kernel<8><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_var);
		GLCMFeaturesP2D_kernel<2><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_var, d_sdx);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time2, start, stop));

		checkCudaErrors(cudaEventRecord(start, 0));

		GLCMFeaturesP3A_kernel<8><<<depth/2, depth, sizeof(float) * depth>>>(d_glcm, d_ux, d_cor, d_cls, d_clp);
		GLCMFeaturesP3B_kernel<2><<<1, depth/4, sizeof(float) * (depth/4)>>>(d_cor, d_var, d_cls, d_clp);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&time3, start, stop));

		break;
	default:
		// handle?
		break;
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	printf("time elapsed in p1: %f\n", time1);
	printf("time elapsed in p2: %f\n", time2);
	printf("time elapsed in p2: %f\n", time3);

	checkCudaErrors(cudaMemcpy(&(gi.h_ux), &d_ux[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_dis), &d_dis[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_con), &d_con[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_idm), &d_idm[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_ent), &d_ent[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_asm), &d_asm[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_map), &d_map[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_mip), &d_mip[0], sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&(gi.h_sen), &d_sen[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_sav), &d_sav[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_sva), &d_sva[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_den), &d_den[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_dva), &d_dva[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_var), &d_var[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_sdx), &d_sdx[0], sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&(gi.h_cor), &d_cor[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_cls), &d_cls[0], sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&(gi.h_clp), &d_clp[0], sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_ux);
	cudaFree(d_dis);
	cudaFree(d_con);
	cudaFree(d_idm);
	cudaFree(d_ent);
	cudaFree(d_asm);
	cudaFree(d_map);
	cudaFree(d_mip);
	cudaFree(d_pxpy);
	cudaFree(d_pxmy);

	cudaFree(d_sen);
	cudaFree(d_sav);
	cudaFree(d_sva);
	cudaFree(d_den);
	cudaFree(d_dva);
	cudaFree(d_var);
	cudaFree(d_sdx);

	cudaFree(d_cor);
	cudaFree(d_cls);
	cudaFree(d_clp);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
}