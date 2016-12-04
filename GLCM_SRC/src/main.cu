#include <Windows.h>
#include <WinUser.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include "GLCMCalculationCPU.h"
#include "GLCMCalculationGPU.cuh"
#include "BitmapLoader.h"

#include "HRTimer.h"

const LPCSTR IMAGE = "bm1.bmp";

//int main(int argc, char** argv)
//{
//	BitmapData* bmdata = LoadBitmapData(IMAGE);
//	unsigned char* d_intensity = BitmapToIntensityGPU(bmdata);
//	int rows = bmdata->bmi->bmiHeader.biHeight;
//	int cols = bmdata->bmi->bmiHeader.biWidth;
//
//	unsigned char* d_intensityD8 = IntensityScaleDepthGPU(d_intensity, rows, cols, 256, 8);
//
//	//float* glcmNorm;
//	//GLCMGetNormalizedMatrix(glcminfo, glcmNorm);
//	//PrintGLCM(glcmNorm, glcminfo->depth);
//	//free(glcmNorm);
//
//	for (int i = 0; i < 10; i++) 
//	{
//		GLCMInfo* glcminfo = new GLCMInfo(rows, cols, 256, 0, 1);
//		GLCMInitializeMemory(glcminfo, d_intensity);
//		GLCMCalculateGPU(glcminfo);
//
//		float time;
//		GLCMCalculateFeaturesGPU(glcminfo, &time);
//		GLCMPrintFeatures(glcminfo);
//
//		//GLCMFreeGPUMemory(glcminfo);
//	}
//
//	cudaFree(d_intensity);
//	cudaFree(d_intensityD8);
//}

void ComputeOnCPU(unsigned char* intensity, int rows, int cols, int depth)
{
	stopWatch timer;
	GLCMInfoCPU c_gi(rows, cols, depth, 0, 1);
	GLCMcpu_InitializeMemory(c_gi, intensity);
	GLCMcpu_CalculateGLCM(c_gi);
	GLCMcpu_NormalizeGLCM(c_gi);
	GLCMcpu_CalculateFeatures(c_gi);
	printf("\n(CPU) Features for Depth %d\n", depth);
	GLCMcpu_PrintFeatureSet(c_gi.F);
	GLCMcpu_FreeMemory(c_gi);	
	printf("\n ************************ \n");
}

void ComputeOnGPU(unsigned char* intensity, int rows, int cols, int depth)
{
	float timer;
	GLCMInfo g_gi(rows, cols, depth, 0, 1);
	GLCMgpu_InitializeMemory(g_gi, intensity);
	GLCMgpu_CalculateGLCM(g_gi);
	GLCMgpu_NormalizeGLCM(g_gi);
	GLCMgpu_CalculateFeatures(g_gi, &timer);
	printf("\n(GPU) Features for Depth %d\n", depth);
	GLCMgpu_PrintFeatures(g_gi);
	GLCMgpu_FreeMemory(g_gi);
	printf("\n ---- \n");
}

int main(int argc, char** argv)
{
	BitmapData* bmdata = LoadBitmapData(IMAGE);
	int rows = bmdata->bmi->bmiHeader.biHeight;
	int cols = bmdata->bmi->bmiHeader.biWidth;

	printf("\nwidth: %d\nheight: %d\nbits per pixel: %d\n\n", bmdata->bmi->bmiHeader.biHeight, bmdata->bmi->bmiHeader.biWidth, bmdata->bmi->bmiHeader.biBitCount);

	unsigned char* intensityD256 = BitmapToIntensityCPU(bmdata);
	unsigned char* intensityD128 = IntensityScaleDepthCPU(intensityD256, rows * cols, 256, 128);
	unsigned char* intensityD64  = IntensityScaleDepthCPU(intensityD256, rows * cols, 256, 64);
	unsigned char* intensityD32  = IntensityScaleDepthCPU(intensityD256, rows * cols, 256, 32);
	unsigned char* intensityD16  = IntensityScaleDepthCPU(intensityD256, rows * cols, 256, 16);
	unsigned char* intensityD8   = IntensityScaleDepthCPU(intensityD256, rows * cols, 256, 8);

	ComputeOnGPU(intensityD256, rows, cols, 256);
	ComputeOnCPU(intensityD256, rows, cols, 256);
	ComputeOnGPU(intensityD128, rows, cols, 128);
	ComputeOnCPU(intensityD128, rows, cols, 128);
	ComputeOnGPU(intensityD64, rows, cols, 64);
	ComputeOnCPU(intensityD64, rows, cols, 64);
	ComputeOnGPU(intensityD32, rows, cols, 32);
	ComputeOnCPU(intensityD32, rows, cols, 32);
	ComputeOnGPU(intensityD16, rows, cols, 16);
	ComputeOnCPU(intensityD16, rows, cols, 16);
	ComputeOnGPU(intensityD8, rows, cols, 8);
	ComputeOnCPU(intensityD8, rows, cols, 8);

	free(intensityD256);
	free(intensityD128);
	free(intensityD64);
	free(intensityD32);
	free(intensityD16);
	free(intensityD8);
	delete(bmdata);

	return 0;
}