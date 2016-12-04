#include <malloc.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include "BitmapLoader.h"

#ifndef GLCMCALULATION_CUH
#define GLCMCALULATION_CUH

struct GLCMInfo {
	unsigned char* d_intensity = 0;
	float* d_glcm = 0;

	int rows;
	int cols;
	int depth;
	int angle;
	int distance;
	int R;

	/* p1 features */
	float h_ux = 0;
	float h_dis = 0;
	float h_con = 0;
	float h_idm = 0;
	float h_ent = 0;
	float h_asm = 0;
	float h_map = 0;
	float h_mip = 0;

	/* p2 features */
	float h_sen = 0;
	float h_sav = 0;
	float h_sva = 0;
	float h_den = 0;
	float h_dva = 0;
	float h_var = 0;
	float h_sdx = 0;

	/* p3 features */
	float h_cor = 0;
	float h_cls = 0;
	float h_clp = 0;

	GLCMInfo(const int in_rows, 
		const int in_cols, 
		const int in_depth, 
		const int in_angle, 
		const int in_distance)
	{
		rows = in_rows;
		cols = in_cols;
		depth = in_depth;
		angle = in_angle;
		distance = in_distance;

		if		(angle == 0)	{ R = 2 * rows * (cols - distance); }
		else if (angle == 45)	{ R = 2 * (rows - distance) * (cols - distance); }
		else if (angle == 90)	{ R = 2 * (rows - distance) * cols; }
		else if (angle == 135)	{ R = 2 * (rows - distance) * (cols - distance); } 
		else					{ R = 0; } // invalid angle
	}
};

void GLCMgpu_InitializeMemory(GLCMInfo &gi, unsigned char* in_intensity);

void GLCMgpu_FreeMemory(GLCMInfo &gi);

void GLCMgpu_CalculateGLCM(GLCMInfo &glcminfo);

void GLCMgpu_NormalizeGLCM(GLCMInfo &gi);

void GLCMgpu_GetGLCMMatrixCPU(GLCMInfo &gi, float*& glcmNorm);

void GLCMgpu_CalculateFeatures(GLCMInfo &gi, float* time);

void GLCMgpu_PrintFeatures(GLCMInfo &gi);

unsigned char* IntensityScaleDepthGPU(const unsigned char* d_intensity, const int rows,
	const int cols, const int currentDepth, const int newDepth);

unsigned char* BitmapToIntensityGPU(BitmapData* bmdata);

void CompareGLCMs(const float* const glcm1, const float* const glcm2, const int depth);

#endif