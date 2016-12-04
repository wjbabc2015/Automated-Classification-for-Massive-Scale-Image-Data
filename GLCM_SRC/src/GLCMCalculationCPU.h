#include <malloc.h>
#include <stdio.h>
#include <math.h>

#include "BitmapLoader.h"
#include "HRTimer.h"

struct GLCMFeatureSetCPU {
	/* phase 1 vars */
	float s_dis = 0.0f;
	float s_con = 0.0f;
	float s_idm = 0.0f;
	float s_ent = 0.0f;
	float s_asm = 0.0f;
	float s_map = 0.0f;
	float s_mip = 0.0f;
	float s_ux = 0.0f;

	/* phase 2 vars */
	float s_sen = 0.0f;
	float s_den = 0.0f;
	float s_sav = 0.0f;
	float s_dva = 0.0f;
	float s_var = 0.0f;
	float s_sva = 0.0f;

	/* phase 3 vars */
	float s_cor = 0.0f;
	float s_cls = 0.0f;
	float s_clp = 0.0f;

	GLCMFeatureSetCPU() {}
};

struct GLCMInfoCPU {
	unsigned char* intensity = 0;
	float* glcm = 0;

	int rows;
	int cols;
	int depth;
	int angle;
	int distance;
	int R;

	GLCMFeatureSetCPU F;

	GLCMInfoCPU(int in_rows, int in_cols, int in_depth, int in_angle, int in_distance)
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

void GLCMcpu_CalculateGLCM(GLCMInfoCPU &gi);

void GLCMcpu_NormalizeGLCM(GLCMInfoCPU &gi);

void PrintIntensityValues(const unsigned char* intensity, const int rows, const int cols);

void GLCMcpu_CalculateFeatures(GLCMInfoCPU &gi);

unsigned char* BitmapToIntensityCPU(BitmapData* bmdata);

unsigned char* IntensityScaleDepthCPU(unsigned char* intensity, const int size, const int currentDepth, const int newDepth);

void GLCMcpu_InitializeMemory(GLCMInfoCPU &gi, unsigned char* in_intensity);

void GLCMcpu_FreeMemory(GLCMInfoCPU &gi);

void GLCMcpu_AddFeatureSet(GLCMFeatureSetCPU &dest, GLCMFeatureSetCPU &src);

void GLCMcpu_ScaleFeatureSet(GLCMFeatureSetCPU &set, const float scalar);

void GLCMcpu_PrintFeatureSet(GLCMFeatureSetCPU &set);

float* GLCMDiagSerial(unsigned char* intensity, const int rows, const int cols, const int depth);

float* GLCMAntiDiagSerial(unsigned char* intensity, const int rows, const int cols, const int depth);

float* GLCMVerticalSerial(unsigned char* intensity, const int rows, const int cols, const int depth);

float* GLCMHorizontalSerial(unsigned char* intensity, const int rows, const int cols, const int depth);

float* GLCMNormalizeSerial(const float* glcm, const int depth, const int R);