
#include "GLCMCalculationCPU.h"

float* GLCMNormalizeSerial(const float* glcm, const int depth, const int R)
{
	float* glcmNorm = 0;
	glcmNorm = (float*)malloc(sizeof(float) * depth * depth);

	for (int i = 0; i < depth * depth; i++)
	{
		glcmNorm[i] = glcm[i] / R;
	}

	return glcmNorm;
}

float* GLCMHorizontalSerial(unsigned char* intensity, const int rows, const int cols, const int depth)
{
	float* glcm = (float*)calloc(depth * depth, sizeof(float));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols - 1; j++)
		{
			unsigned char x = intensity[i * cols + j];
			unsigned char y = intensity[i * cols + j + 1];
			glcm[x * depth + y] += 1;
			glcm[y * depth + x] += 1;
		}
	}

	return glcm;
}

float* GLCMVerticalSerial(unsigned char* intensity, const int rows, const int cols, const int depth)
{
	float* glcm = (float*)calloc(depth * depth, sizeof(float));

	for (int i = 1; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			unsigned char x = intensity[i * cols + j];
			unsigned char y = intensity[(i - 1) * cols + j];
			glcm[x * depth + y] += 1;
			glcm[y * depth + x] += 1;
		}
	}

	return glcm;
}

float* GLCMAntiDiagSerial(unsigned char* intensity, const int rows, const int cols, const int depth)
{
	float* glcm = (float*)calloc(depth * depth, sizeof(float));

	for (int i = 1; i < rows; i++)
	{
		for (int j = 1; j < cols; j++)
		{
			unsigned char x = intensity[i * cols + j];
			unsigned char y = intensity[(i - 1) * cols + j - 1];
			glcm[x * depth + y] += 1;
			glcm[y * depth + x] += 1;
		}
	}

	return glcm;
}

float* GLCMDiagSerial(unsigned char* intensity, const int rows, const int cols, const int depth)
{
	float* glcm = (float*)calloc(depth * depth, sizeof(float));

	for (int i = 1; i < rows; i++)
	{
		for (int j = 0; j < cols - 1; j++)
		{
			unsigned char x = intensity[i * cols + j];
			unsigned char y = intensity[(i - 1) * cols + j + 1];
			glcm[x * depth + y] += 1;
			glcm[y * depth + x] += 1;
		}
	}

	return glcm;
}

void GLCMcpu_NormalizeGLCM(GLCMInfoCPU &gi)
{
	int glcmsize = gi.depth * gi.depth;

	for (int i = 0; i < glcmsize; i++)
		gi.glcm[i] /= gi.R;
}

void GLCMcpu_CalculateGLCM(GLCMInfoCPU &gi)
{
	int angle = gi.angle;
	int distance = gi.distance;
	int cols = gi.cols;
	int rows = gi.rows;
	int depth = gi.depth;

	int istart, jstart, jend, offset;

	if (angle == 0)		{ istart = 0;		 jstart = 0;		jend = cols - distance; offset = distance; }
	if (angle == 45)	{ istart = distance; jstart = 0;		jend = cols - distance; offset = 1 - cols; }
	if (angle == 90)	{ istart = distance; jstart = 0;		jend = cols;			offset = -cols; }
	if (angle == 135)	{ istart = distance; jstart = distance; jend = cols;			offset = -1 - cols; }
	else {} // invalid angle -- handle?

	for (int i = istart; i < rows; i++)
	{
		for (int j = jstart; j < jend; j++)
		{
			unsigned char x = gi.intensity[i * cols + j];
			unsigned char y = gi.intensity[(i * cols + j) + offset];
			gi.glcm[x * depth + y] += 1;
			gi.glcm[y * depth + x] += 1;
		}
	}
}