
#include "GLCMCalculationCPU.h"

unsigned char* BitmapToIntensityCPU(BitmapData* bmdata)
{
	const float size = bmdata->bmi->bmiHeader.biHeight * bmdata->bmi->bmiHeader.biWidth;
	unsigned char* intensity = (unsigned char*)malloc(sizeof(unsigned char) * size);

	for (int k = 0; k < size; k++)
	{
		unsigned char rgbBlue	= bmdata->pixels[k * 4 + 0];
		unsigned char rgbGreen	= bmdata->pixels[k * 4 + 1];
		unsigned char rgbRed	= bmdata->pixels[k * 4 + 2];

		// LUMA Coding standard conversion: Y' = 0.299R' + 0.587G' + 0.114B'
		intensity[k] = (0.299 * rgbRed) + (0.587 * rgbGreen) + (0.114 * rgbBlue);
	}

	return intensity;
}

unsigned char* IntensityScaleDepthCPU(unsigned char* intensity, const int size, const int currentDepth, const int newDepth)
{
	const float scaleFactor = currentDepth / newDepth;

	unsigned char* newIntensity = (unsigned char*)malloc(sizeof(unsigned char) * size);

	for (int k = 0; k < size; k++)
	{
		newIntensity[k] = (int)((float)intensity[k] / scaleFactor);
	}

	return newIntensity;
}

void GLCMcpu_InitializeMemory(GLCMInfoCPU &gi, unsigned char* in_intensity)
{
	gi.intensity = in_intensity;
	gi.glcm = (float*)calloc(gi.depth * gi.depth, sizeof(float));

}

void GLCMcpu_FreeMemory(GLCMInfoCPU &gi)
{
	free(gi.glcm);
}

void PrintIntensityValues(const unsigned char* intensity, const int rows, const int cols)
{
	printf("\n");

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int stride = (r * cols + c);
			printf("%d ", intensity[stride]);
		}
		printf("\n");
	}
}

void GLCMcpu_PrintFeatureSet(GLCMFeatureSetCPU &set)
{
	printf("\nPhase 1 Features Serial\n\n");
	printf("%30s: %.20f\n", "Mean", set.s_ux);
	printf("%30s: %.20f\n", "Dissimilarity", set.s_dis);
	printf("%30s: %.20f\n", "Contrast", set.s_con);
	printf("%30s: %.20f\n", "Inverse Difference Momentum", set.s_idm);
	printf("%30s: %.20f\n", "Entropy", set.s_ent);
	printf("%30s: %.20f\n", "Angular Second Momentum", set.s_asm);
	printf("%30s: %.20f\n", "Maximum Probability", set.s_map);
	printf("%30s: %.20f\n", "Minimum Probability", set.s_mip);

	printf("\nPhase 2 Features Serial\n\n");
	printf("%30s: %.20f\n", "Sum Entropy", set.s_sen);
	printf("%30s: %.20f\n", "Difference Entropy", set.s_den);
	printf("%30s: %.20f\n", "Sum Average", set.s_sav);
	printf("%30s: %.20f\n", "Sum Variance", set.s_sva);
	printf("%30s: %.20f\n", "Difference Variance", set.s_dva);
	printf("%30s: %.20f\n", "Variance", set.s_var);
	printf("%30s: %.20f\n", "Standard Deviation", sqrt(set.s_var));

	printf("\nPhase 3 Features Serial\n\n");
	printf("%30s: %.20f\n", "Correlation", set.s_cor);
	printf("%30s: %.20f\n", "Cluster Shade", set.s_cls);
	printf("%30s: %.20f\n", "Cluster Prominance", set.s_clp);
	printf("\n");
}

/* adds the values of the features in set1 to the values of the features in set0
*/

void GLCMcpu_AddFeatureSet(GLCMFeatureSetCPU &dest, GLCMFeatureSetCPU &src)
{
	/* phase 1 vars */
	dest.s_dis += src.s_dis;
	dest.s_con += src.s_con;
	dest.s_idm += src.s_idm;
	dest.s_ent += src.s_ent;
	dest.s_asm += src.s_asm;
	dest.s_map += src.s_map;
	dest.s_mip += src.s_mip;
	dest.s_ux += src.s_ux;

	/* phase 2 vars */
	dest.s_sen += src.s_sen;
	dest.s_den += src.s_den;
	dest.s_sav += src.s_sav;
	dest.s_dva += src.s_dva;
	dest.s_var += src.s_var;
	dest.s_sva += src.s_sva;

	/* phase 3 vars */
	dest.s_cor += src.s_cor;
	dest.s_cls += src.s_cls;
	dest.s_clp += src.s_clp;
}

/* divides the values of the features in set by scalar
*/

void GLCMcpu_ScaleFeatureSet(GLCMFeatureSetCPU &set, const float scalar)
{
	/* phase 1 vars */
	set.s_dis /= scalar;
	set.s_con /= scalar;
	set.s_idm /= scalar;
	set.s_ent /= scalar;
	set.s_asm /= scalar;
	set.s_map /= scalar;
	set.s_mip /= scalar;
	set.s_ux /= scalar;

	/* phase 2 vars */
	set.s_sen /= scalar;
	set.s_den /= scalar;
	set.s_sav /= scalar;
	set.s_dva /= scalar;
	set.s_var /= scalar;
	set.s_sva /= scalar;

	/* phase 3 vars */
	set.s_cor /= scalar;
	set.s_cls /= scalar;
	set.s_clp /= scalar;
}