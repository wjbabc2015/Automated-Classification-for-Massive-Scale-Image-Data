
#include "GLCMCalculationCPU.h"

void SerialFeaturesP1(const float* const glcm, float* ux, float* dis, float* con, float* idm, float* ent,
	float* a_sm, float* map, float* mip, float* pxpy, float* pxmy, const int depth)
{
	float p_ux = 0.0f;
	float p_dis = 0.0f; 
	float p_con = 0.0f; 
	float p_idm = 0.0f; 
	float p_ent = 0.0f; 
	float p_asm = 0.0f; 
	float p_mip = glcm[0]; 
	float p_map = glcm[0];

	for (int i = 0; i < depth; i++)
	{
		for (int j = 0; j < depth; j++)
		{
			float element = glcm[i * depth + j];

			p_ux += i * element;
			p_dis += abs(i - j) * element;
			float t = (float)(i - j);
			p_con += (t * t) * element;
			p_idm += (1.0f / (1.0f + (t * t))) * element;
			float log_elmt = (element == 0) ? 0.0f : log(element);
			p_ent += element * log_elmt;
			p_asm += element * element;
			if (element > p_map)
				p_map = element;
			if (element < p_mip)
				p_mip = element;

			int k = i + j;
			pxpy[k] += element;
			k = abs(i - j);
			pxmy[k] += element;
		}
	}

	*ux = p_ux;
	*dis = p_dis;
	*con = p_con;
	*idm = p_idm;
	*ent = -p_ent; // sign flipped
	*a_sm = p_asm;
	*mip = p_mip;
	*map = p_map;
}

void SerialFeaturesP2(const float* const glcm, const float* const pxpy, const float* const pxmy, const float ux,
	float* sen, float* den, float* sav, float* dva, float* var, float* sva, const int depth)
{
	float p_sen = 0.0f;
	float p_den = 0.0f;
	float p_sav = 0.0f;
	float p_dva = 0.0f;
	float p_var = 0.0f;
	float p_sva = 0.0f;

	int bound = 2 * depth - 1;

	for (int k = 0; k < bound; k++)
	{
		float element_p = pxpy[k];
		float log_elmt = (element_p == 0) ? 0.0f : log(element_p);
		p_sen += element_p * log_elmt;

		p_sav += ((float)k) * element_p;

		if (k < depth)
		{
			float element_m = pxmy[k];
			float log_elmt = (element_m == 0) ? 0.0f : log(element_m);
			p_den += element_m * log_elmt;

			p_dva += ((float)(k * k)) * element_m;

			for (int j = 0; j < depth; j++)
			{
				int i = k;
				float t = i - ux;
				p_var += (t * t) * glcm[i * depth + j];
			}
		}
	}

	*sen = -p_sen; // sign flipped
	*den = -p_den; // sign flipped
	*sav = p_sav;
	*dva = p_dva;
	*var = p_var;

	for (int k = 0; k < bound; k++)
	{
		float t = ((float)k) - *sen;
		p_sva += (t * t) * pxpy[k];
	}

	*sva = p_sva;
}

void SerialFeaturesP3(const float* const glcm, const float ux, const float var, float* cor, float* cls,
	float* clp, const int depth)
{
	float p_cor = 0.0f;
	float p_cls = 0.0f;
	float p_clp = 0.0f;

	float sd = sqrtf(var * var);

	for (int i = 0; i < depth; i++)
	{
		float iminus_ux = (float)i - ux;

		for (int j = 0; j < depth; j++)
		{
			float element = glcm[i * depth + j];
			float jminus_uy = (float)j - ux;
			p_cor += element * (iminus_ux * jminus_uy);
			float t = ((float)i) + ((float)j) - ux - ux;
			float t_cubed = t * t * t;
			p_cls += (t_cubed)* element;
			p_clp += (t_cubed * t) * element;
		}
	}

	*cor = (sd == 0) ? 0.0f : (p_cor / sd); // check for zero division
	*cls = p_cls;
	*clp = p_clp;
}

void GLCMcpu_CalculateFeatures(GLCMInfoCPU &gi)
{
	//stopWatch timer2;

	float* pxpy = (float*)calloc((2 * gi.depth - 1), sizeof(float));
	float* pxmy = (float*)calloc(gi.depth, sizeof(float));

	/* phase 1 */
	//startTimer(&timer2);
	SerialFeaturesP1(gi.glcm, &(gi.F.s_ux), &(gi.F.s_dis), &(gi.F.s_con), &(gi.F.s_idm), &(gi.F.s_ent), 
		&(gi.F.s_asm), &(gi.F.s_map), &(gi.F.s_mip), pxpy, pxmy, gi.depth);
	//stopTimer(&timer2);
	//printf("Time to Calculate Phase 1 Features: %lf ms\n", getElapsedTimeMillis(&timer2));

	/* phase 2 */
	//startTimer(&timer2);
	SerialFeaturesP2(gi.glcm, pxpy, pxmy, gi.F.s_ux, &(gi.F.s_sen), &(gi.F.s_den), &(gi.F.s_sav), 
		&(gi.F.s_dva), &(gi.F.s_var), &(gi.F.s_sva), gi.depth);
	//stopTimer(&timer2);
	//printf("Time to Calculate Phase 2 Features: %lf ms\n", getElapsedTimeMillis(&timer2));

	/* phase 3 */
	//startTimer(&timer2);
	SerialFeaturesP3(gi.glcm, gi.F.s_ux, gi.F.s_var, &(gi.F.s_cor), &(gi.F.s_cls), &(gi.F.s_clp), gi.depth);
	//stopTimer(&timer2);
	//printf("Time to Calculate Phase 2 Features: %lf ms\n", getElapsedTimeMillis(&timer2));

	free(pxpy);
	free(pxmy);
}