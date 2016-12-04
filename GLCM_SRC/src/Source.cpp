#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <stdio.h>
#include <malloc.h>
#include <tchar.h>
#include <strsafe.h>
#pragma comment(lib, "User32.lib")

#include "GLCMCalculationCPU.h"
#include "BitmapLoader.h"
#include "HRTimer.h"

const int DEPTH = 256;
const int DISTANCE = 1;

char directory[MAX_PATH];
char* subdirectories[] = { "\\45deg\\camera0", "\\45deg\\camera1",
							"\\hor\\camera0", "\\hor\\camera1",
							"\\ver\\camera0", "\\ver\\camera1" };

void WriteCSVHeader(FILE* fp)
{
	fprintf(fp, "s-IMAGE,p-IMAGE,");
	fprintf(fp, "s-COR, s-DIS, s-CON, s-IDM, s-ENT, s-SEN, s-DEN, s-ASM, s-VAR, s-SVA, s-DVA, s-MEA, s-SAV, s-CLS, s-CLP, s-MAP, s-MIP, s-IMIN, s-IMAX, s-MEA,");
	fprintf(fp, "p-COR, p-DIS, p-CON, p-IDM, p-ENT, p-SEN, p-DEN, p-ASM, p-VAR, p-SVA, p-DVA, p-MEA, p-SAV, p-CLS, p-CLP, p-MAP, p-MIP, p-IMIN, p-IMAX, p-MEA,\n");
}

void CoImage_FeatureToCSV(GLCMFeatureSetCPU &set_s, GLCMFeatureSetCPU &set_p, 
	const char* filename_s, const char* filename_p, FILE *fp)
{
	fprintf(fp, "%s, ", filename_s);
	fprintf(fp, "%s, ", filename_p);
	fprintf(fp, "%.17f, ", set_s.s_cor);
	fprintf(fp, "%.17f, ", set_s.s_dis);
	fprintf(fp, "%.17f, ", set_s.s_con);
	fprintf(fp, "%.17f, ", set_s.s_idm);
	fprintf(fp, "%.17f, ", set_s.s_ent);
	fprintf(fp, "%.17f, ", set_s.s_sen);
	fprintf(fp, "%.17f, ", set_s.s_den);
	fprintf(fp, "%.17f, ", set_s.s_asm);
	fprintf(fp, "%.17f, ", set_s.s_var);
	fprintf(fp, "%.17f, ", set_s.s_sva);
	fprintf(fp, "%.17f, ", set_s.s_dva);
	fprintf(fp, "%.17f, ", set_s.s_ux);
	fprintf(fp, "%.17f, ", set_s.s_sav);
	fprintf(fp, "%.17f, ", set_s.s_cls);
	fprintf(fp, "%.17f, ", set_s.s_clp);
	fprintf(fp, "%.17f, ", set_s.s_map);
	fprintf(fp, "%.17f, ", set_s.s_mip);
	fprintf(fp, ", ");
	fprintf(fp, ", ");
	fprintf(fp, ", ");	
	fprintf(fp, "%.17f, ", set_p.s_cor);
	fprintf(fp, "%.17f, ", set_p.s_dis);
	fprintf(fp, "%.17f, ", set_p.s_con);
	fprintf(fp, "%.17f, ", set_p.s_idm);
	fprintf(fp, "%.17f, ", set_p.s_ent);
	fprintf(fp, "%.17f, ", set_p.s_sen);
	fprintf(fp, "%.17f, ", set_p.s_den);
	fprintf(fp, "%.17f, ", set_p.s_asm);
	fprintf(fp, "%.17f, ", set_p.s_var);
	fprintf(fp, "%.17f, ", set_p.s_sva);
	fprintf(fp, "%.17f, ", set_p.s_dva);
	fprintf(fp, "%.17f, ", set_p.s_ux);
	fprintf(fp, "%.17f, ", set_p.s_sav);
	fprintf(fp, "%.17f, ", set_p.s_cls);
	fprintf(fp, "%.17f, ", set_p.s_clp);
	fprintf(fp, "%.17f, ", set_p.s_map);
	fprintf(fp, "%.17f, ", set_p.s_mip);
	fprintf(fp, ", ");
	fprintf(fp, ", ");
	fprintf(fp, ", ");
	fprintf(fp, "\n");
}

void CoImage_AllDirections(const char* path_s, const char* filename_s, const char* path_p, 
	const char* filename_p, int depth, int distance, FILE** fpp)
{
	//printf("Going to load: %s & %s\n", path_s, path_p);

	BitmapData* bitmap_s = LoadBitmapData(path_s);
	BitmapData* bitmap_p = LoadBitmapData(path_p);
	unsigned char* intensity_s = BitmapToIntensityCPU(bitmap_s);
	unsigned char* intensity_p = BitmapToIntensityCPU(bitmap_p);
	int rows_s = bitmap_s->bmi->bmiHeader.biHeight;
	int rows_p = bitmap_p->bmi->bmiHeader.biHeight;
	int cols_s = bitmap_s->bmi->bmiHeader.biWidth;
	int cols_p = bitmap_p->bmi->bmiHeader.biWidth;

	GLCMFeatureSetCPU avgfeatures_s;
	GLCMFeatureSetCPU avgfeatures_p;

	int j = 0;
	for (int i = 0; i <= 135; i += 45)
	{
		GLCMInfoCPU glcminfo_s(rows_s, cols_s, depth, i, distance);
		GLCMcpu_InitializeMemory(glcminfo_s, intensity_s);
		GLCMcpu_CalculateGLCM(glcminfo_s);
		GLCMcpu_NormalizeGLCM(glcminfo_s);
		GLCMcpu_CalculateFeatures(glcminfo_s);
		GLCMcpu_AddFeatureSet(avgfeatures_s, glcminfo_s.F);

		GLCMInfoCPU glcminfo_p(rows_p, cols_p, depth, i, distance);
		GLCMcpu_InitializeMemory(glcminfo_p, intensity_p);
		GLCMcpu_CalculateGLCM(glcminfo_p);
		GLCMcpu_NormalizeGLCM(glcminfo_p);
		GLCMcpu_CalculateFeatures(glcminfo_p);
		GLCMcpu_AddFeatureSet(avgfeatures_p, glcminfo_p.F);

		CoImage_FeatureToCSV(glcminfo_s.F, glcminfo_p.F, filename_s, filename_p, fpp[j]);

		GLCMcpu_FreeMemory(glcminfo_s);
		GLCMcpu_FreeMemory(glcminfo_p);
		j++;
	}

	GLCMcpu_ScaleFeatureSet(avgfeatures_s, 4.0f);
	GLCMcpu_ScaleFeatureSet(avgfeatures_p, 4.0f);
	CoImage_FeatureToCSV(avgfeatures_s, avgfeatures_p, filename_s, filename_p, fpp[4]);

	free(intensity_s);
	free(intensity_p);
	delete(bitmap_s);
	delete(bitmap_p);
}

void ProcessSubdirectories(char* fileprefix)
{
	WIN32_FIND_DATA ffd0;
	WIN32_FIND_DATA ffd1;
	LARGE_INTEGER filesize;

	char path0[MAX_PATH];
	char path1[MAX_PATH];

	char absolutename0[MAX_PATH];
	char absolutename1[MAX_PATH];

	char searchkey0[MAX_PATH];
	char searchkey1[MAX_PATH];

	char filename[5][64];
	FILE* fpp[5];

	HANDLE hFind0 = INVALID_HANDLE_VALUE;
	HANDLE hFind1 = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;

	for (int i = 0; i < 6; i += 2)
	{
		for (int j = 0; j < 5; j++)
		{
			strcpy(filename[j], fileprefix);
			if (subdirectories[i][1] == '4')
				strcat(filename[j], "45deg_");
			else if (subdirectories[i][1] == 'h')
				strcat(filename[j], "hor_");
			else
				strcat(filename[j], "ver_");
		}

		strcat(filename[0], "0.csv");
		strcat(filename[1], "45.csv");
		strcat(filename[2], "90.csv");
		strcat(filename[3], "135.csv");
		strcat(filename[4], "avg.csv");

		for (int j = 0; j < 5; j++)
		{
			fpp[j] = fopen(filename[j], "w");
			if (fpp[j] == NULL)
			{
				fprintf(stderr, "Error opening %s for writing! Exiting.\n", filename[j]);
				exit(1);
			}
			WriteCSVHeader(fpp[j]);
		}

		strcpy(path0, directory);
		strcat(path0, subdirectories[i]);
		strcpy(searchkey0, path0);
		strcat(searchkey0, "\\*.bmp");
		strcpy(path1, directory);
		strcat(path1, subdirectories[i + 1]);
		strcpy(searchkey1, path1);
		strcat(searchkey1, "\\*.bmp");


		hFind0 = FindFirstFile(searchkey0, &ffd0);
		hFind1 = FindFirstFile(searchkey1, &ffd1);

		printf("\nCalculating features for subdirectories %s & %s and writing into files:\n\n", subdirectories[i], subdirectories[i + 1]);
		for (int j = 0; j < 5; j++)
			printf("  %s\n", filename[j]);

		stopWatch timer;
		startTimer(&timer);

		int count = 0;
		do
		{
			count += 2;
			strcpy(absolutename0, path0);
			strcat(absolutename0, "\\");
			strcat(absolutename0, ffd0.cFileName);
			strcpy(absolutename1, path1);
			strcat(absolutename1, "\\");
			strcat(absolutename1, ffd1.cFileName);

			CoImage_AllDirections(absolutename0, ffd0.cFileName, absolutename1, ffd1.cFileName, DEPTH, DISTANCE, fpp);
		} while ((FindNextFile(hFind0, &ffd0) != 0) && (FindNextFile(hFind1, &ffd1) != 0));

		dwError = GetLastError();
		if (dwError != ERROR_NO_MORE_FILES)
			printf("\nRead Error! Exiting.\n");

		stopTimer(&timer);
		printf("\n...Calculated features for %d images in %lf seconds.\n", count, getElapsedTimeSecs(&timer));

		FindClose(hFind0);
		FindClose(hFind1);
		
		for (int j = 0; j < 5; j++)
			fclose(fpp[j]);
	}
}

int main(int argc, char** argv)
{
	stopWatch timer;
	startTimer(&timer);

	SYSTEMTIME lt;
	GetLocalTime(&lt);

	char fileprefix[100];
	sprintf_s(fileprefix, 100, "%d_%d_%d__%02d%02d_", lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute);

	printf("\nChecking existence of target directory...\n\n");
	FILE* ifp = fopen("directory.cfg", "r");
	if (ifp == NULL) 
	{
		fprintf(stderr, "  Can't open directory.cfg! Exiting.\n");
		return 1;
	}

	fscanf_s(ifp, "%[^\n]%*c", directory, MAX_PATH);
	fclose(ifp);

	if (strlen(directory) > (MAX_PATH - 22))
	{
		printf("  Directory path is too long! Exiting.\n");
		return 1;
	}
	printf("  \"%s\"\n", directory);
	printf("\n...target directory exists.\n");

	char dir0[MAX_PATH];
	char dir1[MAX_PATH];

	printf("\nChecking existence of required subdirectories...\n\n");
	for (int i = 0; i < 6; i += 2)
	{
		WIN32_FIND_DATA ffd0;
		WIN32_FIND_DATA ffd1;

		HANDLE hFind0 = INVALID_HANDLE_VALUE;
		HANDLE hFind1 = INVALID_HANDLE_VALUE;

		strcpy(dir0, directory);
		strcat(dir0, subdirectories[i]);
		strcat(dir0, "\\*.bmp");
		strcpy(dir1, directory);
		strcat(dir1, subdirectories[i + 1]);
		strcat(dir1, "\\*.bmp");

		// Find the first file in the directory.

		hFind0 = FindFirstFile(dir0, &ffd0);

		if (INVALID_HANDLE_VALUE == hFind0)
		{
			printf("  %s unreadable! Exiting.\n", subdirectories[i]);
			return 1;
		}

		printf("  %s detected\n", subdirectories[i]);

		hFind1 = FindFirstFile(dir1, &ffd1);

		if (INVALID_HANDLE_VALUE == hFind1)
		{
			printf("  %s unreadable! Exiting.\n", subdirectories[i + 1]);
			return 1;
		}

		printf("  %s detected\n", subdirectories[i + 1]);

		FindClose(hFind0);
		FindClose(hFind1);
	}
	printf("\n...required subdirectories exist.\n\n");

	ProcessSubdirectories(fileprefix);

	stopTimer(&timer);
	printf("\nGLCM Calc Tool finished execution in %lf seconds.\n", getElapsedTimeSecs(&timer));

	return 0;
}