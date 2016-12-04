#include <Windows.h>
#include <stdio.h>

#ifndef BITMAPLOADER_H
#define BITMAPLOADER_H

struct BitmapData {
	BITMAPINFO *bmi;
	char       *pixels;

	BitmapData()
	{
		bmi = NULL;
		pixels = NULL;
	}

	~BitmapData()
	{
		free(bmi);
		delete[] pixels;
	}
};

BitmapData* LoadBitmapData(const LPCSTR imagepath);

#endif