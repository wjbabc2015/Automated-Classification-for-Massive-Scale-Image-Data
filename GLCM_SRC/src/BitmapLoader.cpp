#include <Windows.h>
#include <malloc.h>

#include "BitmapLoader.h"

void PrintRGBChannels(BitmapData* bmdata)
{
	printf("...\n");
	const int offset = 4;
	const unsigned char *pSource = (const unsigned char*)bmdata->pixels;

	for (int r = 0; r < bmdata->bmi->bmiHeader.biHeight; r++)
	{
		for (int c = 0; c < bmdata->bmi->bmiHeader.biWidth; c++)
		{
			int stride = (r * bmdata->bmi->bmiHeader.biWidth + c) * offset;
			unsigned char rgbBlue = pSource[stride];
			unsigned char rgbGreen = pSource[stride + 1];
			unsigned char rgbRed = pSource[stride + 2];
			printf("(%3d,%3d,%3d) ", rgbRed, rgbGreen, rgbBlue);
		}
		printf("\n");
	}
	printf("\n");
}

int GetBytesPerPixel(int depth)
{
	return (depth == 32 ? 4 : 3);
}

int GetBytesPerRow(int width, int depth)
{
	int bytesPerPixel = GetBytesPerPixel(depth);
	int bytesPerRow = ((width * bytesPerPixel + 3) & ~3);

	return bytesPerRow;
}

int GetBitmapBytes(int width, int height, int depth)
{
	return height * GetBytesPerRow(width, depth);
}

int GetBitmapBytes(const BITMAPINFOHEADER *bmih)
{
	return GetBitmapBytes(bmih->biWidth, bmih->biHeight, bmih->biBitCount);
}

void GetBitmapData(BitmapData*& bmdata, HDC hdc, HBITMAP hbm, BITMAP bm)
{
	BITMAPINFO* bmi = (BITMAPINFO *)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD));
	memset(&bmi->bmiHeader, 0, sizeof(BITMAPINFOHEADER));
	bmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	int scanLineCount = GetDIBits(hdc, hbm, 0, bm.bmHeight, NULL, bmi, DIB_RGB_COLORS);

	int imageBytes = GetBitmapBytes(&bmi->bmiHeader);
	char *pSourceBits = (char *)malloc(imageBytes);

	if (0 == GetDIBits(hdc, hbm, 0, bm.bmHeight, pSourceBits, bmi, DIB_RGB_COLORS)) 
	{
		printf("Error loading bitmap! Exiting.\n");
		exit(1);
	}

	bmdata->bmi = bmi;
	bmdata->pixels = pSourceBits;
}

void LoadBitmapHandle(HBITMAP &hbm, const LPCSTR imagepath)
{
	hbm = (HBITMAP)LoadImage(NULL, imagepath, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
	if (hbm == NULL)
	{
		printf("Could not load bitmap! Exiting.\n");
		exit(1);
	}
}

BitmapData* LoadBitmapData(const LPCSTR imagepath)
{
	BitmapData* bmdata = new BitmapData();

	HBITMAP hbm = NULL;
	LoadBitmapHandle(hbm, imagepath);

	BITMAP bm;
	GetObject(hbm, sizeof(bm), &bm);

	HDC hdc = GetDC(NULL);
	GetBitmapData(bmdata, hdc, hbm, bm);

	DeleteObject(hbm);
	ReleaseDC(NULL, hdc);
	return bmdata;
}