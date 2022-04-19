#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmpheader.h"

int readBmpHeader(FILE *fp, BmpFileHeader *fileHdr, BmpInfoHeader *infoHdr)
{
	// fread(読み込み先変数, 1回あたりの読み込みサイズ, 読み込み回数, ファイルポインタ)
	fread(fileHdr, sizeof(BmpFileHeader), 1, fp);
	fread(infoHdr, sizeof(BmpInfoHeader), 1, fp);

	return 0;
}

int readBmpBody(FILE *fp, Img *bmp)
{
	fread(bmp->pixel, sizeof(BGR), bmp->width * bmp->height, fp);

	return 0;
}

int readBmp(char *filename, Img *bmp, BmpFileHeader *fileHdr, BmpInfoHeader *infoHdr)
{
	// ファイルをバイナリモードで開く
	FILE *fp = fopen(filename, "rb");
	readBmpHeader(fp, fileHdr, infoHdr);

	bmp->width = infoHdr->biWidth;
	bmp->height = infoHdr->biHeight;
	bmp->pixel = (BGR *)malloc(bmp->width * bmp->height * sizeof(BGR));
	readBmpBody(fp, bmp);

	fclose(fp);

	return 0;
}

int writeBmp(char *outputfilename, Img *bmp, BmpFileHeader *fileHdr, BmpInfoHeader *infoHdr)
{
	// ファイルをバイナリモードで開く
	FILE *fp = fopen(outputfilename, "wb");
	// fwrite(書き込み元変数, 1回あたりの書き込みサイズ, 書き込み回数, ファイルポインタ)
	fwrite(fileHdr, sizeof(BmpFileHeader), 1, fp);
	fwrite(infoHdr, sizeof(BmpInfoHeader), 1, fp);
	fwrite(bmp->pixel, sizeof(BGR), bmp->width * bmp->height, fp);
	fclose(fp);

	return 0;
}

/**
 * Color inveriosn process
 *
 * Replace each pixel color (B, G, R) with (255-B, 255-G, 255-R).
 *
 * @param[in/out] bmp Img
 * @return void
 */
void negative(Img *bmp)
{
	for (int j = 0; j < bmp->height; j++)
	{
		for (int i = 0; i < bmp->width; i++)
		{
			int ij = i + bmp->width * j;
			bmp->pixel[ij].B = 255 - bmp->pixel[ij].B;
			bmp->pixel[ij].G = 255 - bmp->pixel[ij].G;
			bmp->pixel[ij].R = 255 - bmp->pixel[ij].R;
		}
	}
}

/**
 * Reverse the geometric position of the image in the Y direction
 *
 * @param[in/out] bmp Img
 * @return void
 */
void yreflect(Img *bmp)
{
	BGR *filtered;
	filtered = (BGR *)malloc(bmp->width * bmp->height * sizeof(BGR));

	for (int j = 0; j < bmp->height; j++)
	{
		for (int i = 0; i < bmp->width; i++)
		{
			int ij = i + bmp->width*j;
			int ijreflected = i + bmp->width * (bmp->height -j -1);
			filtered[ijreflected].B = bmp->pixel[ij].B;
			filtered[ijreflected].G = bmp->pixel[ij].G;
			filtered[ijreflected].R = bmp->pixel[ij].R;
		}
	}
	for (int j = 1; j < (bmp->height - 1); j++)
	{
		for (int i = 1; i < (bmp->width - 1); i++)
		{
			int i__j__ = (i  ) + bmp->width*(j  );
			bmp->pixel[i__j__].B = filtered[i__j__].B;
			bmp->pixel[i__j__].G = filtered[i__j__].G;
			bmp->pixel[i__j__].R = filtered[i__j__].R;
		}
	}
	free(filtered);
}

/**
 * Gray scale
 *
 * Converts full color image to 256 shades of monochrome
 * using the NTSC weighted average method
 * @param[in/out] bmp Img
 * @return void
 */
void gray(Img *bmp)
{
	for (int j = 0; j < bmp->height; j++)
	{
		for (int i = 0; i < bmp->width; i++)
		{
			int ij = i + bmp->width * j;
			unsigned char gray = (unsigned char)( 0.114478f * (float)bmp->pixel[ij].B
												 +0.586611f * (float)bmp->pixel[ij].G
												 +0.298912f * (float)bmp->pixel[ij].R);
			bmp->pixel[ij].B = gray; // ビットmナップファイルフォーマットを変更すれば1色のみ保持する
			bmp->pixel[ij].G = gray; // だけで良くなるが、今回は24bitのまま変更せずに、BGRに同じ値を
			bmp->pixel[ij].R = gray; // 代入する
		}
	}
}

void boxfilter(Img *bmp, float filter[3][3])
{
	BGR *filtered  = (BGR *)malloc(bmp->width * bmp->height * sizeof(BGR));
	int width = bmp->width;

	for (int j = 1; j < bmp->height-1; j++)
	{
		for (int i = 1; i < bmp->width -1; i++)
		{
			int i__j__ = (i  ) + width*(j  );
			int ip1j__ = (i+1) + width*(j  );
			int im1j__ = (i-1) + width*(j  );
			int i__jp1 = (i  ) + width*(j+1);
			int i__jm1 = (i  ) + width*(j-1);
			int ip1jp1 = (i+1) + width*(j+1);
			int im1jp1 = (i-1) + width*(j+1);
			int ip1jm1 = (i+1) + width*(j-1);
			int im1jm1 = (i-1) + width*(j-1);

			filtered[i__j__].B = (unsigned char)(
					  filter[0][0] * bmp->pixel[im1jm1].B
					+ filter[1][0] * bmp->pixel[i__jm1].B
					+ filter[2][0] * bmp->pixel[ip1jm1].B
					+ filter[0][1] * bmp->pixel[im1j__].B
					+ filter[1][1] * bmp->pixel[i__j__].B
					+ filter[2][1] * bmp->pixel[ip1j__].B
					+ filter[0][2] * bmp->pixel[im1jp1].B
					+ filter[1][2] * bmp->pixel[i__jp1].B
					+ filter[2][2] * bmp->pixel[ip1jp1].B);

			filtered[i__j__].G = (unsigned char)(
					  filter[0][0] * bmp->pixel[im1jm1].G
					+ filter[1][0] * bmp->pixel[i__jm1].G
					+ filter[2][0] * bmp->pixel[ip1jm1].G
					+ filter[0][1] * bmp->pixel[im1j__].G
					+ filter[1][1] * bmp->pixel[i__j__].G
					+ filter[2][1] * bmp->pixel[ip1j__].G
					+ filter[0][2] * bmp->pixel[im1jp1].G
					+ filter[1][2] * bmp->pixel[i__jp1].G
					+ filter[2][2] * bmp->pixel[ip1jp1].G);

			filtered[i__j__].R = (unsigned char)(
					  filter[0][0] * bmp->pixel[im1jm1].R
					+ filter[1][0] * bmp->pixel[i__jm1].R
					+ filter[2][0] * bmp->pixel[ip1jm1].R
					+ filter[0][1] * bmp->pixel[im1j__].R
					+ filter[1][1] * bmp->pixel[i__j__].R
					+ filter[2][1] * bmp->pixel[ip1j__].R
					+ filter[0][2] * bmp->pixel[im1jp1].R
					+ filter[1][2] * bmp->pixel[i__jp1].R
					+ filter[2][2] * bmp->pixel[ip1jp1].R);
		}
	}
	for (int j = 1; j < bmp->height -1; j++)
	{
		for (int i = 1; i < bmp->width -1; i++)
		{
			int i__j__ = (i  ) + bmp->width*(j  );
			bmp->pixel[i__j__].B = filtered[i__j__].B;
			bmp->pixel[i__j__].G = filtered[i__j__].G;
			bmp->pixel[i__j__].R = filtered[i__j__].R;
		}
	}
	free(filtered);
}

int main(void)
{
	char *filename = "Parrots.bmp";
	char *outputfilename = "output.bmp";
	BmpFileHeader fileHdr;
	BmpInfoHeader infoHdr;

	Img bmp;

	readBmp(filename, &bmp, &fileHdr, &infoHdr);

	// 画像処理を記述
	
	float gaussfilter[3][3] = {
		1.0/16.0, 2.0/16.0, 1.0/16.0,
		2.0/16.0, 4.0/16.0, 2.0/16.0,
		1.0/16.0, 2.0/16.0, 1.0/16.0
	};
		
	// negative(&bmp);
	// yreflect(&bmp);
	// gray(&bmp);
	boxfilter(&bmp, gaussfilter);
	
	writeBmp(outputfilename, &bmp, &fileHdr, &infoHdr);

	free(bmp.pixel);
	return 0;
}


