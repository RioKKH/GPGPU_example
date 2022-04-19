#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bmpheader.h"

#include "kernel.cu"

extern __constant__ float cfilter[3][3]; 

int main(void)
{
	float blurfilter[3][3] = {
		{1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
		{1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
		{1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
	};

	float gaussianfilter[3][3] = {
		{1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
		{2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
		{1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
	};

	// フィルタ用の係数テーブルをコンスタントメモリへコピー
	cudaMemcpyToSymbol(cfilter, gaussianfilter, 3*3*sizeof(float));

	char *filename = "Parrots.bmp";
	char *outputfilename = "output.bmp";

	BmpFileHeader fileHdr;
	BmpInfoHeader infoHdr;
	Img bmp;

	// ビットマップファイルの読み込み
	readBmp(filename, &bmp, &fileHdr, &infoHdr);
	// 画像データをBGR型でGPUへコピー
	BGR *dev_bmp;
	cudaMalloc((void **)&dev_bmp, sizeof(BGR) * bmp.width * bmp.height);
	cudaMemcpy(dev_bmp, bmp.pixel, sizeof(BGR) * bmp.width * bmp.height, cudaMemcpyHostToDevice);



