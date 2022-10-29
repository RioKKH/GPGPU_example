#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bmpheader.h"

#include "kernel.cu"

#define THREADX 256
#define THREADY 1

// extern __constant__ float cfilter[3][3]; 

#ifdef _UCHAR4
constexpr bool is_UCHAR4 = true;
#else
constexpr bool is_UCHAR4 = false;
#endif // _UCHAR4

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

int readBmp(const char *filename, Img *bmp, BmpFileHeader *fileHdr, BmpInfoHeader *infoHdr)
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

int writeBmp(const char *outputfilename, Img *bmp, BmpFileHeader *fileHdr, BmpInfoHeader *infoHdr)
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


int main(void)
{
	// フィルタ用の係数テーブルをコンスタントメモリへコピー
	// cudaMemcpyToSymbol(cfilter, gaussianfilter, 3*3*sizeof(float));

	const char *filename = "Parrots.bmp";
	const char *outputfilename = "output.bmp";

	BmpFileHeader fileHdr;
	BmpInfoHeader infoHdr;
	Img bmp;

	// ビットマップファイルの読み込み
	readBmp(filename, &bmp, &fileHdr, &infoHdr);
	// 画像データをBGR型でGPUへコピー
	BGR *dev_bmp;
	cudaMalloc((void **)&dev_bmp, sizeof(BGR) * bmp.width * bmp.height);
	cudaMemcpy(dev_bmp, bmp.pixel, sizeof(BGR) * bmp.width * bmp.height, cudaMemcpyHostToDevice);

	// ここに画像処理を記述
	if (is_UCHAR4)
	{
		printf("UCHAR4: \n");
		dim3 thread = dim3(THREADX, THREADY, 1);
		dim3 block  = dim3(bmp.width/thread.x, bmp.height/thread.y, 1);

		uchar4 * pixel, *uchar4filtered;
		cudaMalloc((void **)&pixel, sizeof(uchar4)*bmp.width * bmp.height);
		cudaMalloc((void **)&uchar4filtered, sizeof(uchar4)*bmp.width * bmp.height);

		// BGR型をuchar4型に変換
		cvtBGRToUchar4<<<block, thread>>>(dev_bmp, pixel, bmp.width, bmp.height);

		// negativeUchar4<<<block, thread>>>(pixel, bmp.width, bmp.height, uchar4filtered);
		yreflectUchar4<<<block, thread>>>(pixel, bmp.width, bmp.height, uchar4filtered);
		// grayUchar4<<<block, thread>>>(pixel, bmp.width, bmp.height, uchar4filtered);
		// boxfilterUchar4<<<block, thread>>>(pixel, bmp.width, bmp.height, uchar4filtered);

		// フィルタ処理後の画像と原画像のポインタを交換し、処理後の画像をBGR型に変換
		uchar4 *swap = uchar4filtered;
		uchar4filtered = pixel;
		pixel = swap;
		cvtUchar4ToBGR<<<block, thread>>>(pixel, dev_bmp, bmp.width, bmp.height);
		// 画像をホストメモリへコピー
		cudaMemcpy(bmp.pixel, dev_bmp, sizeof(BGR) * bmp.width * bmp.height, cudaMemcpyDeviceToHost);

		cudaFree(pixel);
		cudaFree(uchar4filtered);
	}
	else
	{
		printf("BGR: \n");
		dim3 thread = dim3(THREADX, THREADY, 1);
		dim3 block  = dim3(bmp.width/thread.x, bmp.height/thread.y, 1);

		BGR *BGRfiltered;
		// cudaMalloc((void **)&BGRfiltered, sizeof(BGR) * bmp.width * bmp.height);
        cudaMallocHost((void**)&BGRfiltered, sizeof(BGR) * bmp.width * bmp.height);
        // cudaHostAlloc((void**)&BGRfiltered, sizeof(BGR) * bmp.width * bmp.height, 0);
        // memcpy(BGRfiltered, 

		// negativeBGR<<<block, thread>>>(dev_bmp, bmp.width, bmp.height, BGRfiltered);
		// yreflectBGR<<<block, thread>>>(dev_bmp, bmp.width, bmp.height, BGRfiltered);
		// grayBGR<<<block, thread>>>(dev_bmp, bmp.width, bmp.height, BGRfiltered);
		// boxfilterBGR<<<block, thread>>>(dev_bmp, bmp.width, bmp.height, BGRfiltered);
        // gaussianKernelGPUSimple<<<block, thread>>>(dev_bmp, bmp.width -4, bmp.height -4, bmp.width, BGRfiltered);
        gaussianKernelGPUConstant<<<block, thread>>>(dev_bmp, bmp.width -4, bmp.height -4, bmp.width, 5, BGRfiltered);

		// フィルタ処理後の画像と原画像のポインタを交換し、処理後の画像をBGR型に変換
		BGR *swap = BGRfiltered;
		BGRfiltered = dev_bmp;
		dev_bmp = swap;
		// 画像をホストメモリへコピー
		cudaMemcpy(bmp.pixel, dev_bmp, sizeof(BGR) * bmp.width * bmp.height, cudaMemcpyDeviceToHost);

		cudaFree(dev_bmp);
		cudaFree(BGRfiltered);
	}
    /*
    else
    {
        printf("Pinned: \n");
        dim3 thread = dim3(THREADX, THREADY, 1);
        dim3 block  = dim3(bmp.width/thread.x, bmp.height/thread.y, 1);

        BGR *BGRfiltered;
        cudaHostAlloc((void**)&BGRfiltered, sizeof(BGR) * bmp.width * bmp.height);
    }
    */

	// ビットマップファイルの書き出し
	writeBmp(outputfilename, &bmp, &fileHdr, &infoHdr);
	free(bmp.pixel);
	return 0;
}




