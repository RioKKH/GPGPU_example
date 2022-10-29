// #include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bmpheader.h"

__constant__ float cfilter[3][3];

// BGR型とuchar4との相互変換
__global__ void cvtBGRToUchar4(BGR *bmp, uchar4 *pixel, int width, int height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;

	// make_uchar4はCUDAで定義される関数
	pixel[ij] = make_uchar4(bmp[ij].B, bmp[ij].G, bmp[ij].R, 0);
}

__global__ void cvtUchar4ToBGR(uchar4 *pixel, BGR *bmp, int width, int height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i * width * j;

	bmp[ij].B = pixel[ij].x;
	bmp[ij].G = pixel[ij].y;
	bmp[ij].R = pixel[ij].z;
}

// ネガティブ処理 (uchar4版)
__global__ void negativeUchar4(uchar4 *pixel, int width, int height, uchar4 *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;

	filtered[ij].x = 255 - pixel[ij].x;
	filtered[ij].y = 255 - pixel[ij].y;
	filtered[ij].z = 255 - pixel[ij].z;
	// filtered[ij].w = 255 - pixel[ij].w; // .wの処理は不要
}

// ネガティブ処理 (BGR版)
__global__ void negativeBGR(BGR *pixel, int width, int height, BGR *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;
	filtered[ij].B = 255 - pixel[ij].B;
	filtered[ij].G = 255 - pixel[ij].G;
	filtered[ij].R = 255 - pixel[ij].R;
}

// 画像の反転 (uchar4版)
__global__ void yreflectUchar4(uchar4 *pixel, int width, int height, uchar4 *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;
	int ijreflected = i + width * (height - 1 -j);

	filtered[ijreflected].x = pixel[ij].x;
	filtered[ijreflected].y = pixel[ij].y;
	filtered[ijreflected].z = pixel[ij].z;
	filtered[ijreflected].w = pixel[ij].w; // 単純な代入の場合、.w は無駄でも処理を記述しておく
}

// 画像の反転 (BGR版)
__global__ void yreflectBGR(BGR *pixel, int width, int height, BGR *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;
	int ijreflected = i + width * (height - j - 1);

	filtered[ijreflected].B = pixel[ij].B;
	filtered[ijreflected].G = pixel[ij].G;
	filtered[ijreflected].R = pixel[ij].R;
}

// グレイスケール (uchar4版)
__global__ void grayUchar4(uchar4 *pixel, int width, int height, uchar4 *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;

	unsigned char gray = (unsigned char)(  0.114478f * (float)pixel[ij].x
			                             + 0.586611f * (float)pixel[ij].y
										 + 0.298912f * (float)pixel[ij].z);

	filtered[ij].x = gray;
	filtered[ij].y = gray;
	filtered[ij].z = gray;
}

// グレイスケール (BGR版)
__global__ void grayBGR(BGR *pixel, int width, int height, BGR *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i + width * j;

	unsigned char gray = (unsigned char)(  0.114478f * (float)pixel[ij].B
			                             + 0.586611f * (float)pixel[ij].G
										 + 0.298912f * (float)pixel[ij].R);

	filtered[ij].B = gray;
	filtered[ij].G = gray;
	filtered[ij].R = gray;
}


// 空間フィルター uchar4版
__global__ void boxfilterUchar4(uchar4 *pixel, int width, int height, uchar4 *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (0<i && i<width && 0<j && j<height)
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

		// cfilterはコンスタントメモリ
		filtered[i__j__].x = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].x
			  + cfilter[1][0] * pixel[i__jm1].x
			  + cfilter[2][0] * pixel[ip1jm1].x
			  + cfilter[0][1] * pixel[im1j__].x
			  + cfilter[1][1] * pixel[i__j__].x
			  + cfilter[2][1] * pixel[ip1j__].x
			  + cfilter[0][2] * pixel[im1jp1].x
			  + cfilter[1][2] * pixel[i__jp1].x
			  + cfilter[2][2] * pixel[ip1jp1].x);

		filtered[i__j__].y = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].y
			  + cfilter[1][0] * pixel[i__jm1].y
			  + cfilter[2][0] * pixel[ip1jm1].y
			  + cfilter[0][1] * pixel[im1j__].y
			  + cfilter[1][1] * pixel[i__j__].y
			  + cfilter[2][1] * pixel[ip1j__].y
			  + cfilter[0][2] * pixel[im1jp1].y
			  + cfilter[1][2] * pixel[i__jp1].y
			  + cfilter[2][2] * pixel[ip1jp1].y);

		filtered[i__j__].z = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].z
			  + cfilter[1][0] * pixel[i__jm1].z
			  + cfilter[2][0] * pixel[ip1jm1].z
			  + cfilter[0][1] * pixel[im1j__].z
			  + cfilter[1][1] * pixel[i__j__].z
			  + cfilter[2][1] * pixel[ip1j__].z
			  + cfilter[0][2] * pixel[im1jp1].z
			  + cfilter[1][2] * pixel[i__jp1].z);
	}
}

// 空間フィルター BGR版
__global__ void boxfilterBGR(BGR *pixel, int width, int height, BGR *filtered)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (0<i && i<width && 0<j && j<height)
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

		// cfilterはコンスタントメモリ
		filtered[i__j__].B = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].B
			  + cfilter[1][0] * pixel[i__jm1].B
			  + cfilter[2][0] * pixel[ip1jm1].B
			  + cfilter[0][1] * pixel[im1j__].B
			  + cfilter[1][1] * pixel[i__j__].B
			  + cfilter[2][1] * pixel[ip1j__].B
			  + cfilter[0][2] * pixel[im1jp1].B
			  + cfilter[1][2] * pixel[i__jp1].B
			  + cfilter[2][2] * pixel[ip1jp1].B);

		filtered[i__j__].G = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].G
			  + cfilter[1][0] * pixel[i__jm1].G
			  + cfilter[2][0] * pixel[ip1jm1].G
			  + cfilter[0][1] * pixel[im1j__].G
			  + cfilter[1][1] * pixel[i__j__].G
			  + cfilter[2][1] * pixel[ip1j__].G
			  + cfilter[0][2] * pixel[im1jp1].G
			  + cfilter[1][2] * pixel[i__jp1].G
			  + cfilter[2][2] * pixel[ip1jp1].G);

		filtered[i__j__].R = (unsigned char)(
				cfilter[0][0] * pixel[im1jm1].R
			  + cfilter[1][0] * pixel[i__jm1].R
			  + cfilter[2][0] * pixel[ip1jm1].R
			  + cfilter[0][1] * pixel[im1j__].R
			  + cfilter[1][1] * pixel[i__j__].R
			  + cfilter[2][1] * pixel[ip1j__].R
			  + cfilter[0][2] * pixel[im1jp1].R
			  + cfilter[1][2] * pixel[i__jp1].R);
	}
}

//- ガウシアンフィルター BGR版
__global__ void gaussianKernelGPUSimple(BGR *pixel, int width, int height, int step, BGR *filtered)
{
    const float filter3x3[3][3] = {
        { 0.0625, 0.1250, 0.0625 },
        { 0.1250, 0.2500, 0.1250 },
        { 0.0625, 0.1250, 0.0625 }
    };

    const float filter5x5[5][5] = {
        { 0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f },
        { 0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f },
        { 0.023438f, 0.093750f, 0.140625f, 0.093750f, 0.023438f },
        { 0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f },
        { 0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f },
    };

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        BGR sum; 
        for (int dj = 0; dj < 5; ++dj) // ij loop for kernel
        {
            for (int di = 0; di < 5; ++di)
            {
                sum.B += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].B;
                sum.G += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].G;
                sum.R += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].R;
            }
        }
        filtered[i + j * step].B = (unsigned char)(sum.B + 0.5f);
        filtered[i + j * step].G = (unsigned char)(sum.G + 0.5f);
        filtered[i + j * step].R = (unsigned char)(sum.R + 0.5f);
    }
}


//- Gaussian filter BGR Constantメモリー版
__constant__ float filter5x5[5][5] = {
        { 0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f },
        { 0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f },
        { 0.023438f, 0.093750f, 0.140625f, 0.093750f, 0.023438f },
        { 0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f },
        { 0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f },
};

__global__ void gaussianKernelGPUConstant(BGR *pixel, int width, int height, int step, int ks, BGR *filtered)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        BGR sum = {0, 0, 0};
        for (int dj = 0; dj < ks; ++dj)
        {
            for (int di = 0; di < ks; ++di)
            {
                sum.B += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].B;
                sum.G += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].G;
                sum.R += filter5x5[dj][di] * pixel[(i + di) + (j + dj) * step].R;
            }
        }
        filtered[i + j * step].B = (unsigned char)(sum.B + 0.5f);
        filtered[i + j * step].G = (unsigned char)(sum.G + 0.5f);
        filtered[i + j * step].R = (unsigned char)(sum.R + 0.5f);
    }
}


