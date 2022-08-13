#ifndef FILTER_H
#define FILTER_H

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <opencv4/core.hpp>
#include <opencv4/highgui.hpp>
#include <opencv4/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__host__ __device__ float gaussian_2D(float x, float y, float sigma)
{
    float coefficient = 1.0f / (2.0f * float(M_PI) * sigma * sigma);
    float exponent    = -(x * x + y * y) / (2.0f * sigma * sigma);

    return coefficient * expf(exponent);
}


template <int ksize>
void gaussian_filter_cpu(const uint8_t* src, uint8_t* dst, int rows, int cols, float sigma)
{
    float kernel[ksize][ksize];
    float ksum = 0.0f;

    for (int i = 0; i < ksize; ++i)
    {
        const int ky = i - ksize / 2;
        for (int j = 0; j < ksize; ++j)
        {
            const int kx = j - ksize / 2;
            kernel[i][j] = gaussian_2D(float(kx), float(ky), sigma);
            ksum += kernel[i][j];
        }
    }

    for (int i = 0; i < ksize; ++i)
    {
        for (int j = 0; j < ksize; ++j)
        {
            kernel[i][j] /= ksum;
        }
    }

#pragma omp prallel for
    for (int y = ksize / 2; y < rows - ksize / 2; ++y)
    {
        for (int x = ksize / 2; x < cols - ksize / 2; ++x)
        {
        float sum = 0.0f;
        for (int i = 0; i < ksize; ++i)
        {
            const int ky = i - ksize / 2;
            for (int j = 0; j < ksize; ++j)
            {
                const int kx = j - ksize / 2;
                sum += kernel[i][j] * src[(y + ky) * cols + x + kx];
            }
        }
        dst[y * cols + x] = uint8_t(sum);
        }
    }
}


template<int ksize>
__global__ void gaussian_filter_gpu(const uint8_t* src, uint8_t* dst, int rows, int cols, float sigma)
{
    float kernel[ksize][ksize];
    float ksum = 0.0f;

    for (int i = 0; i < ksize; ++i)
    {
        const int ky = i - ksize / 2;
        for (int j = 0; j < ksize; ++j)
        {
            const int kx = j - ksize / 2;
            kernel[i][j] = gaussian_2D(float(kx), float(ky), sigma);
            ksum += kernel[i][j];
        }
    }

    for (int i = 0; i < ksize; ++i)
    {
        for (int j = 0; j < ksize; ++j)
        {
            kernel[i][j] /= ksum;
        }
    }

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;





#endif // FILTER_H
