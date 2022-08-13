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

#include "filter.h"
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

    if (x < ksize / 2 || x >= cols - ksize / 2) return;
    if (y < ksize / 2 || x >= rows - ksize / 2) return;

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


void sobel_filter_cpu(const uint8_t* src, uint8_t* dst, int rows, int cols)
{
    const int kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };

#pragma omp parallel for
    for (int y = 1; y < rows - 1; ++y)
    {
        for (int x = 1; x < cols - 1; ++x)
        {
            int sum = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    sum += kernel[i][j] * src[(y + i - 1) * cols + x + j - 1];
                }
            }

            auto clamp = [](int val, int low, int high) { return std::min(std::max(val, low), high); };
            dst[y * cols + x] = uint8_t(clamp(sum, 0, 255));
        }
    }
}


__global__ void sobel_filter_gpu(const uint8_t* src, uint8_t* dst, int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= cols - 1) return;
    if (y < 1 || y >= rows - 1) return;

    const int kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };

    int sum = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            sum += kernel[i][j] * src[(y + i - 1) * cols + x + j - 1];
        }
    }
    
    auto clamp = [](int val, int low, int high) { return min(max(val, low), high); };

    dst[y * cols + x] = uint8_t(clamp(sum, 0, 255));
}


template <int ksize>
void check(uint8_t* a, uint8_t* b, int rows, int cols)
{
    const int size = ksize / 2;
    int count = 0;

    for (int y = size; y < rows - size; ++y)
    {
        for (int x = size; x < cols - size; ++x)
        {
            if (std::abs(a[y * cols + x] - b[y * cols + x]) > 1)
            {
                printf("%d %d %u %u\n", y, x, a[y * cols + x], b[y * cols + x]);
                count++;
            }
        }
    }

    const int pixels = (rows - 2 * size) * (cols - 2 * size);
    printf("err rate : %.2f %% : %d / %d\n", float(count) / pixels * 100, count, pixels);
}


#endif // FILTER_H
