#ifndef FILTER_H
#define FILTER_H

// #include <cstdio>
// #include <cstdint>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__host__ __device__ float gaussian_2D(float x, float y, float sigma);

//- Gaussian filter
template <int ksize>
void gaussian_filter_cpu(const uint8_t* src, uint8_t* dst, int rows, int cols, float sigma);

template<int ksize>
__global__ void gaussian_filter_gpu(const uint8_t* src, uint8_t* dst, int rows, int cols, float sigma);

//- Sobel filter
void sobel_filter_cpu(const uint8_t* src, uint8_t* dst, int rows, int cols);

__global__ void sobel_filter_gpu(const uint8_t* src, uint8_t dst, int rows, int cols);

//- 
template <int ksize>
void check(uint8_t* a, uint8_t* b, int rows, int cols);


#endif // FILTER_H
