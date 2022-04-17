#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4096
#define Bytes (SIZE * SIZE * sizeof(float))
#define MILLISEC_PER_SEC	1000
#define FLOPS_TO_GFLOPS		1e-9

void matmul(float *, float *, float *);

int main(void)
{
	float *hA, *hB, *hC;
	int i, j, k;

	clock_t start_c, stop_c;
	float time_s, time_ms;
	float gflops;

	hA = (float *)malloc(Bytes);
	hB = (float *)malloc(Bytes);
	hC = (float *)malloc(Bytes);

	for (i = 0; i < SIZE; i++)
	{
		for (k = 0; k < SIZE; k++)
		{
			hA[i * SIZE + k] = (float)(i + 1) * 0.1f;
		}
	}
	for (k = 0; k < SIZE; k++)
	{
		for (j = 0; j < SIZE; j++)
		{
			hB[k * SIZE + j] = (float)(j + 1) * 0.1f;
		}
	}
	for (i = 0; i < SIZE; i++)
	{
		for (j = 0; j < SIZE; j++)
		{
			hC[i * SIZE + j] = 0.0f;
		}
	}

	start_c = clock();
	matmul(hA, hB, hC);
	stop_c = clock();

	time_s = (stop_c - start_c) / (float)CLOCKS_PER_SEC;
	time_ms = time_s * MILLISEC_PER_SEC;
	gflops = 2.0 * SIZE * SIZE * SIZE / time_s * FLOPS_TO_GFLOPS;

	printf("%f ms\n", time_ms);
	printf("%f GFLOPS\n", gflops);

	return 0;
}

void matmul(float *A, float *B, float *C)
{
	int i, j, k;
	for (i = 0; i < SIZE; i++)
	{
		for (j = 0; j < SIZE; j++)
		{
			for (k = 0; k < SIZE; k++)
			{
				C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
			}
		}
	}
}
