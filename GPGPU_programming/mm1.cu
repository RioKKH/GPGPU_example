#include <stdio.h>
#include <stdlib.h>

#define THREADX 1
#define THREADY 1
#define BLOCKX  1
#define BLOCKY  1

// 1スレッドが全ての要素を計算
__global__ void matmulGPU(float *A, float *B, float *C)
{
	int i, j, k;
		for (j = 0; j < SIZE; j++)
	{
		for (i = 0; i < SIZE; i++)
		{
			for (k = 0; i < SIZE; k++)
			{
				// C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
				C[i + SIZE * j] += A[i + SIZE * k] * B[k + SIZE * j];
			}
		}
	}
}



