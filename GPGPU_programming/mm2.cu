#define SIZE 4096
#define THREADX 16
#define THREADY 16
#define BLOCKX (SIZE / THREADX)
#define BLOCKY (SIZE / THREADY)

__global__ void matmulGPU(float *A, float *B, float *C)
{
	int i, j, k;
	// レジスタを使う
	float sum = 0.0f;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	C[i * SIZE + j] = 0.0f;
	for (k = 0; k < SIZE; k++)
	{
		// 足し算の度にグローバルメモリへのアクセスが生じる
		// C[i + SIZE * j] += A[i + SIZE * k] * B[k + SIZE * j];
		// レジスタを使うことでグローバルメモリへのアクセス回数を減らせる
		sum += A[i + SIZE * k] * B[k + SIZE * j];
	}
	C[i + SIZE * j] = sum;
}

