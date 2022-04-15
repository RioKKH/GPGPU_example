#define N (8 * 1024) // 64kBに収める
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

__global__ void init(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	a[i] = 1.0;
	b[i] = 2.0;
	c[i] = 0.0;
}

__global__ void add(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(void)
{
	float *a, *b, *c;

	cudaMalloc((void **)&a, Nbytes);
	cudaMalloc((void **)&b, Nbytes);
	cudaMalloc((void **)&c, Nbytes);

	init<<<NB, NT>>>(a, b, c);
	add<<<NB, NT>>>(a, b, c);
	
	return 0;
}
