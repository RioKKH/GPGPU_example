#include <iostream>
#include <stdlib.h>
#include <math.h>

#define Lx (2.0 * M_PI)
#define Nx (1024 * 1024)
#define dx (Lx / (Nx - 1))
#define Nbytes (Nx * sizeof(double))
#define NT (256)
#define NB (Nx / NT)

void init(double *u)
{
	int i;
	for (i=0; i<Nx; ++i)
	{
		u[i] = sin(i * dx);
	}
}

__global__ void differentiate(double *u, double *dudx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0)
	{
		dudx[i] = (-3.0 * u[i   ]
			   +4.0 * u[i+1 ]
			   -      u[i+2 ]) / (2.0 * dx);
	}

	if (0<i && i<Nx-1)
	{
		dudx[i] = ( u[i+1]
			   -u[i-1]) / (2.0*dx);
	}

	if (i == Nx - 1)
	{
		dudx[i] = (       u[i - 2]
			   -4.0 * u[i - 1]
			   +3.0 * u[i    ]) / (2.0 * dx);
	}
}

int main(void)
{
	double *host_u, *host_dudx;
	double *u, *dudx;

	host_u		= (double *)malloc(Nbytes);
	host_dudx	= (double *)malloc(Nbytes);
	cudaMalloc((void **)&u, Nbytes);
	cudaMalloc((void **)&dudx, Nbytes);

	init(host_u);
	cudaMemcpy(u, host_u, Nbytes, cudaMemcpyHostToDevice);

	differentiate<<<NB, NT>>>(u, dudx);
	cudaMemcpy(host_dudx, dudx, Nbytes, cudaMemcpyDeviceToHost);

	free(host_u);
	free(host_dudx);
	cudaFree(u);
	cudaFree(dudx);

	return 0;
}


