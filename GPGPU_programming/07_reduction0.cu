#include <stdio.h>
#include <stdlib.h>
#define N (512)
#define Nbytes (N*sizeof(int))
#define NT (1)
#define NB (1)


__global__ void reduction0(int *idata, int *odata)
{
    int i;

    odata[0] = 0;
    for (i=0; i<N; ++i)
    {
        odata[0] += idata[i];
    }
}

void init(int *idata)
{
    int i;
    for (i=0; i<N; ++i)
    {
        idata[i] = 1;
    }
}

int main()
{
    // GPU用変数 idata: 入力、odata: 出力(総和)
    int *idata, *odata;

    // CPU用変数 host_idata: 初期化用、sum: 総和
    int *host_idata, sum;

    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, sizeof(int));

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    reduction0<<<NB, NT>>>(idata, odata);

    // GPUから総和の結果を受け取って画面表示
    cudaMemcpy(&sum, odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum = %d\n", sum);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
