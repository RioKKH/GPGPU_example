#include <stdio.h>
#include <stdlib.h>
#define N (512)
#define Nbytes (N*sizeof(int))
#define NT (N)
#define NB (N / NT)
#define STEP (9) // reductionの段数


__global__ void reduction1(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    // stepはループのカウンターとしてのみ利用しているが、
    // ループカウンターはstrideで代用可能
    // stepはコメントアウトしてしまう
    // int step; // reductionの段数をカウントする変数
    int stride; // 隣の配列要素まで距離
    
    stride = 1;
    // ストライドを2倍し、ストライドがN/2以下ならループを継続
    // <<= : シフト演算の代入演算子 a <<= 1 --> a = a << 1と同じ
    // 最終stepではstrideが配列要素数のN/2となるので、strideがN/2
    // より大きくなるとループを中断
    for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
    {
        // 処理を行うスレッドを選択
        if (tx % (2 * stride) == 0)
        {
            idata[i] = idata[i] + idata[i + stride];
        }
        __syncthreads(); // スレッド間の同期を取る
        // stride = stride * 2; // ストライドを2倍して次のstepに備える
    }
    if (tx == 0) // スレッド0が総和を出力用変数odataに書き込んで終了
    {
        odata[0] = idata[0];
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

    reduction1<<<NB, NT>>>(idata, odata);

    // GPUから総和の結果を受け取って画面表示
    cudaMemcpy(&sum, odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum = %d\n", sum);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
