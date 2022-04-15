#include <stdio.h>
#include <stdlib.h>
#define N (512)
#define Nbytes (N*sizeof(int))
#define NT (N)
#define NB (N / NT)


__global__ void reduction4(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    int stride; // "隣"の配列要素まで距離
    int j;      // 各stepで総和計算を行うスレッドがアクセスする配列要素番号

    // コンパイラの最適化を抑制
    // 複数のスレッドからアクセスされる変数に対する最適化
    // コンパイラが不要と判断して処理を削除してしまうことが有り、
    // 複数スレッドが変数の値をプライベートな領域にコピーして
    // 書き戻さない等が発生してしまう-->なのでvolatileを指定する
    __shared__ volatile int s_idata[NT]; // 共有メモリの宣言

    s_idata[i] = idata[i]; // グローバルメモリから共有メモリへデータをコピー
    __syncthreads; // 共有メモリのデータは全スレッドから参照されるので同期を取る
    
    // ストライドを2倍し、ストライドがN/2以下ならループを継続
    // <<= : シフト演算の代入演算子 a <<= 1 --> a = a << 1と同じ
    // 最終stepではstrideが配列要素数のN/2となるので、strideがN/2
    // より大きくなるとループを中断
    for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
    {
        // 総和計算を行うスレッド番号とアクセスする配列要素の決定
        // 連続なスレッドが2 * stride間隔で配列にアクセスする
        // step 1: 
        j = 2*stride * tx;
        if (j < blockDim.x)
        {
            s_idata[j] += s_idata[j + stride];
        }
        __syncthreads(); // スレッド間の同期を取る
        // stride = stride * 2; // ストライドを2倍して次のstepに備える
    }
    if (tx == 0) // スレッド0が総和を出力用変数odataに書き込んで終了
    {
        odata[blockIdx.x] = s_idata[tx];
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

    reduction4<<<NB, NT>>>(idata, odata);

    // GPUから総和の結果を受け取って画面表示
    cudaMemcpy(&sum, odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum = %d\n", sum);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
