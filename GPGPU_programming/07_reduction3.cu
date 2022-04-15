#include <stdio.h>
#include <stdlib.h>
#define N (1024*8)
#define Nbytes (N*sizeof(int))
#define NT (256)
#define NB (N / NT)

__global__ void reduction3(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    int stride; // "隣"の配列要素まで距離

    // コンパイラの最適化を抑制
    // 複数のスレッドからアクセスされる変数に対する最適化
    // コンパイラが不要と判断して処理を削除してしまうことが有り、
    // 複数スレッドが変数の値をプライベートな領域にコピーして
    // 書き戻さない等が発生してしまう-->なのでvolatileを指定する
    __shared__ volatile int s_idata[NT]; // 共有メモリの宣言

    s_idata[tx] = idata[i]; // グローバルメモリから共有メモリへデータをコピー
    __syncthreads; // 共有メモリのデータは全スレッドから参照されるので同期を取る
    
    // ストライドを2倍し、ストライドがN/2以下ならループを継続
    // <<= : シフト演算の代入演算子 a <<= 1 --> a = a << 1と同じ
    // 最終stepではstrideが配列要素数のN/2となるので、strideがN/2
    // より大きくなるとループを中断
    for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
    {
        // 処理を行うスレッドを選択
        if (tx % (2 * stride) == 0)
        {
            s_idata[tx] = s_idata[tx] + s_idata[tx + stride];
        }
        __syncthreads(); // スレッド間の同期を取る
        // stride = stride * 2; // ストライドを2倍して次のstepに備える
    }
    if (tx == 0) // 各ブロックのスレッド0が総和を出力用変数odataに書き込んで終了
    {
        odata[blockIdx.x] = s_idata[0];
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
    int *host_idata, *sum;

    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, NB*sizeof(int)); // ブロックの数だけ部分和が出るので

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    reduction3<<<NB, NT>>>(idata, odata);

    // GPUから部分和を受け取って総和を計算
    sum = (int *)malloc(NB * sizeof(int));

    cudaMemcpy(sum, odata, NB*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=1; i<NB; i++) // sum[0]にそれ以外の要素を足しこんでいくのでi=1から始まる
    {
        sum[0] += sum[i];
    }

    printf("sum = %d\n", sum[0]);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
