#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N (256)
#define Nbytes (N*sizeof(int))
#define NT (N / 2)      // スレッド数を半分に
#define NB (N / NT / 2) // スレッド数を半分にするとブロック数が増えてしまうので、
                        // ブロック数を増やさないように調整(2で割る)

// 共有メモリの要素数を決定
// NT >  32の時、1 * NT + 0 * (NT + NT / 2)
// NT <= 32の時、0 * NT + 1 * (NT + NT / 2)
#define smemSize ((NT > 32) * NT + (NT <=32) * (NT + NT/2))


//共有メモリによるキャッシュ利用
// Warp内の各スレッドが異なる処理を実行
// --> Stepが進むに連れて処理を行うスレッドが減少
// --> Warp内でdivergentが発生
// --> Warp divergentを排除する総和計算を実装する!!
// 処理を行うスレッド数が32以下になった時点で処理を切り替え

__global__ void reduction7(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    int stride; // "隣"の配列要素まで距離

    // コンパイラの最適化を抑制
    // 複数のスレッドからアクセスされる変数に対する最適化
    // コンパイラが不要と判断して処理を削除してしまうことが有り、
    // 複数スレッドが変数の値をプライベートな領域にコピーして
    // 書き戻さない等が発生してしまう-->なのでvolatileを指定する

    // 入力データの数に応じて必要な共有メモリサイズが変化
    __shared__ volatile int s_idata[smemSize]; // 共有メモリの宣言

    // グローバルメモリから共有メモリへデータをコピー
    // 1スレッドが2点読み込んで加算
    // blockDim.xは1スレッドが読み込む1点目と2点目の間隔
    s_idata[i] = idata[i] + idata[i + blockDim.x]; 
    __syncthreads(); // 共有メモリのデータは全スレッドから参照されるので同期を取る

    // ストライドが32より大きい (=処理を実行するスレッドが32以上)場合にループ内の処理を実行
    for (stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tx < stride) // ストライドよりスレッド番号が小さいスレッドのみ計算に参加
        {
            s_idata[tx] += s_idata[tx + stride];
        }
        __syncthreads(); // スレッド間の同期を取る
    }

    // 処理を実行するスレッドが32(1 Warp内の0 ~ 31)になると処理を切り替え
    // 0 ~ 31スレッド全てが強調してif 文内の処理を実行するため、__syncthreads()が不要！
    if (tx < 32)
    {
        // if (blockDim.x >= ??)は、入力データ数が64以下の条件で正しく結果を求めるために必要
        if (blockDim.x >= 64) s_idata[tx] += s_idata[tx + 32];
        if (blockDim.x >= 32) s_idata[tx] += s_idata[tx + 16];
        if (blockDim.x >= 16) s_idata[tx] += s_idata[tx +  8];
        if (blockDim.x >=  8) s_idata[tx] += s_idata[tx +  4];
        if (blockDim.x >=  4) s_idata[tx] += s_idata[tx +  2];
        if (blockDim.x >=  2) s_idata[tx] += s_idata[tx +  1];
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

    // 実行時間計測用
    float elapsed_time_ms = 0.0f; // 経過時間保存用
    // イベントを取り扱う変数
    cudaEvent_t start, end;
    // イベントのクリエイト
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, sizeof(int));

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    cudaEventRecord(start, 0);
    reduction7<<<NB, NT>>>(idata, odata);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time_ms, start, end);
    std::cout << "Elapsed Time: " << elapsed_time_ms << std::endl;

    // GPUから総和の結果を受け取って画面表示
    cudaMemcpy(&sum, odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum = %d\n", sum);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
