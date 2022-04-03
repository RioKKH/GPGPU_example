#include <stdio.h>
#include <math.h>
#include <cublas_v2.h> // ヘッダーファイルのインクルード

#define N 256

int main()
{
    int i, j;
    // Aは行列で2次元配列になるが、1次元として確保する
    // A[i][j] --> C言語はj方向優先のメモリ配置
    float A[N*N], B[N], C[N];
    float *d_A, *d_B, *d_C;
    float alpha, beta;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < N; i++)
        {
            A[i + N*j] = (float)i / (float)N;
        }
    }

    for (i = 0; i < N; i++)
    {
        C[i] = 0.0f;
    }
    for(i = 0; i < N; i++)
    {
        B[i] = 1.0f;
    }

    // GPU上のメモリ確保
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*sizeof(float)); 
    cudaMalloc((void**)&d_C, N*sizeof(float));

    alpha = 1.0f;
    beta  = 0.0f;

    // CUBLASのハンドル作成
    cublasHandle_t handle;

    // ハンドル(行列やベクトルの情報を管理)の生成
    // 返り値はcublassStatus_t型
    cublasCreate(&handle);

    // データ転送 --> 関数内部でcudaMemcopyが呼ばれる
    cublasSetMatrix(N, N, sizeof(float), A, N, d_A, N);

    // データ転送
    cublasSetVector(N, sizeof(float), B, 1, d_B, 1);

    // データ転送
    cublasSetVector(N, sizeof(float), C, 1, d_C, 1);

    // 演算を行う関数の呼び出し
    // CUBLAS_OP_N 行列Aに対する操作
    // cublasOperation_T型
    // CUBLAS_OP_N OP(A) = A --> 処理しない
    // CUBLAS_OP_T OP(A) = A^T --> 転置
    // CUBLAS_OP_C OP(A) = A^H --> 共役転置
    cublasSgemv(handle, CUBLAS_OP_N, N, N,
                &alpha, d_A, N, d_B, 1, &beta, d_C, 1);

    // データの読み戻し
    cublasGetVector(N, sizeof(float), d_C, 1, C, 1);

    // テキストでは jになっているがiが正しい
    for (i = 0; i < N; i++)
    {
        printf("C[%3d] = %f \n", i, C[i]);
    }

    // GPU上のメモリの解放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // ハンドルの削除
    cublasDestroy(handle);

    return 0;
}
