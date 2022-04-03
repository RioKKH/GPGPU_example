#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
// #include <math.h>

void setTridiagonalMatrix(double*, int);
void setRightHandSideVector(double*, double*, double*, int);
void computeResidual(double*, double*, double*, double*, int);
double innerProduct(double*, double*, int);
void copy(double*, double*, int);
void computeVectorAdd(double*, const double, double*, const double, int);
void computeMxV(double*, double*, double*, int);


int main(void)
{
    int N = 1 << 10;                // 未知数の数2^10
    const double err_tol = 1e-9;    // 許容誤差
    const int max_ite = 1 << 20;    // 反復回数の上限 2^20

    double *x;      // 近似解ベクトル
    double *b;      // 右辺ベクトル
    double *A;      // 係数行列
    double *sol;    // 厳密解
    // double *r, rr;  // 残差ベクトル、残差の内積
    // double *p, *Ax; // 補助ベクトル、行列ベクトル積
    // double c1, c2, dot; // 計算に使う係数
    // int i, k;

    // GPU用変数
    double *d_x;
    double *d_b;
    double *d_A;
    double *d_r, rr;
    double *d_p, *d_Ax;
    double c1, c2, minusC1, dot;

    // メモリの確保

    A = (double *)malloc(sizeof(double)*N*N);
    x = (double *)malloc(sizeof(double)*N);
    b = (double *)malloc(sizeof(double)*N);
    sol = (double *)malloc(sizeof(double)*N);
    r = (double *)malloc(sizeof(double)*N);
    p = (double *)malloc(sizeof(double)*N);
    Ax = (double *)malloc(sizeof(double)*N);

    for (i = 0; i < N; i++)
    {
        sol[i] = (double)i; // 厳密解を設定
        x[i] = 0.0; // 近似解を0で初期化
    }

    // 係数行列Aの生成
    setTridiagonalMatrix(A, N);
    // 右辺ベクトルbの生成
    setRightHandSideVector(b, A, sol, N);
    
    // ここで共役勾配法を実行 -----------------------------------
    // 残差ベクトルの計算 r^(0) = b - Ax^(0)
    computeResidual(r, b, A, x, N);
    // 残差ベクトルの内積を計算
    rr = innerProduct(r, r, N);

    k = 1;
    while (rr > err_tol * err_tol && k <= max_ite)
    {
        if (k == 1)
        {
            // p^(k) = r^(k) + c_2^(k-1)p^(k-1)
            // c_2とpが0の為、p^(k) = r^(k)
            copy(p, r, N);
        }
        else
        {
            c2 = rr / (c1 * dot);
            // p^(k) = r^(k) + c_2^(k-1)p^(k-1)
            computeVectorAdd(p, c2, r, 1.0, N);
        }

        // (p^(k), Ap^(k))を計算
        // 行列ベクトル積Apを実行し、結果とpの内積
        computeMxV(Ax, A, p, N);
        dot = innerProduct(p, Ax, N);
        c1 = rr / dot;

        // x^(k+1) = x^(k) + c_1^(k)p^(k)
        // r^(k+1) = r^(k) + c_1^(k)Ap^(k)

        computeVectorAdd(x, 1.0,  p,  c1, N);
        computeVectorAdd(r, 1.0, Ax, -c1, N);

        // 残差ベクトルの内積を計算
        rr = innerProduct(r, r, N);

        k++;
    }
    // ----------------------------------------------------------

    // 確保したメモリを解放
    free(A);
    free(x);
    free(b);
    free(sol);
    free(r);
    free(p);
    free(Ax);
}

// 係数行列の生成
void setTridiagonalMatrix(double *A, int N)
{
    int i, j;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < N; i++)
        {
            A[i+N*j] = 0.0;
        }
    }

    i = 0;
    A[i + N*i   ] = -4.0;
    A[i + N*i+1 ] = 1.0;

    for (i = 1; i < N-1; i++)
    {
        A[i + N*i-1 ] = 1.0;
        A[i + N*i   ] = -4.0;
        A[i + N*i+1 ] = 1.0;
    }

    i = N - 1;
    A[i + N*i-1 ] = 1.0;
    A[i + N*i   ] = -4.0;
}

void setRightHandSideVector(double *b, double *A, double *x, int N)
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        b[i] = 0.0;
        // 係数行列と厳密解ベクトルを用いて行列-ベクトル積を
        // 計算し、結果を右辺ベクトルに代入
        for (j = 0; j < N; j++)
        {
            b[i] += A[i + N*j] * x[j];
        }
    }
}

// 残差ベクトルr^(0) = b - Ax^(0)の計算
void computeResidual(double *r, double *b, double *A, double *x, int N)
{
    int i, j;
    double Ax;
    for (i = 0; i < N; i++)
    {
        Ax = 0.0;
        for (j = 0; j < N; j++)
        {
            Ax += A[i + N*j] * x[j];
        }
        r[i] = b[i] - Ax;
    }
}

// 内積の計算
double innerProduct(double *vec1, double *vec2, int N)
{
    int i;
    double dot;

    dot = 0.0;
    for (i = 0; i < N; i++)
    {
        dot += vec1[i] * vec2[i];
    }

    return dot;
}

// 値のコピー(単純代入) p^{k} = r^{k}
void copy(double *lhs, double *rhs, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        lhs[i] = rhs[i];
    }
}

// ベクトル和 y^{(k)} = ax^{(k)} + by^{(k)}の計算
void computeVectorAdd(double *y, const double b,
                      double *x, const double a,
                      int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        y[i] = a * x[i] + b * y[i];
    }
}

// 行列−ベクトル積Axの計算
void computeMxV(double *Ax, double *A, double *x, int N)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        Ax[i] = 0.0;
        for (j = 0; j < N; j++)
        {
            Ax[i] += A[i + N*j] * x[j];
        }
    }
}


