#include <stdio.h>

__global__ void hello()
{
    // 画面表示(Fermi世代以降で可能)
    // コンパイル時にオプションが必要
    // -arch=sm_20以降
    printf("Hello Thread\n");
}

int main(void)
{
    // カーネルの実行
    hello<<<1, 1>>>();

    // ホストとデバイスの同期を取る
    // CPUとGPUは原則同期しないので、同期しないと
    // カーネルを呼び出した直後にプログラムが終了
    cudaDeviceSynchronize();

    return 0;
}
