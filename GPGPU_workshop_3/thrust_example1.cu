#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main()
{
    // 3要素からなるvectorをホストメモリに確保
    // templateを利用して型を決定
    thrust::host_vector<int> h_vec(3);

    // 配列のように各要素にアクセス
    h_vec[0] = 10;
    h_vec[1] = 20;
    h_vec[2] = 30;

    // vectorをGPUへコピー
    thrust::device_vector<int> d_vec = h_vec;

    // device上のベクトルの数値を出力する
    std::cout << d_vec[0] << std::endl;
    std::cout << d_vec[1] << std::endl;
    std::cout << d_vec[2] << std::endl;

    // vectorは自動的に解放される
    // free(), cudaFree()を呼ぶ必要がない
    return 0;
}
