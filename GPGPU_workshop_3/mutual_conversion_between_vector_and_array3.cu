#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

int main(void)
{
    // GPU上に確保するメモリへのポインタ
    int *vec1;
    // thrustのベクトル(ホストに確保)
    thrust::host_vector<int> vec2(3);

    // GPU上にメモリを確保
    cudaMalloc((void **)&vec1, 3*sizeof(int));

    // device_ptr型変数dev_vec1を宣言し、vec1を包み隠す
    thrust::device_ptr<int> dev_vec1(vec1);

    // dev_vec1をベクトルとして利用
    dev_vec1[0] = 10;
    dev_vec1[1] = 20;
    dev_vec1[2] = 30;

    // 関数の引数として利用
    thrust::copy(dev_vec1, dev_vec1+3, vec2.begin());
    for(int i=0; i<3; i++)
    {
        printf("%d\n", vec2[i]);
    }

    // vec1は解放が必要
    cudaFree(vec1);
    return 0;
}
