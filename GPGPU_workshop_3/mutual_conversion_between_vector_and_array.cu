#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(void)
{
    thrust::host_vector<int> vec1(3);

    vec1[0] = 10;
    vec1[1] = 20;
    vec1[2] = 30;

    // vec1[0]のアドレスを取り出してポインタ変数に代入
    int *ptr_vec;
    ptr_vec = thrust::raw_pointer_cast(&vec1[0]);

    for (int i=0; i<3; i++)
    {
        // ptr_vecとvec1は同じデータにアクセス
        printf("%d\n", ptr_vec[i]);
    }
}

