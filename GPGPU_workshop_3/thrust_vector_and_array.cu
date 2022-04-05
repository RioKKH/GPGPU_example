#include <iostream>
#include <thrust/host_vector.h>

int main(void)
{
    thrust::host_vector<int> h_vec1(3);

    int *ptr_vec;

    h_vec1[0] = 10;
    h_vec1[1] = 20;
    h_vec1[2] = 30;

    // thrust::raw_pointer_cast()を利用して
    // vec1[0]のアドレスを取り出してポインタ変数に代入
    ptr_vec = thrust::raw_pointer_cast(&h_vec1[0]);

    for (int i = 0; i < 3; ++i)
    {
        std::cout << ptr_vec[i] << std::endl;
    }

    return 0;
}
