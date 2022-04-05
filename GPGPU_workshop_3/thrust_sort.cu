#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main(void)
{
    thrust::host_vector<int> h_vec1(3);
    h_vec1[0] = 30;
    h_vec1[1] = 10;
    h_vec1[2] = 20;

    thrust::device_vector<float> d_vec1(3);
    d_vec1[0] = 3.0f;
    d_vec1[1] = 1.0f;
    d_vec1[2] = 2.0f;

    thrust::sort(h_vec1.begin(), h_vec1.end());
    thrust::sort(d_vec1.begin(), d_vec1.end());

    for (const auto& a : h_vec1)
    {
        std::cout << a << std::endl;
    }

    for (const auto& a : d_vec1)
    {
        std::cout << a << std::endl;
    }

    return 0;
}
