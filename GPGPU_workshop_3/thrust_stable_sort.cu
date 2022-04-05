#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

int main()
{
    const int N = 6;

    thrust::host_vector<int> h_vec(N);
    h_vec[0] = 1;
    h_vec[1] = 4;
    h_vec[2] = 2;
    h_vec[3] = 8;
    h_vec[4] = 5;
    h_vec[5] = 7;

    thrust::stable_sort(h_vec.begin(), h_vec.end(), thrust::greater<int>());

    for (const auto& a : h_vec)
    {
        std::cout << a << std::endl;
    }

    return 0;
}

