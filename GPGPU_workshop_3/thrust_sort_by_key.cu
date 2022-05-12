#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main(void)
{
    const int N = 6;

    std::cout << "Host:" << std::endl;
    thrust::host_vector<int> h_keys(N);
    h_keys[0] = 1; h_keys[1] = 4; h_keys[2] = 2; 
    h_keys[3] = 8; h_keys[4] = 5; h_keys[5] = 7;

    thrust::host_vector<char> h_values(N);
    h_values[0] = 'a'; h_values[1] = 'b'; h_values[2] = 'c';
    h_values[3] = 'd'; h_values[4] = 'e'; h_values[5] = 'f';

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

    for (int i=0; i<N; ++i)
    {
        std::cout << h_keys[i] << ":" << h_values[i] << std::endl;
    }


    std::cout << "Device:" << std::endl;
    thrust::device_vector<int> d_keys(N);
    d_keys[0] = 1; d_keys[1] = 4; d_keys[2] = 2; 
    d_keys[3] = 8; d_keys[4] = 5; d_keys[5] = 7;

    thrust::device_vector<char> d_values(N);
    d_values[0] = 'a'; d_values[1] = 'b'; d_values[2] = 'c';
    d_values[3] = 'd'; d_values[4] = 'e'; d_values[5] = 'f';

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    for (int i=0; i<N; ++i)
    {
        std::cout << d_keys[i] << ":" << d_values[i] << std::endl;
    }

    return 0;
}
