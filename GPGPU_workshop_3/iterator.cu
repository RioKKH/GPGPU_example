#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main()
{
    thrust::host_vector<int> vec1(3);
    thrust::host_vector<int> vec2(3);


    vec1[0] = 10;
    vec1[1] = 20;
    vec1[2] = 30;

    // thrustのvectorは範囲for文が使える
    // vec1.begin(): vectorの最初の要素を返す
    // vec1.end(): vectorの最後の要素の1つ後ろを返す
    thrust::copy(vec1.begin(), vec1.end(), vec2.begin());
    for (auto a : vec2)
    {
        std::cout << a << std::endl;
    }

    // device上のvectorに対しても範囲for文が使える
    thrust::device_vector<int> d_vec = vec2;
    for (auto a : d_vec)
    {
        std::cout << a << std::endl;
    }

    return 0;
}
