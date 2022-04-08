#include <iostream>
#include <numeric> // accumulate()

template <typename T, int N>
T reduction_cpu(const T* x)
{
    T sum{};
    for (int i = 0; i < N; ++i)
    {
        sum += x[i];
    }
    return sum;
}

int main(void)
{
    const int N = 5;
    int x[N] = {1, 2, 3, 4, 5};
    std::cout << "reduction_cpu:" << reduction_cpu<int, N>(x) << std::endl;
    std::cout << "reduction_cpu_stl:" << std::accumulate(x, x + N, 0) << std::endl;

    return 0;
}
