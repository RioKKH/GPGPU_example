#include <iostream>

__global__ void kernel()
{
}

int main()
{
    kernel<<<1, 1>>>();
    std::cout << "hello, world" << std::endl;

    return 0;
}
