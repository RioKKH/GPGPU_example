#include <stdio.h>
#include <thrust/host_vector.h>

int main(void)
{
    // C言語の標準的な配列
    int vec1[3];
    // thrustのベクトル(ホストに確保)
    thrust::host_vector<int> vec2(3);

    vec1[0] = 10;
    vec1[1] = 20;
    vec1[2] = 30;

    // 配列名(ポインタ)をthrust関数の引数として利用
    thrust::copy(vec1, vec1+3, vec2.begin());

    for (int i=0; i<3; i++)
    {
        printf("%d\n", vec2[i]);
    }
    return 0;
}
