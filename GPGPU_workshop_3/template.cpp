#include <iostream>

template <typename T>
T add(T a, T b)
{
    return a + b;
}

int main(void)
{
    int ia = 1;
    int ib = 2;

    float fa = 1.0f;
    float fb = 2.0f;

    add<int>(ia, ib); // typename Tが全てintになる
    add<float>(fa, fb); // typename Tが全てfloatになる

    return 0;
}

