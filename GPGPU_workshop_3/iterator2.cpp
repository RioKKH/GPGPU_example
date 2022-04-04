#include <iostream>

int main(void)
{
    int vec1[3];
    int vec2[3];

    vec1[0] = 10;   vec1[1] = 20;   vec1[2] = 30;
    vec2[0] =  0;    vec2[1] = 0;   vec2[2] =  0;

    for (int i = 0; i < 3; ++i) // 標準的なC言語的書き方
    {
        vec2[i] = vec1[i]; // ループと配列、配列添字を利用
    }

    for (int i = 0; i < 3; ++i)
    {
        std::cout << vec2[i] << std::endl;
    }

    // ポインターを使った処理
    int *src = vec1;    // vec1の先頭要素、終端要素+1を設定
    int *end = src+3;   // コピー先の先頭要素を設定
    int *dst = vec2;
    // ポインターに対する演算を使った書き方
    while(src != end)
    {
        *(dst++) = *(src++);
    }

    for (int i = 0; i < 3; i++)
    {
        std::cout << vec2[i] << std::endl;
    }

    return 0;
}
