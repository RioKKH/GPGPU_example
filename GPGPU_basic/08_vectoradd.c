#include <stdio.h>
#include <time.h>

#define N (1024*1024)

int main()
{
    float a[N], b[N], c[N];
    int i;
    clock_t start_c, stop_c;
    float time_s;

    start_c = clock(); // プログラム実行時からの経過時間を取得
    for (i = 0; i < N; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    for (i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
    stop_c = clock(); // プログラム実行時からの経過時間を取得

    // for (i = 0; i < N; i++)
    // {
    //     printf("%f+%f=%f\n", a[i], b[i], c[i]);
    // }

    // 処理に要した時間を秒に変換
    time_s = (stop_c = start_c)/(float)CLOCKS_PER_SEC;
    printf("%f\n", time_s);
    return 0;
}
