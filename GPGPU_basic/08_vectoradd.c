#include <stdio.h>

#define N (1024*1024)

int main()
{
    float a[N], b[N], c[N];
    int i;

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

    for (i = 0; i < N; i++)
    {
        printf("%f+%f=%f\n", a[i], b[i], c[i]);
    }

    return 0;
}
