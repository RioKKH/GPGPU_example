#include <stdio.h>

#define N (1024*1024)

void init(float *a, float *b, float *c)
{
    int i;

    for (i=0; i<N; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
}

void add(float *a, float *b, float *c)
{
    int i;
    for (i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    float a[N], b[N], c[N];

    init(a, b, c);
    add(a, b, c);

    return 0;
}
