#include <stdio.h>
#include <stdlib.h>

#define N (1024*1024)
#define Nbytes (N*sizeof(float))

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
    float *a, *b, *c;

    a = (float *)malloc(Nbytes);
    b = (float *)malloc(Nbytes);
    c = (float *)malloc(Nbytes);

    init(a, b, c);
    add(a, b, c);

    free(a);
    free(b);
    free(c);

    return 0;
}
