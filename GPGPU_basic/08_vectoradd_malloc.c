#include <stdio.h> // printf
#include <stdlib.h> // malloc

#define N (1024*1024)
#define Nbytes (N*sizeof(float))

int main()
{
    float *a, *b, *c;
    int i;

    a = (float *)malloc(Nbytes);
    b = (float *)malloc(Nbytes);
    c = (float *)malloc(Nbytes);

    for (i=0; i<N; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    for (i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }

    for (i=0; i<N; i++)
    {
        printf("%f+%f=%f\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}

