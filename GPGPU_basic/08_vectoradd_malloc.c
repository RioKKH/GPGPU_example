#include <stdio.h> // printf
#include <stdlib.h> // malloc
#include <time.h> // clock(), time_s

#define N (1024*1024)
#define Nbytes (N*sizeof(float))

int main()
{
    float *a, *b, *c;
    int i;
    clock_t start_c, stop_c;
    float time_s;

    a = (float *)malloc(Nbytes);
    b = (float *)malloc(Nbytes);
    c = (float *)malloc(Nbytes);

    start_c = clock();
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
    stop_c = clock();

    // for (i=0; i<N; i++)
    // {
    //     printf("%f+%f=%f\n", a[i], b[i], c[i]);
    // }

    time_s = (stop_c - start_c)/(float)CLOCKS_PER_SEC;
    printf("%f\n", time_s);

    free(a);
    free(b);
    free(c);

    return 0;
}

