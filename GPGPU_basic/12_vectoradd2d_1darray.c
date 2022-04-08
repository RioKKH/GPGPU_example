#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Nx (1024)
#define Ny (1024)
#define Nbytes (Nx * Ny * sizeof(float))

void init(float *a, float *b, float *c)
{
    int i, j, ij;
    for (i = 0; i < Nx; i++)
    {
        for (j = 0; j < Ny; j++)
        {
            // 1次元配列の場合、どちらを内側のループで回すかで
            // スピードに差は出るか？ --> 当然差は出る
            // ij = i * Ny + j; // much faster
            ij = i + Nx * j;    // much slower

            a[ij] = 1.0;
            b[ij] = 2.0;
            c[ij] = 0.0;
        }
    }
}

void add(float *a, float *b, float *c)
{
    int i, j, ij;
    for (i = 0; i < Nx; i++)
    {
        for (j = 0; j < Ny; j++)
        {
            // ij = i*Ny + j; // much faster
            ij = i + Nx * j;  // much slower
            c[ij] = a[ij] + b[ij];
        }
    }
}

int main(void)
{
    float *a, *b, *c;
    clock_t start_c, middle_c, stop_c;
    float time_s1, time_s2;

    a = (float *)malloc(Nbytes);
    b = (float *)malloc(Nbytes);
    c = (float *)malloc(Nbytes);

    start_c = clock();
    init(a, b, c);
    middle_c = clock();
    add(a, b, c);
    stop_c = clock();

    time_s1 = (middle_c - start_c) / (float)CLOCKS_PER_SEC;
    time_s2 = (stop_c - middle_c) / (float)CLOCKS_PER_SEC;

    printf("%f,%f\n", time_s1, time_s2);

    free(a);
    free(b);
    free(c);
    return 0;
}

