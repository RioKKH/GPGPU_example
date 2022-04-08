#include <stdio.h>
#include <time.h>

#define Nx (1024)
#define Ny (1024)

void init(float a[Nx][Ny], float b[Nx][Ny], float c[Nx][Ny])
{
    int i, j;
    // i とjのループを入れ替えるとどうなるか
    for (i=0; i<Nx; i++)
    {
        for (j=0; j<Ny; j++)
        {
            a[i][j] = 1.0;
            b[i][j] = 2.0;
            c[i][j] = 0.0;
        }
    }
}

void add(float a[Nx][Ny], float b[Nx][Ny], float c[Nx][Ny])
{
    int i, j;
    // i とjのループを入れ替えるとどうなるか
    // for (j=0; j<Nx; j++) // faster
    for (i=0; i<Nx; i++)    // slower
    {
        // for (i=0; i<Ny; i++) // faster
        for (j=0; j<Ny; j++)    // slower
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

int main(void)
{
    float a[Nx][Ny], b[Nx][Ny], c[Nx][Ny];
    clock_t start_c, middle_c, stop_c;
    float time_s1, time_s2;

    start_c = clock();
    init(a, b, c);
    middle_c = clock();
    add(a, b, c);
    stop_c = clock();
    time_s1 = (middle_c - start_c)/(float)CLOCKS_PER_SEC;
    time_s2 = (stop_c - middle_c)/(float)CLOCKS_PER_SEC;
    printf("%f,%f\n", time_s1, time_s2);
    
    return 0;
}

