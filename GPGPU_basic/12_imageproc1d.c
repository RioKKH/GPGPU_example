#include <stdio.h>
#include <stdlib.h>

#define WHITE (255)
#define BLACK (0)
#define WIDTH (64)
// #define HEIGHT WIDTH
#define Nbytes (WIDTH * sizeof(unsigned char))
// #define Nbytes (WIDTH * HEIGHT * sizeof(unsigned char))


void create(unsigned char *);
void copy(unsigned char *, unsigned char *);
void print(unsigned char *);

void negative(unsigned char *, unsigned char *);
void hreflect(unsigned char *, unsigned char *);
void boxfilter(unsigned char *, unsigned char *, float *);
void mosaic(unsigned char *, unsigned char *, int);


int main(void)
{
    unsigned char *p = (unsigned char *)malloc(Nbytes);
    unsigned char *filtered = (unsigned char *)malloc(Nbytes);

    // ここで空間フィルタのカーネルを宣言
    create(p);
    copy(p, filtered);
// ここで処理をおこない、結果をfilteredに格納 
    // 画面に各画素の値を表示
    print(filtered);

    return 0;
}


void negative(unsigned char *p, unsigned char *filtered)
{
    int i;
    for (i = 0; i < WIDTH; i++)
    {
        filtered[i] = WHITE - p[i];
    }
}

void hreflect(unsigned char *p, unsigned char *filtered)
{
    int i;
    for (i =0; i < WIDTH; i++)
    {
        // (WIDTH -1) - iは反転後の横位置
        filtered[(WIDTH - 1) - i] = p[i];
    }
}

void boxfilter(unsigned char *p, unsigned char *filtered, float *filter)
{
    int i;
    int result = BLACK;

    for (i = 1; i < WIDTH - 1; i++) // 端の画像は処理をしない
    {
        result = filter[0]*p[i-1] + filter[1]*p[i] + filter[2]*p[i+1];
        // フィルタ後の値が負になれば0に切り上げ、255を超えたら255に収める
        if (result < BLACK)
        {
            result = 0;
        }
        if (result > WHITE)
        {
            result = WHITE;
        }
        filtered[i] = (unsigned char)result;
    }
}

void mosaic(unsigned char *p, unsigned char *filtered, int mosaicSize)
{
    int i, isub, average;
    for (i = 0; i < WIDTH; i += mosaicSize) // 少領域を移動するループ
    {
        // 少領域内の画素の平均値を計算
        average = 0;
        for(isub = 0; isub<mosaicSize; isub++)
        {
            average += p[(i + isub)];
        }
        average /= mosaicSize;

        // 少領域内の全画素を平均値で塗りつぶす
        for (isub = 0; isub<mosaicSize; isub++)
        {
            filtered[(i + isub)] = (unsigned char)average;
        }
    }
}


// 画像の作成
void create(unsigned char *p)
{
    int i, x_origin;
    for (i = 0; i < WIDTH; i++)
    {
        p[i] = WHITE;
    }

    x_origin= 3 * WIDTH / 16;
    for (i = 0; i < 6 * WIDTH / 16; i++)
    {
        p[i + x_origin] = BLACK;
    }
}

// 画像の内容をコピー
void copy(unsigned char *src, unsigned char *dst)
{
    int i;
    for (i = 0; i < WIDTH; i++)
    // for (i = 0; i < WIDTH*HEIGHT; i++)
    {
        dst[i] = src[i];
    }
}

// 画像の内容を画面に出力(gnuplotで表示するよう)
void print(unsigned char *p)
{
    int i;
    // int i, j;
    // for (j = 0; j < HEIGHT; j++)
    // {
        for (i = 0; i < WIDTH; i++)
        {
            // 横位置、縦位置、画像の色情報
            printf("%d %d\n", i, p[i]);
            // printf("%d %d %d\n", i, j, p[i + WIDTH*j]);
        }
        // 1行分表示したら改行を入れる(gnuplotで必要)
        printf("\n");
    // }
}

