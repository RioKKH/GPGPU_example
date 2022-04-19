#include <stdio.h>
#ifndef ___BMPHEADER__
#define ___BMPHEADER__

// 構造体のサイズ
// メンバ変数サイズの合計にならない場合がある
// アクセス最適化のためのパディングが原因
// これは多くの場合4バイト境界にアライメントされる
// そこで以下のpragma命令を書く事でアライメントを変更することが出来る
// #pragma pack(push, 1) ~ #pragma pack(pop)
// pack ~ popで挟まれた範囲では1バイト境界にアライメントされる
// これによって実質的に構造体のパディングを防止することが出来る
#pragma pack(push, 1)

// ビットマップファイルヘッダ
typedef struct {
	unsigned short	bfType;	// ファイルの種類
	unsigned int	bfSize;	// ファイルサイズ
	unsigned short	bfReserverd1;	// 予約領域1
	unsigned short	bfReserverd2;	// 予約領域2
	unsigned int	bfOffBits;		// オフセット
} BmpFileHeader;

// ビットマップ情報ヘッダ
typedef struct {
	unsigned int	biSize;	// 情報ヘッダサイズ
			 int	biWidth;	// 幅
			 int	biHeight;	// 高さ
	unsigned short	biPlanes;	// プレーン数
	unsigned short	biBitCount;	// 色ビット数
	unsigned int	biCompression;	// 圧縮形式
	unsigned int	biSizeImage;	// 画像サイズ
			 int	biXPelsPerMeter;	// 解像度
			 int	biYPelsPerMeter;	// 解像度
	unsigned int	biClrUsed;	// 使用色数
	unsigned int	biClrImportant;	// 重要色
} BmpInfoHeader;

// 1画素の情報
typedef struct {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} BGR;

#pragma pack(pop)

typedef struct {
	BGR *pixel;
	int width;
	int height;
} Img;

#endif // ___BMPHEADER__
	
