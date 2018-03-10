#include <stdio.h>
#include <stdlib.h>
#include <assert.h> 
#include "qdbmp.h"

#define CHANNELS	3
#define N_THREADS	32.0
#define THRESHOLD	75
#define MAX(x,y) (((x) > (y) ? (x) : (y)))
#define MIN(x,y) (((x) < (y) ? (x) : (y)))


__global__ void convertToGray(unsigned char* rgb, unsigned char *gray, size_t height, size_t width){

	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < width && Row < height) {
		gray[Row * width + Col] = MIN(255, (unsigned char) 
											0.21 * rgb[(Row * width + Col) * CHANNELS] + 
											0.72 * rgb[(Row * width + Col) * CHANNELS + 1] + 
											0.07 * rgb[(Row * width + Col) * CHANNELS + 2]);
	}
}

__global__ void findEdges(unsigned char* gray, unsigned char* edges, int height, int width) {
	const int sobelX[3][3] = {{1,0,-1}, {2,0,-2}, {1,0,-1}};
	const int sobelY[3][3] = {{1,2,1}, {0, 0,0}, {-1,-2,-1}};
	
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (Col < width && Row < height) {
		
		int kernelW = 1;
		int convRow, convCol, sobelXSum, sobelYSum;
		sobelXSum = 0;
		sobelYSum = 0;
		for (convRow = 0 - kernelW; convRow < kernelW + 1; ++convRow){
			for (convCol = 0 - kernelW; convCol < kernelW + 1; ++convCol) {

				if (convRow + Row > -1 && convRow + Row < height && 
					convCol + Col > -1 && convCol + Col < width){
					
					sobelXSum += sobelX[convRow + 1][convCol + 1] * 
								 gray[(convRow + Row) * width + (convCol + Col)]; 
								
					sobelYSum += sobelY[convRow + 1][convCol + 1] * 
								 gray[(convRow + Row) * width + (convCol + Col)];

				}

			}
		}
		edges[Row * width + Col] = 
			sqrtf(powf(sobelXSum, 2) + powf(sobelYSum, 2)) > THRESHOLD ? 255 : 0;

	}
}

int main(int argc, char**argv){
	char srcName[100];
	char destName[100];
	if (argc >= 2) {
		strcpy(srcName, argv[1]);
		strcpy(destName, argv[2]);
	} else {
		printf("Run the program %s (input) (output)\n", argv[0]);
		exit(0);
	}
	
	BMP *bmp = BMP_ReadFile(srcName);
	BMP_CHECK_ERROR(stdout, -1);

	int imageWidth = BMP_GetWidth(bmp);
	int imageHeight = BMP_GetWidth(bmp);
	int imageDepth = BMP_GetDepth(bmp);

	assert(imageDepth == 24);

	unsigned char *origImage, *edges;
	unsigned char *d_origImage, *d_gray, *d_edges;

	origImage = (unsigned char*)malloc(imageHeight * imageWidth * CHANNELS);
	edges = (unsigned char *)malloc(imageHeight * imageWidth);

	cudaMalloc((void **)&d_origImage, imageHeight * imageWidth * CHANNELS);
	cudaMalloc((void **)&d_gray, imageHeight * imageWidth);
	cudaMalloc((void **)&d_edges, imageHeight * imageWidth);

	int i, j;
	for (i = 0; i < imageWidth; i++) {
		for (j = 0; j < imageHeight; j++) {
			int offset = (j * imageWidth + i) * CHANNELS;
			BMP_GetPixelRGB(bmp, i, j, &origImage[offset],&origImage[offset+1], &origImage[offset+2]);		  }
	}

	cudaMemcpy(d_origImage, origImage, imageHeight * imageWidth * CHANNELS, cudaMemcpyHostToDevice);

	dim3 DimGrid(ceil(imageWidth / N_THREADS), ceil(imageHeight / N_THREADS), 1);
	dim3 DimBlock(N_THREADS, N_THREADS, 1);

	convertToGray<<<DimGrid, DimBlock>>>(d_origImage, d_gray, imageHeight, imageWidth);
	findEdges<<<DimGrid, DimBlock>>>(d_gray, d_edges, imageHeight, imageWidth);

	cudaMemcpy(edges, d_edges, imageHeight * imageWidth, cudaMemcpyDeviceToHost);

	BMP *destBmp = BMP_Create(imageWidth, imageHeight, 8);
	
	for (i = 0; i < 256; i++) {
		BMP_SetPaletteColor(destBmp, i, i ,i, i);
	}

	for (i = 0; i < imageWidth; i++){
		for (j = 0; j < imageHeight; j++) {
			int offset = (j * imageWidth + i);
			BMP_SetPixelIndex(destBmp, i, j, edges[offset]);
		}
	}


	BMP_WriteFile(destBmp, destName);

	return 0;

}
