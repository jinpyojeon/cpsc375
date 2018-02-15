#include <stdio.h>
#include <stdlib.h>
#include <assert.h> 
#include "qdbmp.h"

#define CHANNELS	3
#define BLUR_RADIUS	5

__global__ void blurKernel(unsigned char* input, unsigned char* output, 
			 int height, int width, int batchSize) {
	assert(batchSize % 2 != 0);
	
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < width && Row < height) {
		int redPixVal = 0;
		int redPixels = 0;
		
		int greenPixVal = 0;
		int greenPixels = 0;
		
		int bluePixVal = 0;
		int bluePixels = 0;

		int  blurRow, blurCol;

		int batchW = batchSize / 2;
		for (blurRow = 0 - batchW; blurRow < batchW + 1; ++blurRow){
			for (blurCol = 0 - batchW; blurCol < batchW + 1; ++blurCol) {
				
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;
				int offset = (curRow * width + curCol) * CHANNELS;

				if (curRow > -1 && curRow < height && 
					curCol > -1 && curCol < width){
					
					redPixVal += input[offset];
					redPixels++;

					greenPixVal += input[offset + 1];
					greenPixels++;

					bluePixVal += input[offset + 2];
					bluePixels++;
				}

			}
		}

		int offset = (Row * width + Col) * CHANNELS;
		output[offset] = (unsigned char)(redPixVal / redPixels);
		output[offset + 1] = (unsigned char)(greenPixVal / greenPixels);
		output[offset + 2] = (unsigned char)(bluePixVal / bluePixels);
	}
}

int main(int argc, char**argv){
	char srcName[100];
	char destName[100];
	if (argc >= 2) {
		strcpy(srcName, argv[1]);
		strcpy(destName, argv[2]);
	}
	
	BMP *bmp = BMP_ReadFile(srcName);
	BMP_CHECK_ERROR(stdout, -1);

	int imageWidth = BMP_GetWidth(bmp);
	int imageHeight = BMP_GetWidth(bmp);
	int imageDepth = BMP_GetDepth(bmp);

	assert(imageDepth == 24);

	unsigned char *origImage, *blurredImage;
	unsigned char *d_origImage, *d_blurredImage;

	origImage = (unsigned char*)malloc(imageHeight * imageWidth * CHANNELS);
	blurredImage = (unsigned char *)malloc(imageHeight * imageWidth * CHANNELS);
	
	cudaMalloc((void **)&d_origImage, imageHeight * imageWidth * CHANNELS);
	cudaMalloc((void **)&d_blurredImage, imageHeight * imageWidth * CHANNELS);
	
	int i, j;
	for (i = 0; i < imageWidth; i++) {
		for (j = 0; j < imageHeight; j++) {
			int offset = (j * imageWidth + i) * CHANNELS;
			BMP_GetPixelRGB(bmp, i, j, &origImage[offset], &origImage[offset+1], &origImage[offset+2]);	
		}
	}

	cudaMemcpy(d_origImage, origImage, imageHeight * imageWidth * CHANNELS, cudaMemcpyHostToDevice);

	float threadNum = 32;
	dim3 DimGrid(ceil(imageWidth / threadNum), ceil(imageHeight / threadNum), 1);
	dim3 DimBlock(threadNum, threadNum, 1);
	blurKernel<<<DimGrid, DimBlock>>>(d_origImage, d_blurredImage, imageHeight, imageWidth, BLUR_RADIUS);

	cudaMemcpy(blurredImage, d_blurredImage, imageHeight * imageWidth * CHANNELS, cudaMemcpyDeviceToHost);

	BMP *destBmp = BMP_Create(imageWidth, imageHeight, 8 * CHANNELS);
	
	for (i = 0; i < imageWidth; i++){
		for (j = 0; j < imageHeight; j++) {
			int offset = (j * imageWidth + i) * CHANNELS;
			BMP_SetPixelRGB(destBmp, i, j, blurredImage[offset], blurredImage[offset+1], blurredImage[offset+2]);
		}
	}

	BMP_WriteFile(destBmp, destName);

	return 0;

}
