#include <stdio.h>
#include <stdlib.h>

__global__ void addKernel(const double* x1, const double* x2, double result, int length){

}

void cudaAdd(const double* x1, const double* x2, double* result, int length){
	//set block size, i.e. number of threads in each block
	int block_size = 1024;
	//set grid size, i.e. number of blocks
	int grid_size = length / block_size + 1;
	
	//allocate some memory on the device
	double* d_x1, d_x2, d_result;
	int num_bytes = length*sizeof(double);
	cudaMalloc(&d_x1, num_bytes);
	cudaMalloc(&d_x2, num_bytes);
	cudaMalloc(&d_result, num_bytes);
	
	//copy data to the device
	cudaMemcpy(&)
	
	//execute the kernel
	
	//copy data back from the device
}

void serialAdd(const double* x1, const double* x2, double* result, int length){
	for(int i = 0; i < length; i++){
		result[i] = x1[i] + x2[i];
	}
}

int main(){

	//create and initialize vectors
	
	int size = 60;
	int num_bytes = size * sizeof(double);
	double* x1 = (double*) malloc(num_bytes);
	double* x2 = (double*) malloc(num_bytes);
	double* result = (double*) malloc(num_bytes);
	for(int i = 0; i < size; i++){
		x1[i] = i;
		x2[i] = i*i;
	}
	
	//call cudaAdd
	
	serialAdd(x1, x2, result, size);
	
	//print result
	
	for(int i = 0; i < size; i++){
		printf("%f", result[i]);
		printf("\n");
	}
	
	//free allocated memory
	free(x1);
	free(x2);
	free(result);
	
}
