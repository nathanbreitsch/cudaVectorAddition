#include <stdio.h>
#include <stdlib.h>

// questions:
// 1. if I had a bunch of vectors to add in succession, would there be any benefit in keeping the memory allocated
// and doing all the additions before freeing memory?  What would be the impact?

// 2. given that thread creation has some overhead, would it make sense to do multiple additions per thread instead of one?  What is optimal num_add?

// todo: modify this code and use timer to see which parameters are optimal for card 

// 3. if i pass a device memory location into *result for cudaAdd, does it store the result on the card? (could probs google this, but... todo: try it out)


__global__ void addKernel(float *x1, float*x2, float *result, int length){
	//get unique index on which to compute
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	//add vectors on this index
	if(index < length){
		result[index] = x1[index] + x2[index];
	}
}

void cudaAdd(const float *x1, const float *x2, float *result, int length){
	//set block size, i.e. number of threads in each block
	int block_size = 512;
	//set grid size, i.e. number of blocks
	int grid_size = length / block_size + 1;
	
	//allocate some memory on the device
	float *d_x1, *d_x2, *d_result;
	
	size_t num_bytes = length*sizeof(float);
	cudaMalloc(&d_x1, num_bytes);
	cudaMalloc(&d_x2, num_bytes);
	cudaMalloc(&d_result, num_bytes);
	
	//copy data to the device
	cudaMemcpy(&d_x1, &x1, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(&d_x2, &x2, num_bytes, cudaMemcpyHostToDevice);
	
	//start the kernel
	addKernel<<<grid_size, block_size>>>(d_x1, d_x2, d_result, length);
	
	//copy data back from the device
	cudaMemcpy(d_result, result, num_bytes, cudaMemcpyDeviceToHost);
	
	//free memory
	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_result);
	
}

void serialAdd(const float *x1, const float *x2, float *result, int length){
	for(int i = 0; i < length; i++){
		result[i] = x1[i] + x2[i];
	}
}

int main(){

	//create and initialize vectors
	
	int size = 60;
	int num_bytes = size * sizeof(float);
	float *x1 = (float*) malloc(num_bytes);
	float *x2 = (float*) malloc(num_bytes);
	float *result = (float*) malloc(num_bytes);
	for(int i = 0; i < size; i++){
		x1[i] = i;
		x2[i] = i*i;
	}
	
	//call cudaAdd
	
	cudaAdd(x1, x2, result, size);
	
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
