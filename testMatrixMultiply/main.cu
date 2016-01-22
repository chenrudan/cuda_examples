///
/// \file main.cu 
/// \brief This file compare different implementation of matrix multiply
///
/// \author Rudan Chen
/// \date 2016-01-21

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include "multiply_kernel.cuh"

using namespace std;

///
/// This function malloc memory on gpu with exception.
///
float* mallocOnGpu(const int len){
	float* data;
	try{
		cudaError_t status;
		status = cudaMalloc((void**) &data, \
				len * sizeof(float));
		throw status;
	}   
	catch(cudaError_t status){
		if (status != cudaSuccess) {
			fprintf(stderr, "!!!! device memory allocation error\n");
			exit(EXIT_FAILURE);
		}
	}   
	return data;
}

///
/// This function call the cublas library function to implement 
///	mat multiply, which represents the standard.
///
void blasVersion(const float *x_gpu, const float *w_gpu, float *y_gpu, \
		const int M, const int K, const int N){

	cublasHandle_t handle;
	cublasCreate(&handle);
	float scale = 1;
	float scale_tar = 0;

	clock_t t = clock();
	for(int i=0; i<100; i++){
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &scale, \
	    	        w_gpu, N, x_gpu, K, &scale_tar, y_gpu, N);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	cout << "cu: " << ((float)t/CLOCKS_PER_SEC)/100 << "s.\n";
}

void v1(const float *x_gpu, const float *w_gpu, float *y_gpu, const int M, \
		const int K, const int N, const int num_threads){
	clock_t t;
	t = clock();
	for(int i=0; i<100; i++){
		kComputeMatMultiply_v1<<<(M*N)/num_threads, num_threads>>>(x_gpu, w_gpu, y_gpu, M, K, N);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	cout << "v1: " << ((float)t/CLOCKS_PER_SEC)/100 << "s.\n";
}

void v2(const float *x_gpu, const float *w_gpu, float *y_gpu, const int M, \
		const int K, const int N, const int num_threads){
	clock_t t;
	t = clock();
	dim3 num_blocks = dim3(M,N);
	for(int i=0; i<10; i++){
		kComputeMatMultiply_v2<<<num_blocks, num_threads, sizeof(float)*K>>>(x_gpu, w_gpu, y_gpu, K, N);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	cout << "v2: " << ((float)t/CLOCKS_PER_SEC)/10 << "s.\n";
}

void v3(const float *x_gpu, const float *w_gpu, float *y_gpu, const int M, \
		const int K, const int N, const int num_threads){
	clock_t t;
	t = clock();
	for(int i=0; i<100; i++){
		kComputeMatMultiply_v3<<<M, num_threads, sizeof(float)*K>>>(x_gpu, w_gpu, y_gpu, K, N);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	cout << "v3: " << ((float)t/CLOCKS_PER_SEC)/100 << "s.\n";
}

void v4(const float *x_gpu, const float *w_gpu, float *y_gpu, const int M, \
		const int K, const int N){
	clock_t t;
	t = clock();
	for(int i=0; i<100; i++){
		kComputeMatMultiply_v4<<<(M*N)/(16*256), 128>>>(x_gpu, w_gpu, y_gpu, M, K, N);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	cout << "v4: " << ((float)t/CLOCKS_PER_SEC)/100 << "s.\n";
}

#define M_TYPES 1
#define N_TYPES 1
#define THREAD_TYPES 5

int M_set[9] = {1024};
int N_set[4] = {1024};
int threads_set[8] = {64, 128, 256, 512, 1024};

void matMultiply(const int M, const int K, const int N, const int num_threads){
	cout << M << ", " <<K << ", "<< N <<", " << num_threads << endl;

	float *x_cpu = new float[M*K];
	float *w_cpu = new float[K*N];
	float *y_cpu = new float[M*N];
	
	float *x_gpu = mallocOnGpu(M*K);
	float *w_gpu = mallocOnGpu(K*N);
	float *y_gpu = mallocOnGpu(M*N);


	for(int i=0; i<M*K; i++){
		x_cpu[i] = 1;
	}
	for(int i=0; i<N*K; i++){
		w_cpu[i] = 1;
	}
	for(int i=0; i<N*M; i++){
		y_cpu[i] = 1;
	}

	cudaMemcpy(x_gpu, x_cpu, sizeof(float)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, w_cpu, sizeof(float)*K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y_cpu, sizeof(float)*M*N, cudaMemcpyHostToDevice);

	blasVersion(x_gpu, w_gpu, y_gpu, M, K, N);
	v1(x_gpu, w_gpu, y_gpu, M, K, N, num_threads);
	v2(x_gpu, w_gpu, y_gpu, M, K, N, num_threads);
	v3(x_gpu, w_gpu, y_gpu, M, K, N, num_threads);

	for(int i=0; i<N*M; i++){
		y_cpu[i] = 1;
	}
	cudaMemcpy(y_gpu, y_cpu, sizeof(float)*M*N, cudaMemcpyHostToDevice);
	v4(x_gpu, w_gpu, y_gpu, M, K, N);

	delete[] x_cpu;
	delete[] w_cpu;
	delete[] y_cpu;

	cudaFree(x_gpu);
	cudaFree(w_gpu);
	cudaFree(y_gpu);
}

int main(){
	for(int i=0; i<M_TYPES; i++){
		for(int j=0; j<N_TYPES; j++){
			for(int k=0; k<THREAD_TYPES; k++){
				matMultiply(M_set[i], 3072, N_set[j], threads_set[k]);
				cout << endl;
			}
		}
	}
	cout << "done\n";

	return 0;
}
