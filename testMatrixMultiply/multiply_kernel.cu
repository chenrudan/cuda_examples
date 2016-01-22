///
/// \file multiply_kernel.cuh
/// \brief This file provide different kernel function definations \
/// of matrix multiply. a is M*K, b is K*N. c = a*b, c is M*N.
///
/// \author Rudan Chen
/// \date 2016-01-21


__global__ void kComputeMatMultiply_v1(const float *a, const float *b, \
		float *c, const int M, const int K, const int N){
	const int idx = (blockIdx.x%M)*N + (blockIdx.x/M)*blockDim.x + threadIdx.x;
	float result = 0;
	for(int i=0; i<K; i++){
		result += a[(blockIdx.x%M)*K+i]*b[i*N+(blockIdx.x/M)*blockDim.x+threadIdx.x];
	}
	c[idx] = result;
}

__global__ void kComputeMatMultiply_v2(const float *a, const float *b, \
		float *c, const int K, const int N){
	extern __shared__ float result[];
	float local_result=0;
	for(int i=0; (i*blockDim.x+threadIdx.x)<K; i++){
		local_result += a[blockIdx.x*K+i*blockDim.x+threadIdx.x]*b[(i*blockDim.x+threadIdx.x)*N+blockIdx.y];
	}
	result[threadIdx.x] = local_result;
	__syncthreads();
	for(int activeThreads = blockDim.x/2; activeThreads; activeThreads/=2){
		if(threadIdx.x < activeThreads)
			result[threadIdx.x] += result[threadIdx.x + activeThreads];
		__syncthreads();
	}
	if(threadIdx.x == 0)
		c[blockIdx.x*N+blockIdx.y] = result[0];
	__syncthreads();
}

__global__ void kComputeMatMultiply_v3(const float *a, const float *b, \
		float *c, const int K, const int N){
	
	extern __shared__ float sh_a[];  ///save one row of a, shared with b
	const int idx = blockIdx.x*N + threadIdx.x;
	int i = threadIdx.x;
	while(i<K){
		sh_a[i] = a[blockIdx.x*K+i];
		i += blockDim.x;
	}
	for(int j=0; j<(N/blockDim.x); j++){
		float result = 0;
		for(int i=0; i<K; i++){
			result += sh_a[i]*b[i*N + j*blockDim.x + threadIdx.x];
		}
		c[idx + j*blockDim.x] = result;

	}
}

#define ASUB_HEIGHT 16
#define ASUB_WIDTH 32
#define BSUB_HEIGHT 32
#define BSUB_WIDTH 256
#define CSUB_HEIGHT 16
#define CSUB_WIDTH 256

/// thread number of one block is fixed at 128
/// each thread compute 16*2 region of c
/// 
__global__ void kComputeMatMultiply_v4(const float *a, const float *b, \
		float *c, const int M, const int K, const int N){
	__shared__ float sh_a[ASUB_HEIGHT*ASUB_WIDTH];
	float local_c[CSUB_HEIGHT][2];

	const int c_block_row = blockIdx.x / (N/CSUB_WIDTH);
	const int c_block_col = blockIdx.x % (N/CSUB_WIDTH);

	const int v1 = c_block_row*CSUB_HEIGHT; ///v1 is the tmp variable, so as the v2...
	const int v2 = c_block_col*CSUB_WIDTH;
	const int v3 = threadIdx.x*2;
	//copy c to local variable
	for(int i=0; i<CSUB_HEIGHT; i++){
		local_c[i][0] = c[(v1+i)*N + v2 + v3];
		local_c[i][1] = c[(v1+i)*N + v2 + v3 + 1];
	}
	
	for(int i=0; i<(K/ASUB_WIDTH); i++){
		const int v4 = i*ASUB_WIDTH;
		const int v5 = i*BSUB_HEIGHT;

		for(int j=0; j<4; j++){
			int row_id = (threadIdx.x + j*blockDim.x)/ASUB_WIDTH;
			int col_id = (threadIdx.x + j*blockDim.x)%ASUB_WIDTH;
			sh_a[threadIdx.x + j*blockDim.x] = a[(v1+row_id)*K + v4 + col_id];
		}
		__syncthreads();
		for(int k=0; k<BSUB_HEIGHT; k++){
			for(int m=0; m<CSUB_HEIGHT; m++){
				local_c[m][0] += sh_a[m*ASUB_WIDTH + k]*b[(v5 + k)*N \
								 + v2 + v3];
				local_c[m][1] += sh_a[m*ASUB_WIDTH + k]*b[(v5 + k)*N \
								 + v2 + v3 + 1];
			}
		}
		__syncthreads();
	}
	for(int i=0; i<CSUB_HEIGHT; i++){
		c[(v1+i)*N + v2 + v3] = local_c[i][0];
		c[(v1+i)*N + v2 + v3 + 1] = local_c[i][1];
	}
}
