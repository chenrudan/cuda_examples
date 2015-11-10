///
/// \@file: v1_copy.cu
///

#include<iostream>
#include<cuda_runtime.h>

texture<int, cudaTextureType1D, cudaReadModeElementType> tex;

/// 每个thread计算一个输出像素点，重复将全局内存复制到共享内存中
///
__global__ void v1_kernel(const int *in, int *out, const int in_length, \
		const int box_in_length, const int out_length){
	extern __shared__ int sh_in[];
	const int copy_time = in_length / box_in_length + (in_length % box_in_length ? 1 : 0);

	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
	int sum = 0;
	if(thread_id < out_length){
		for(int i=0; i < copy_time; i++){
			const int copy_length = box_in_length > (in_length-i*box_in_length) \
									? in_length-i*box_in_length : box_in_length;
			const int times = copy_length / blockDim.x + (copy_length % blockDim.x ? 1 : 0);
			for(int j=0; j<times; j++){
				const int tmp = blockDim.x > copy_length-j*blockDim.x \
								? copy_length-j*blockDim.x : blockDim.x;
				if(threadIdx.x < tmp){
					sh_in[threadIdx.x + j*blockDim.x] = in[i*box_in_length \
														+ threadIdx.x + j*blockDim.x];
				}
			}

			for(int j=0; j < copy_length; j++){
				sum += sh_in[j];		
			}
		}
		out[thread_id] = sum;
	}
}

__global__ void v2_kernel(const int *in, int *out, const int in_length, \
		const int box_in_length, const int out_length){
	extern __shared__ int sh_in[];
	const int copy_time = in_length / box_in_length + (in_length % box_in_length ? 1 : 0);

	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
	int sum1 = 0;
	int sum2 = 0;
	if((thread_id*2+1) < out_length){
		for(int i=0; i < copy_time; i++){
			const int copy_length = box_in_length > (in_length-i*box_in_length) \
									? in_length-i*box_in_length : box_in_length;
			const int times = copy_length / blockDim.x + (copy_length % blockDim.x ? 1 : 0);
			for(int j=0; j<times; j++){
				const int tmp = blockDim.x > copy_length-j*blockDim.x \
								? copy_length-j*blockDim.x : blockDim.x;
				if(threadIdx.x < tmp){
					sh_in[threadIdx.x + j*blockDim.x] = in[i*box_in_length \
														+ threadIdx.x + j*blockDim.x];
				}
			}

			for(int j=0; j < copy_length; j++){
				sum1 += sh_in[j];		
			}
			for(int j=0; j < copy_length; j++){
				sum2 += sh_in[j];		
			}
		}
		out[thread_id*2] = sum1;
		out[thread_id*2+1] = sum2;
	}
}

__global__ void v3_kernel(const int *in, int *out, const int in_length, \
		const int out_length){
	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

	int sum = 0;
	if(thread_id < out_length){
		for(int j=0; j < in_length; j++){
			sum += in[j];
		}
		out[thread_id] = sum;
	}
}

__global__ void v4_kernel(const int *in, int *out, const int in_length, \
		const int out_length){
	__shared__ int sh_in[1024];
	const int reduce_length = 1024;
	const int copy_time = in_length / reduce_length + (in_length % reduce_length ? 1 : 0);

	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
	int sum = 0;
	for(int i=0; i < copy_time; i++){
		const int copy_length = reduce_length > (in_length-i*reduce_length) \
								? in_length-i*reduce_length : reduce_length;
		if(threadIdx.x < copy_length){
			sh_in[threadIdx.x] = in[i*reduce_length + threadIdx.x];
		}else{
			sh_in[threadIdx.x] = 0;
		}
		__syncthreads();

		for(int activeThreads = reduce_length/2; activeThreads; \
				activeThreads /= 2){ 
			if(threadIdx.x < activeThreads)
				sh_in[threadIdx.x] += sh_in[threadIdx.x+activeThreads];
			__syncthreads();
		}
		sum += sh_in[0];
		__syncthreads();
	}
	if(thread_id < out_length){
		out[thread_id] = sum;
	}
}

__global__ void v5_kernel(int *out, const int in_length, \
		const int out_length){
	__shared__ int sh_in[1024];
	const int reduce_length = 1024;
	const int copy_time = in_length / reduce_length + (in_length % reduce_length ? 1 : 0);

	const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
	int sum = 0;
	for(int i=0; i < copy_time; i++){
		const int copy_length = reduce_length > (in_length-i*reduce_length) \
								? in_length-i*reduce_length : reduce_length;
		if(threadIdx.x < copy_length){
			sh_in[threadIdx.x] = tex1Dfetch(tex, i*reduce_length + threadIdx.x);
		}else{
			sh_in[threadIdx.x] = 0;
		}
		__syncthreads();

		for(int activeThreads = reduce_length/2; activeThreads; \
				activeThreads /= 2){ 
			if(threadIdx.x < activeThreads)
				sh_in[threadIdx.x] += sh_in[threadIdx.x+activeThreads];
			__syncthreads();
		}
		sum += sh_in[0];
		__syncthreads();
	}
	if(thread_id < out_length){
		out[thread_id] = sum;
	}
}

int main(){

	int *d_in, *d_out;

	const int in_length = 20000;

	const int height = 1980;
	const int width = 1200;
	const int channel = 3;
	const int num_img = 1;

	const int out_length = height*width*channel*num_img;
	int *h_in = new int[in_length];
	int *h_out = new int[out_length];
	for(int i=0; i<in_length; i++){
		h_in[i] = 1;
	}

	cudaMalloc((void**) &d_out, sizeof(int)*out_length);
	cudaMalloc((void**) &d_in, sizeof(int)*in_length);
	cudaMemcpy(d_in, h_in, sizeof(int)*in_length, cudaMemcpyHostToDevice);

	const int box_in_length = 10000;
	clock_t t = clock();


///通过共享内存的方式，将输入加起来
///
///
	for(int i=0; i < 100; i++){
		v1_kernel<<<6961, 1024, sizeof(int)*box_in_length>>>(d_in, d_out, \
				in_length, box_in_length, out_length);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	std::cout << "将输入数据放入共享内存             v1_kernel cost: "<< ((float)t / CLOCKS_PER_SEC)*10 << "ms.\n";

///每个线程处理两个输出
///
///
	t = clock();
	for(int i=0; i < 100; i++){
		v2_kernel<<<3481, 1024, sizeof(int)*box_in_length>>>(d_in, d_out, \
				in_length, box_in_length, out_length);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	std::cout << "采用跟v1一致的方法减少block的个数  v2_kernel cost: "<< ((float)t / CLOCKS_PER_SEC)*10 << "ms.\n";

///直接从全局内存中读取
///
///
	t = clock();
	for(int i=0; i < 100; i++){
		v3_kernel<<<6961, 1024>>>(d_in, d_out, \
				in_length, out_length);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	std::cout << "数据直接从全局内存中读取           v3_kernel cost: " << ((float)t / CLOCKS_PER_SEC)*10 << "ms.\n";

///采用reduce的方式
///
///
	t = clock();
	for(int i=0; i < 100; i++){
		v4_kernel<<<6961, 1024>>>(d_in, d_out, \
				in_length, out_length);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	std::cout << "数据存入共享内存再进行reduce       v4_kernel cost: " << ((float)t / CLOCKS_PER_SEC)*10 << "ms.\n";

///把全局内存映射到纹理内存中
///
///
	cudaBindTexture(0, tex, d_in, sizeof(int)*in_length);

	t = clock();
	for(int i=0; i < 100; i++){
		v5_kernel<<<6961, 1024>>>(d_out, \
				in_length, out_length);
		cudaDeviceSynchronize();
	}
	t = clock() - t;
	std::cout << "共享内存从纹理映射中获取数据       v5_kernel cost: " << ((float)t / CLOCKS_PER_SEC)*10 << "ms.\n";


///查看输出
///
///
	cudaMemcpy(h_out, d_out, sizeof(int)*out_length, cudaMemcpyDeviceToHost);
	for(int i=100; i<110; i++){
		std::cout << h_out[i] << "\n";
	}
	cudaUnbindTexture(tex);
	
	cudaFree(d_in);
	cudaFree(d_out);
	delete[] h_in;
	delete[] h_out;

	return 0;
}
