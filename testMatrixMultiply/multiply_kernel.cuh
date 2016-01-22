///
/// \file multiply_kernel.cuh
/// \brief This file provide different kernel function declarations \
/// of matrix multiply. a is M*K, b is K*N. c = a*b, c is M*N.
///
/// \author Rudan Chen
/// \date 2016-01-21

///
/// This version is the simplest, one thread computes 
/// one point of c.
///
__global__ void kComputeMatMultiply_v1(const float *a, const float *b, \
		float *c, const int M, const int K, const int N);

///
/// This function means one block computes one point of c.
///
__global__ void kComputeMatMultiply_v2(const float *a, const float *b, \
		float *c, const int K, const int N);

///
/// This function means one block computes one row of c.
///
__global__ void kComputeMatMultiply_v3(const float *a, const float *b, \
		float *c, const int K, const int N);

///
/// This function adopt an acceleration algorithm.
///
__global__ void kComputeMatMultiply_v4(const float *a, const float *b, \
		float *c, const int M, const int K, const int N);
