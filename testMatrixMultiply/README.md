
I have tested five kernel function for implementing matrix multiply.

Machine: 6-core Intel Core i7-5930K CPU@3.50GHz + NVIDIA gtx 970 + ubuntu 12.04

Input matrixs is a(M\*K) and b(K\*N), result matrix is c(M\*N).

* blas_version: call the cublas interfaces
* v1: one gpu thread compute one point of c
* v2: one gpu block compute one point of c
* v3: one gpu block compute one row of c
* v4: one gpu block compute one block of c

***Test1*** 

Compare the speed of five kernel function with different size of N. And the number of threads in one block is fixed to 128.

Y axis means the cost time.

|N|blas_version|v1|v2|v3|v4|
|:-----:|:----:|:-----:|:------:|:----:|:---:|
|256|	0.0006|	0.0098	|0.063|	0.0076|	0.0015|
|512|	0.0009|	0.0188|	0.126|	0.0153|	0.0031|
|1024|	0.0019|	0.037|	0.251|	0.0304|	0.005|
|4096|	0.0066|	0.1465|	1.005|	0.1627|	0.0186|

![performance](http://7xkmdr.com1.z0.glb.clouddn.com/testmatmulti1.png)

***Test2***

Compare the influence of different size of threads in one block under v3. 

Value 50,100,250,500,1000 compute M=1024,K=3072,N=1000

Value 64,128,256,512,1024 compute M=1024,K=3072,N=1024

From the result it tells that threads number which is the multiple of 16 outperform others.

|num_threads|cost time|
|:----:|:----:|
|50|	0.0453|
|64|	0.037|
|100|	0.0429|
|128|	0.0371|
|250|	0.0437|
|256|	0.0382|
|500|	0.0635|
|512|	0.0543|
|1000|	0.0657|
|1024|	0.061|


![performance](http://7xkmdr.com1.z0.glb.clouddn.com/testmatmulti2.png)




