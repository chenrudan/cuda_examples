测试一下用gpu来处理大量数据计算，不同版本的提升速度
数据自己模拟，输入大小为20000，输出大小为1280*1200*3
每一个输出都是由所有输入算出

将输入数据放入共享内存             v1_kernel cost: 272.9ms.
采用跟v1一致的方法减少block的个数  v2_kernel cost: 272.2ms.
数据直接从全局内存中读取           v3_kernel cost: 577.9ms.
数据存入共享内存再进行reduce       v4_kernel cost: 13.5ms.
共享内存从纹理映射中获取数据       v5_kernel cost: 13.5ms.

