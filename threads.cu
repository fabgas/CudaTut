/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */
extern "C"
__global__ void threads(float *g_idata,unsigned int n)
{
   

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
	if (i<n) {
		printf("Hello world %d - %d : %f\n",tid,i,g_idata[i]);
	}
}