/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */
extern "C"
__global__ void sum(float *g_idata,float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[]; 
    unsigned int tid = threadIdx.x; // thread courant dans le block
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // index g�n�ral
	sdata[tid] = g_idata[i]; // copy vers la shared memory du block
	__syncthreads(); // on attends tous les blocks
	
	if (i >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	for(int s=blockDim.x/2; s >0; s >>= 1) {
		if (tid<s) { // si correspond � un multiple de la dimension
		sdata[tid] += sdata[tid + s];
		}
		__syncthreads(); // on attends
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
	}