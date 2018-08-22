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
	
	unsigned int idx = blockIdx.x*blockDim.x*8 + threadIdx.x; // index général
	sdata[tid] = g_idata[idx]; // copy vers la shared memory du block
	
	if (idx+blockDim.x<n) {
		sdata[tid]+=g_idata[idx+blockDim.x];
	}
	if (idx+2*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+2*blockDim.x];
	}
	
	if (idx+3*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+3*blockDim.x];
	}
	
	if (idx+4*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+4*blockDim.x];
	}
	
	if (idx+5*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+5*blockDim.x];
	}
	
	if (idx+6*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+6*blockDim.x];
	}
	
	if (idx+7*blockDim.x<n) {
		sdata[tid]+=g_idata[idx+7*blockDim.x];
	}
	
	__syncthreads(); // on attends tous les blocks
	
	if (idx >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	for(int s=blockDim.x/2; s >0; s >>= 1) {
		if (tid<s) { // si correspond à un multiple de la dimension
		sdata[tid] += sdata[tid + s];
		}
		__syncthreads(); // on attends
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
	}