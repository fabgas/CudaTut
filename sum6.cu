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
	
		if (idx+7*blockDim.x<n) {
			sdata[tid]+=g_idata[idx+blockDim.x];
			sdata[tid]+=g_idata[idx+2*blockDim.x];
			sdata[tid]+=g_idata[idx+3*blockDim.x];
			sdata[tid]+=g_idata[idx+4*blockDim.x];
			sdata[tid]+=g_idata[idx+5*blockDim.x];

			sdata[tid]+=g_idata[idx+6*blockDim.x];
			sdata[tid]+=g_idata[idx+7*blockDim.x];
		}
	
	__syncthreads(); // on attends tous les blocks
	
	if (idx >= n) return; // on coupe au dela du cutoff
	// do reduction in shared mem for one block 
	if (blockDim.x>=1024 && tid < 512) sdata[tid] += sdata[tid + 512];
	__syncthreads();
	
	if (blockDim.x>=512 && tid < 256) sdata[tid] += sdata[tid + 256];
	__syncthreads();
	
	if (blockDim.x>=256 && tid < 128) sdata[tid] += sdata[tid + 128];
	__syncthreads();
	
	if (blockDim.x>=128 && tid < 64) sdata[tid] += sdata[tid + 64];
	__syncthreads();
	
	if (tid < 32) {
		volatile float *vsmem = sdata;
		vsmem[tid] +=vsmem[tid+32];
		vsmem[tid] +=vsmem[tid+16];
		vsmem[tid] +=vsmem[tid+8];
		vsmem[tid] +=vsmem[tid+4];
		vsmem[tid] +=vsmem[tid+2];
		vsmem[tid] +=vsmem[tid+1];
			
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
	}