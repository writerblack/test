#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <helper_cuda.h>
//#include <helper_functions.h> 
#include <time.h>
#include <sys/time.h>

#define TILE_WIDTH 256

__global__ void reduce1(double *d_in,double *d_out){
    __shared__ double cache[TILE_WIDTH];

    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;

    cache[tid]=d_in[i];
    __syncthreads();

    for(int step=blockDim.x/2;step>0;step>>=1){
        if(tid<step){
            cache[tid]+=cache[tid+step];
        }
        __syncthreads();
    }

    if(tid==0)d_out[blockIdx.x]=cache[tid];
}

__device__ void warpReduce(volatile double* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce2(double *d_in,double *d_out){
    __shared__ double cache[TILE_WIDTH];

    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;

    cache[tid]=d_in[i];
    __syncthreads();

    for(int step=blockDim.x/2;step>32;step>>=1){
        if(tid<step){
            cache[tid]+=cache[tid+step];
        }
        __syncthreads();
    }

    if(tid<32)warpReduce(cache,tid);
    if(tid==0)d_out[blockIdx.x]=cache[tid];
}

bool check(double *out,double *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=512*1024*1024;
    double *a=(double *)malloc(N*sizeof(double));
    double *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(double));

    int block_num=N/TILE_WIDTH;
    double *out=(double *)malloc((N/TILE_WIDTH)*sizeof(double));
    double *d_out;
    cudaMalloc((void **)&d_out,(N/TILE_WIDTH)*sizeof(double));
    double *res=(double *)malloc((N/TILE_WIDTH)*sizeof(double));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    for(int i=0;i<block_num;i++){
        double cur=0;
        for(int j=0;j<TILE_WIDTH;j++){
            cur+=a[i*TILE_WIDTH+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(double),cudaMemcpyHostToDevice);

    dim3 Grid( N/TILE_WIDTH,1);
    dim3 Block( TILE_WIDTH,1);

    struct timeval begin1, end1;
    gettimeofday(&begin1, NULL);
    reduce1<<<Grid,Block>>>(d_a,d_out);
    cudaDeviceSynchronize();
    gettimeofday(&end1, NULL);
    double elapsedTime1 = (end1.tv_sec - begin1.tv_sec) * 1000000.0 + (end1.tv_usec - begin1.tv_usec);
    printf("%lf s\n", elapsedTime1 / 1000000);
    printf("the bandwith is %f GB per second\n",0.001*N*sizeof(double)/elapsedTime1);

    cudaMemcpy(out,d_out,block_num*sizeof(double),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
