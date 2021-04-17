#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#define BLOCK_SIZE 16

// CPU Implementation
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
          int tmp = 0.0;
          for (int h = 0; h < n; ++h) {
              tmp += h_a[i * n + h] * h_b[h * k + j];
          }
          h_result[i * k + j] = tmp;
      }
  }
}

// GPU Implementation

__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];
 
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

for (int sub = 0; sub < gridDim.x; ++sub) {
    idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
    if(idx >= n*n) {
        // n may not divisible by BLOCK_SIZE
        tile_a[threadIdx.y][threadIdx.x] = 0;
    } else {
        tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
    }

    idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
    if(idx >= n*n) {
        tile_b[threadIdx.y][threadIdx.x] = 0;
    } else {
        tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
		 tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }
  if(row < n && col < n) {
    d_result[row * n + col] = tmp;
  }
}
// Main Function

int main(int argc, char const *argv[])
{
    int m, n, k; // m x n matrix multiplied with n x k matrix
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
 
    srand(time(NULL));
 
    printf("Enter square matrix size (nxn): ");
    scanf("%d", &n);
 
    // We consider only square matrices.
    m = k = n;
 
    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a,  sizeof(int) * m * n);
    cudaMallocHost((void **) &h_b,  sizeof(int) * n * k);
    cudaMallocHost((void **) &h_c,  sizeof(int) * m * k);
    cudaMallocHost((void **) &h_cc, sizeof(int) * m * k);
 
printf("Filling matrices A and B with random values...\n");
 
// random initialize matrix A
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        h_a[i * n + j] = rand() % 1024;
    }
}
// random initialize matrix B
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
        h_b[i * k + j] = rand() % 1024;
    }
}
// some events to count the execution time
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
 
printf("Running GPU Algorithm...\n");
// start to count execution time of GPU version
cudaEventRecord(start, 0);
 
// Allocate memory space on the device
int *d_a, *d_b, *d_c;
cudaMalloc((void **) &d_a, sizeof(int)*m*n);
cudaMalloc((void **) &d_b, sizeof(int)*n*k);
cudaMalloc((void **) &d_c, sizeof(int)*m*k);
 
// copy matrix A and B from host to device memory
cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);
 
unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
dim3 dimGrid(grid_cols, grid_rows);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
 
// Launch kernel
gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
 
// Transfer results from device to host
cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// time counting terminate
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

// compute time elapse on GPU computing
cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
printf("GPU Time: %f ms.\n\n", gpu_elapsed_time_ms);

printf("Running CPU Algorithm...\n");

cudaEventRecord(start, 0);

cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
printf("CPU Time: %f ms.\n\n", cpu_elapsed_time_ms);

// validate results computed by GPU

int all_ok = 1;

for (int i = 0; i < m; ++i) {
	for (int j = 0; j < k; ++j) {
		if(h_cc[i*k + j] != h_c[i*k + j]) {
			all_ok = 0;
		}
	}
}

// roughly compute speedup
if(all_ok) {
	printf("Results cross-verified as correct, speedup = %f\n", 
			cpu_elapsed_time_ms / gpu_elapsed_time_ms);
} else {
	printf("Incorrect results. \n");
}

// free memeory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
cudaFreeHost(h_a);
cudaFreeHost(h_b);
cudaFreeHost(h_c);
cudaFreeHost(h_cc);

printf("Exiting.\n");
return 0;
}
