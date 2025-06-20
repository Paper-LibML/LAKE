/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * Copyright (C) 2022-2024 Ariel Szekely
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#ifdef __KERNEL__
#include <linux/module.h>
#include <linux/random.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "cuda.h"
#include "lake_shm.h"
#define PRINT(...) pr_warn(__VA_ARGS__)
#else
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <errno.h>

static inline uint64_t get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, 0);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}

#define usleep_range(X,Y) sleep(X/1000000)
#define ktime_get_ns() get_tsns()
#define u64 uint64_t
#define vmalloc(X) malloc(X)
#define vfree(X) free((void *)X)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#define PRINT(...) printf(__VA_ARGS__)
#include <cuda.h>
#endif

static char *cubin_path = "/home/gic/Desktop/LAKE/src/matmul/knncuda.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to .cubin");
#endif

#define BLOCK_DIM 16
#define WARMS 2
#define RUNS 5

// XXX Need to handle FLOATs eventually

struct cuda_ctx
{
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    CUfunction matrix_multiply;
};

typedef struct
{
    u64 x;
    u64 y;
    u64 z;
} dim3;

// CUDA vars
struct cuda_ctx ctx;

int init_cuda(void)
{
    int ret = 0;

    ret = cuInit(0);
    if (ret) {
        PRINT("Err cuInit %d\n", ret);
        goto out;
    }

    ret = cuDeviceGet(&ctx.dev, 0);
    if (ret) {
        PRINT("Err cuDeviceGet %d\n", ret);
        goto out;
    }

    ret = cuCtxCreate(&ctx.ctx, 0, ctx.dev);
    if (ret) {
        PRINT("Err cuCtxCreate %d\n", ret);
        goto out;
    }

    PRINT("Loading cubin from: %s\n", cubin_path);
    ret = cuModuleLoad(&ctx.mod, cubin_path);
    if (ret) {
        PRINT("Err cuModuleLoad %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.matrix_multiply, ctx.mod,
                              "_Z15matrix_multiplyPiS_S_i");
    if (ret) {
        PRINT("Err cuModuleGetFunction %d\n", ret);
        goto out;
    }

out: 
    return ret;
}
// ==================== End CUDA ====================

// ==================== Start Matmul ====================
void init_matrix(int *dst, int n, int min, int max)
{
    int span = max - min + 1;
    PRINT("min: %d\n", min);
    PRINT("max: %d\n", max);

#ifdef __KERNEL__
    /* ---------- kernel space : strong CSPRNG already available ---------- */
    for (int i = 0; i < n; ++i) {
        u32 r;
        get_random_bytes(&r, sizeof(r));   /* 32 random bits */
        dst[i] = min + (r % span);
    }

#else
    /* ------------------------ user space : libc rand() ------------------ */

    /* RAND_MAX is guaranteed ≥ 32 767; suitable for spans ≤ 32 767.
       For larger spans we stitch 2×16-bit chunks into 32 bits.          */
    for (int i = 0; i < n; ++i) {
        unsigned int r;

#if RAND_MAX >= 0x7FFFFFFF          /* 31 good bits in one call */
        r = (unsigned int)rand();
#else                               /* typical 0-x7FFF : combine two calls */
        r = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
#endif
        dst[i] = min + (r % span);
    }
#endif
}

static u64 ctime, ttime;

int matmul_cuda(const int *m1, const int *m2, int dimension,
                const int *mat_result, int measure_compute)
{
    int ret = 0;

    // Launch params
    dim3 block0;
    dim3 grid0;

    // Vars for computation
    CUdeviceptr m1_dev, m2_dev, mat_result_dev;

    int size = dimension * dimension;

    u64 t_start, t_stop, c_start, c_stop;
    ctime = 0; ttime = 0; 

    t_start = ktime_get_ns();

    // Allocate global memory (using cuMemAlloc instead of cuMemAllocPitch)
    ret = cuMemAlloc(&m1_dev, size * sizeof(int));  // 1D memory allocation for m1
    PRINT("cuMemAlloc m1_dev: %d\n", ret);
    if (ret) {
        PRINT("Memory allocation error for m1\n");
        goto out;
    }

    ret = cuMemAlloc(&m2_dev, size * sizeof(int));  // 1D memory allocation for m2
    PRINT("cuMemAlloc m2_dev: %d\n", ret);
    if (ret) {
        PRINT("Memory allocation error for m2\n");
        goto out;
    }

    ret = cuMemAlloc(&mat_result_dev, size * sizeof(int));  // 1D memory allocation for mat_result
    PRINT("cuMemAlloc mat_result_dev: %d\n", ret);
    if (ret) {
        PRINT("Memory allocation error for mat_result\n");
        goto out;
    }

    // Copy reference and query data from the host to the device (synchronous)
    ret = cuMemcpyHtoD(m1_dev, m1, size * sizeof(int));  // Copy m1 data to device
    PRINT("cuMemcpyHtoD m1_dev: %d\n", ret);
    if (ret) {
        PRINT("Unable to copy data from host to device for m1\n");
        goto out;
    }

    ret = cuMemcpyHtoD(m2_dev, m2, size * sizeof(int));  // Copy m2 data to device
    PRINT("cuMemcpyHtoD m2_dev: %d\n", ret);
    if (ret) {
        PRINT("Unable to copy data from host to device for m2\n");
        goto out;
    }

    // If we're measuring just compute, wait until everything is done
    if (measure_compute) {
        cuCtxSynchronize();
        c_start = ktime_get_ns();
    }

    // Compute the squared Euclidean distances (kernel launch)
    block0 = (dim3) { BLOCK_DIM, BLOCK_DIM, 1 };
    grid0 = (dim3) { size / BLOCK_DIM, size / BLOCK_DIM, 1 };
    if (size % BLOCK_DIM != 0) {
        grid0.x += 1;
    }
    if (size % BLOCK_DIM != 0) {
        grid0.y += 1;
    }

    int n = dimension;
    void *args0[] = { &m1_dev, &m2_dev, &mat_result_dev, &n };
    ret = cuLaunchKernel( ctx.matrix_multiply, grid0.x, grid0.y,
                          grid0.z, block0.x, block0.y,
                          block0.z, 0, 0,
                          args0, NULL); // Launch the kernel synchronously
    PRINT("cuLaunchKernel: %d\n", ret);
    if (ret) {
        PRINT("Kernel launch failed\n");
        goto out;
    }

    // Wait for kernel to finish
    if (measure_compute) {
        cuCtxSynchronize();
        c_stop = ktime_get_ns();
        ctime = c_stop - c_start;
    }

    // Copy mat_result from device to host with correct size
    PRINT("mat_result_dev: %p, mat_result: %p, size: %d\n", mat_result_dev, mat_result, size * sizeof(int));
    ret = cuMemcpyDtoH(mat_result, mat_result_dev, size * sizeof(int));  // Correct size copy
    PRINT("cuMemcpyDtoH mat_result: %d\n", ret);
    if (ret) {
        PRINT("Unable to copy mat_result from device to host\n");
        goto out;
    }

    ret = cuCtxSynchronize();
    PRINT("cuCtxSynchronize: %d\n", ret);
    if (ret) {
        PRINT("Unable to synchronize context\n");
        goto out;
    }

    PRINT("\n");
    t_stop = ktime_get_ns();
    ttime = t_stop - t_start;

out:
    cuMemFree( m1_dev );
    cuMemFree( m2_dev );
    cuMemFree( mat_result_dev );
    return ret;
}


void print_matrix(int *m, int dim)
{
    for (int i = 0; i < (dim < 5? dim : 5); ++i)
    {
        for (int j = 0; j < (dim < 5? dim : 5); ++j)
        {
            PRINT("%d ", m[i*dim + j]);
        }
        PRINT("\n");
    }
    PRINT("\n");
}


// XXX Should time at some point
int test(const int *m1, const int *m2, int dimension)
{
    int ret = 0;
    int i, measure_comp;
    int *mat_result;
    u64 ctimes;
    u64 ttimes;

    // Allocate memory for computed k-NN neighbors
    mat_result = (int *) kava_alloc(dimension * dimension * sizeof(int));

    // Allocation check
    if (!mat_result) {
        PRINT("Error allocating CPU memory for KNN results\n");
        ret = -ENOMEM;
        goto out;
    }

    usleep_range(200, 500);
    ctimes = 0;
    ttimes = 0;
    
    for (measure_comp = 0; measure_comp < 2; ++measure_comp) {
        for (i = 0; i < WARMS + RUNS; ++i) {
            if ((ret = matmul_cuda(m1, m2, dimension, mat_result, measure_comp))) {
                PRINT("Computation failed on round %d\n", i);
                goto out;
            }

            PRINT("Printing first 5 rows and cols:\n");
            print_matrix(mat_result, dimension);            
            if (i >= WARMS) {
                if (measure_comp == 1)
                    ctimes += ctime;
                else
                    ttimes += ttime;
            }
            usleep_range(2000, 5000);
        }
    }
    PRINT("matmul_GPU_BATCH_, %lld, %lld\n", ctimes / (RUNS * 1000), ttimes / (RUNS * 1000));

out:
    kava_free(mat_result);

    return ret;
}

// Allocate
int run_matmul(void)
{
    int ret = 0;

    int *m1;
    int *m2;

    int min = 0, max = 1;
    int dimension = 5;
    int size = dimension * dimension;

    m1 = (int *) kava_alloc(size * sizeof(int));
    m2 = (int *) kava_alloc(size * sizeof(int));

    // Allocation checks
    if (!m1 || !m2) {
        PRINT("Error allocating matmul CPU resources\n");
        ret = -ENOMEM;
        goto out;
    }
    // Initialize reference and query points with random values
    init_matrix(m1, size, min, max);
    init_matrix(m2, size, min, max);

    PRINT("Matrix m1:\n");
    print_matrix(m1, dimension);
    PRINT("\n");
    PRINT("Matrix m2:\n");
    print_matrix(m2, dimension);

    ret = test(m1, m2, dimension);
    if (ret) {
        PRINT("Matmul execution test failed\n");
        ret = -ENOENT;
        goto out;
    }
out:
    kava_free(m1);
    kava_free(m2);

    return 0;
}
// ==================== End Matmul ====================

#ifdef __KERNEL__
static int __init ghost_buster_init(void)
{
    int ret = 0;
    if ((ret = init_cuda())) {
        PRINT("error init cuda");
        return ret;
    }
    if ((ret = run_matmul())) {
        return ret;
    }
    return ret;
}

static void __exit ghost_buster_fini(void)
{
}

module_init(ghost_buster_init);
module_exit(ghost_buster_fini);

MODULE_AUTHOR("Juan Diego Castro & Alvaro Guerrero");
MODULE_DESCRIPTION("A matrix multiplication module");
MODULE_LICENSE("GPL");
MODULE_VERSION(
    __stringify(1) "." __stringify(0) "." __stringify(0) "."
                                                         "0");

#else

int main() {
    int ret = 0;
    if ((ret = init_cuda())) {
        return ret;
    }
    if ((ret = run_matmul())) {
        return ret;
    }
    return ret;
}

#endif
