#ifdef __KERNEL__
#include <linux/delay.h>
#include <linux/ktime.h>
#else
#define kava_free(X) free(X)
#define kava_alloc(X) malloc(X)
#include <stdint.h>
#include <stdio.h>
#define u64 uint64_t
#include <unistd.h>
#define usleep_range(X,Y) sleep(X/1000)
#include <sys/time.h>
u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}

#define ktime_get_ns() get_tsns()
#endif

#include "weights.h"
#include "helpers.h"
//#include <asm/fpu/api.h>

#ifdef __KERNEL__
static char *cubin_path = "kml.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to kml.cubin, default ./kml.cubin");
#else
static char *cubin_path = "/disk/hfingler/HACK/kava/driver/kernel-ml/kml.cubin";
#endif

static int run_cpu(void) {
    return 0;
}

static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_cols;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;

CUdeviceptr d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_out0, d_out1, d_out2, d_input, d_result_cols, d_intital_stats;
static float* batch_input;
static int *result;

//refactor here
int readahead_online_data_cols = 5;
int readahead_online_data_rows = 1;
CUdeviceptr d_readahead_norm_online_data;
CUfunction get_average, get_variance, matrix_map, normalize_data, matrix_transpose,
    matrix_mult, add_bias, matrix_argmax;
//readahead_normalized_online_data
CUdeviceptr diff, local_average, local_std_dev, local_variance, readahead_norm_online_data_last_values;

CUdeviceptr wx0, wx1, wx2, bias;
CUdeviceptr wt0, wt1, wt2;

static void setup_gpu(int batch_size) {
    w0_rows = 15;
    w0_cols = 5;
    b0_rows = 15;
    b0_cols = 1;
    w1_rows = 5;
    w1_cols = 15;
    b1_rows = 5;
    b1_cols = 1; 
    w2_rows = 4;
    w2_cols = 5;
    b2_rows = 4;
    b2_cols = 1;

    float *w0, *w1, *w2, *b0, *b1, *b2;
    w0 = w0_arr;
    b0 = b0_arr;
    w1 = w1_arr;
    b1 = b1_arr;
    w2 = w2_arr;
    b2 = b2_arr;
    float *stats = &intial_stats[0];

    int input_features = 5;
    input_cols = input_features;
    check_error(cuMemAlloc((CUdeviceptr*) &d_w0, sizeof(float) *w0_rows * w0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w0, w0, sizeof(float) * w0_rows * w0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b0, sizeof(float) *b0_rows * b0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b0, b0, sizeof(float) * b0_rows * b0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w1, sizeof(float) *w1_rows * w1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w1, w1, sizeof(float) * w1_rows * w1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b1, sizeof(float) *b1_rows * b1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b1, b1, sizeof(float) * b1_rows * b1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w2, sizeof(float) *w2_rows * w2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w2, w2, sizeof(float) * w2_rows * w2_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b2, sizeof(float) *b2_rows * b2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b2, b2, sizeof(float) * b2_rows * b2_cols), "cuMemcpyHtoD", __LINE__);
    
    float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    //float batch_input[batch_size][5];
    int i ,j;
    for(i = 0; i < batch_size; i++) {
        for(j = 0 ; j < 5; j++) {
            batch_input[i*5 + j] = input[j];
        }
    }

    check_error(cuMemAlloc((CUdeviceptr*) &d_input, sizeof(float) *input_features * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_result_cols, sizeof(int) * batch_size), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_intital_stats, sizeof(float) *input_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_intital_stats, stats, sizeof(float) * input_cols), "cuMemcpyHtoD", __LINE__);
    

    //refactor below
    check_error(cuMemAlloc((CUdeviceptr*) &d_readahead_norm_online_data, 
        sizeof(float) *readahead_online_data_cols * batch_size), "cuMemAlloc ", __LINE__);

    out0_rows = batch_size;
    out0_cols = w0_rows;
    out1_rows = batch_size;
    out1_cols = w1_rows;
    out2_rows = batch_size;
    out2_cols = w2_rows;

    //readahead_normalized_online_data
    check_error(cuMemAlloc((CUdeviceptr*) &local_average, 
        sizeof(float)  * readahead_online_data_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &local_std_dev, 
        sizeof(float)  * readahead_online_data_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &local_variance, 
        sizeof(float) * readahead_online_data_cols), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &readahead_norm_online_data_last_values, 
        sizeof(float) * readahead_online_data_cols * batch_size), "cuMemAlloc ", __LINE__);

    //autodiff_forward
    check_error(cuMemAlloc((CUdeviceptr*) &d_out0, sizeof(float) *w0_rows * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_out1, sizeof(float) *w1_rows * out0_rows), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_out2, sizeof(float) *w2_rows * out1_rows), "cuMemAlloc ", __LINE__);

    //linear_layer_forward
    check_error(cuMemAlloc((CUdeviceptr*) &wt0, 
        sizeof(float) * w0_cols * w0_rows), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &wx0, 
        sizeof(float) * batch_size * w0_rows), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &wt1, 
        sizeof(float) * w1_cols * w1_rows), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &wx1, 
        sizeof(float) * batch_size * w1_rows), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &wt2, 
        sizeof(float) * w2_cols * w2_rows), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &wx2, 
        sizeof(float) * batch_size * w2_rows), "cuMemAlloc ", __LINE__);

}

static void copy_batch_inputs(int batch_size) {
    check_error(cuMemcpyHtoD(d_input, batch_input, sizeof(float) * input_cols * batch_size), "cuMemcpyHtoD", __LINE__);
}

static void get_result_batch(int batch_size) {
    check_error(cuMemcpyDtoH(result, d_result_cols, sizeof(int) * batch_size), "cuMemcpyDtoH", __LINE__);
}

void clean_batch(void) {
    cuMemFree(d_w0);
    cuMemFree(d_w1);
    cuMemFree(d_w2);
    cuMemFree(d_b0);
    cuMemFree(d_b1);
    cuMemFree(d_b2);
    cuMemFree(d_out0);
    cuMemFree(d_out1);
    cuMemFree(d_out2);
	cuMemFree(d_input);

    kava_free(batch_input);
    kava_free(result);

    //refactor below
    cuMemFree(local_average);
    cuMemFree(local_std_dev);
    cuMemFree(local_variance);
    cuMemFree(readahead_norm_online_data_last_values);

    cuMemFree(wt0);
    cuMemFree(wx0);
    cuMemFree(wt1);
    cuMemFree(wx1);
    cuMemFree(wt2);
    cuMemFree(wx2);
}

void linear_layer_forward(CUdeviceptr x, CUdeviceptr linear_w, int linear_w_rows, 
            int linear_w_columns, CUdeviceptr bias_vector, CUdeviceptr out, CUdeviceptr wx, CUdeviceptr wt, int batch_size) {

    void *args[] = {
		&linear_w, &wt, &linear_w_columns, &linear_w_rows
	};
    check_error(cuLaunchKernel(matrix_transpose, 
				linear_w_columns, 1, 1,          //blocks
				linear_w_rows, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    //wx = matrix_mult(x, wt);
    int block_size = 16;
    int grid_rows = (batch_size + block_size - 1) / block_size;
    int grid_cols = (linear_w_rows + block_size - 1) / block_size;

    void *args1[] = {
		&x, &wt, &wx, &batch_size, &linear_w_columns, &linear_w_rows
	};

    check_error(cuLaunchKernel(matrix_mult, 
				grid_cols, grid_rows, 1,          //blocks
				block_size, block_size, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
		"cuLaunchKernel", __LINE__);

    void *args3[] = {
		&wx, &bias_vector, &out
	};
    check_error(cuLaunchKernel(add_bias, 
				batch_size, 1, 1,          //blocks
				linear_w_rows, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
			"cuLaunchKernel", __LINE__);
}


void autodiff_forward(int batch_size) { 
    /*
    void linear_layer_forward(CUdeviceptr x, CUdeviceptr linear_w, int linear_w_rows, 
            int linear_w_columns, CUdeviceptr bias_vector, int layer_index, CUdeviceptr out,
            CUfunction* matrix_transpose, CUfunction* matrix_mult, 
            CUfunction* add_bias, int batch_size) {

 CUdeviceptr x, CUdeviceptr linear_w, int linear_w_rows, int linear_w_columns, CUdeviceptr bias_vector, int layer_index, CUdeviceptr out,

    linear_layer_forward(d_readahead_norm_online_data, d_w0, w0_rows, w0_cols, d_b0, 0, d_out0, batch_size);
    linear_layer_forward(d_out0, d_w1, w1_rows, w1_cols, d_b1, 1, d_out1, batch_size);
    linear_layer_forward(d_out1, d_w2, w2_rows, w2_cols, d_b2, 2, d_out2, batch_size);
    */

    linear_layer_forward(d_readahead_norm_online_data, 
                                 d_w0, w0_rows, w0_cols, d_b0, d_out0, wx0, wt0, batch_size);
    linear_layer_forward(d_out0, d_w1, w1_rows, w1_cols, d_b1, d_out1, wx1, wt1, batch_size);
    linear_layer_forward(d_out1, d_w2, w2_rows, w2_cols, d_b2, d_out2, wx2, wt2, batch_size);

    void *args3[] = {
		&d_out2, &w2_rows, &d_result_cols
	};
    check_error(cuLaunchKernel(matrix_argmax, 
				1, 1, 1,          //blocks
				out2_rows, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
			"cuLaunchKernel", __LINE__);

    //check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
    
    //int *result_cols = (int*) kava_alloc(sizeof(int)* batch_size);
    //check_error(cuMemcpyDtoH(result_cols, d_result_cols, sizeof(int) * batch_size), "cuMemcpyDtoH", __LINE__);
    //return result_cols;
}

void readahead_normalized_online_data(int batch_size) {
    int n_seconds = 10;
    int n_1_seconds = 9;

    void *args[] = {
		&d_intital_stats, &n_seconds, &n_1_seconds, &batch_size, &d_input, &local_average
	};
    check_error(cuLaunchKernel(get_average, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);


    void *args1[] = {
		&d_intital_stats, &n_seconds, &n_1_seconds, &batch_size, &d_input, 
        &readahead_norm_online_data_last_values, &local_variance
	};
    check_error(cuLaunchKernel(get_variance, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);
    //check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
    
    void *args2[] = {
		&local_variance, &local_std_dev
	};
    check_error(cuLaunchKernel(matrix_map, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
		"cuLaunchKernel", __LINE__);
    //check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);

    void *args3[] = {
		&d_input, &local_average, &local_std_dev, &d_readahead_norm_online_data
	};
    check_error(cuLaunchKernel(normalize_data, 
				batch_size, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
			"cuLaunchKernel", __LINE__);
    //check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
}

void predict_readahead_class(int batch_size) {
    readahead_normalized_online_data(batch_size);
    autodiff_forward(batch_size);
}

static int run_gpu(void) {
    int i, j;
    int RUNS;
    const int n = 1024;
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int batch_size;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
  
    CUcontext cuContext;
    gpu_init(0, &cuContext);

    int n_batches = 11;

    gpu_get_cufunc(cubin_path, "_Z11get_averagePfiiiS_S_", &get_average);
    gpu_get_cufunc(cubin_path, "_Z12get_variancePfffiS_S_S_", &get_variance);
    gpu_get_cufunc(cubin_path, "_Z10matrix_mapPfS_", &matrix_map);
    gpu_get_cufunc(cubin_path, "_Z14normalize_dataPfS_S_S_", &normalize_data);
    gpu_get_cufunc(cubin_path, "_Z16matrix_transposePfS_ii", &matrix_transpose);
    gpu_get_cufunc(cubin_path, "_Z11matrix_multPfS_S_iii", &matrix_mult);
    gpu_get_cufunc(cubin_path, "_Z8add_biasPfS_S_", &add_bias);
    gpu_get_cufunc(cubin_path, "_Z13matrix_argmaxPfiPi", &matrix_argmax);
    RUNS = 10;

    
#ifdef __KERNEL__
    comp_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
#else
    comp_run_times = (u64*) malloc(RUNS*sizeof(u64));
    total_run_times = (u64*) malloc(RUNS*sizeof(u64));
#endif


    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

        batch_input = (float*) kava_alloc(batch_size * 5 * sizeof(float));
        result = (int*) kava_alloc(batch_size * sizeof(int));
        setup_gpu(batch_size);
        copy_batch_inputs(batch_size);
        predict_readahead_class(batch_size);

        usleep_range(1000, 2000);
        cuCtxSynchronize();
    
        for (j = 0 ; j < RUNS ; j++) {
            //PRINT(V_INFO, "Runing for batch size %d\n", batch_size);
            t_start = ktime_get_ns();
            copy_batch_inputs(batch_size);
            c_start = ktime_get_ns();
            predict_readahead_class(batch_size);
            cuCtxSynchronize();
            c_stop = ktime_get_ns();
            get_result_batch(batch_size);
            t_stop = ktime_get_ns();
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
	    }

	    avg = 0; avg_total = 0;
        best = 0; best_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
            if (best == 0 || comp_run_times[j] < best) best = comp_run_times[j];
            if (best_total == 0 || total_run_times[j] < best_total) best_total = total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
        best = best / 1000; best_total = best_total / 1000;

#ifdef __KERNEL__
        PRINT(V_INFO, "GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
#else
        printf("GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
#endif

        clean_batch();
	}
    return 0;
}


#ifdef __KERNEL__

/**
 * Program main
 */
static int __init kml_init(void)
{
	return run_gpu();
}

static void __exit kml_fini(void)
{

}

module_init(kml_init);
module_exit(kml_fini);

MODULE_AUTHOR("Isha Tarte");
MODULE_DESCRIPTION("Kernel module of a KML program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
#else

int main() {
    run_gpu();
    return 0;
}

#endif
