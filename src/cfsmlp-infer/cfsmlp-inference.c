#include <linux/module.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include "libml.h"
#include "lake_shm.h"

#define PRINT(...) pr_warn(__VA_ARGS__)

static char* kava_string(const char* s)
{
    char* ret = kava_alloc(strlen(s) + 1);
    if (!ret) {
        PRINT("kava_alloc failed\n");
        return NULL;
    }
    strcpy(ret, s);
    return ret;
}


static int run_cfsmlp_inference(void)
{
    const char* features_csv = "/home/gic/Documents/Alvaro/migration-data-collection-alv/smithwa_256-features.csv";
    const char* mlpi_model_path = "/home/gic/Documents/Alvaro/Training-CFS-Python/model-256t.mlpi";

    char* kava_feat = kava_string(features_csv);
    if (!kava_feat) {
      return -1;
    }
  
    PRINT("[cfsmlp-inference] Ejecutando quantized inference...\n");

    char* kava_mlpi_model_path = kava_string(mlpi_model_path);
    if (!kava_mlpi_model_path) {
      return -1;
    }

    PRINT("Loading model at %s\n", kava_mlpi_model_path);
    MlpiModel* m = mlpi_load(kava_mlpi_model_path);
    PRINT("Model loaded");

    int model_in = mlpi_model_in_dim(m);

    PRINT("Model in dimensions: %i\n", model_in);

    int8_t* x_q = kava_alloc(sizeof(int8_t) * model_in);
    int r = read_row_floats_and_quantize(m, kava_feat, 93, x_q);

    PRINT("Quantized:\n");
    for (int i = 0; i < model_in; i++) {
      PRINT("%i\n", x_q[i]);
    }
    PRINT("\n");

    int* predicted_class_out = kava_alloc(sizeof(int));

    int ret = infer_on_quantized(m, x_q, predicted_class_out);

    PRINT("[cfsmlp-inference] infer_on_quantized() retornó: %i\n", *predicted_class_out);

    kava_free(kava_feat);

    return ret;
}


static int __init cfsmlp_inference_init(void)
{
    u64 start = ktime_get_ns();
    int ret = run_cfsmlp_inference();
    u64 end = ktime_get_ns();

    PRINT("[cfsmlp-inference] Tiempo total: %lld ns\n", end - start);
    return ret;
}

static void __exit cfsmlp_inference_exit(void)
{
    PRINT("[cfsmlp-inference] exit.\n");
}

module_init(cfsmlp_inference_init);
module_exit(cfsmlp_inference_exit);

MODULE_AUTHOR("Juan Diego Castro");
MODULE_DESCRIPTION("CFSMLP Inference Kernel Module");
MODULE_LICENSE("GPL");
