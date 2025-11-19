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


static int run_cfsmlp_training(void)
{
    const char* features_csv = "/home/gic/Documents/Alvaro/migration-data-collection-alv/";
    const char* labels_csv = "/home/gic/Documents/Alvaro/migration-data-collection-alv/";
    const char* hidden = "32";

    char* kava_feat = kava_string(features_csv);
    char* kava_labels = kava_string(labels_csv);
    char* kava_hidden = kava_string(hidden);

    if (!kava_feat || !kava_labels || !kava_hidden)
        return -1;

    PRINT("[cfsmlp-training] Ejecutando pytorch_train()...\n");

    int ret = pytorch_train(kava_feat, kava_labels, kava_hidden);

    PRINT("[cfsmlp-training] pytorch_train() retornó: %d\n", ret);

    kava_free(kava_feat);
    kava_free(kava_labels);
    kava_free(kava_hidden);

    return ret;
}


static int __init cfsmlp_training_init(void)
{
    u64 start = ktime_get_ns();
    int ret = run_cfsmlp_training();
    u64 end = ktime_get_ns();

    PRINT("[cfsmlp-training] Tiempo total: %lld ns\n", end - start);
    return ret;
}

static void __exit cfsmlp_training_exit(void)
{
    PRINT("[cfsmlp-training] exit.\n");
}

module_init(cfsmlp_training_init);
module_exit(cfsmlp_training_exit);

MODULE_AUTHOR("Juan Diego Castro");
MODULE_DESCRIPTION("CFSMLP Training Kernel Module");
MODULE_LICENSE("GPL");
