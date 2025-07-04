#include <linux/module.h>
#include <linux/random.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "libml.h"
#include "lake_shm.h"
#define PRINT(...) pr_warn(__VA_ARGS__)

// ==================== Start Load Dataset Module ====================
void init_matrix(int *dst, int n, int min, int max)
{
    int span = max - min + 1;
    for (int i = 0; i < n; ++i) {
        u32 r;
        get_random_bytes(&r, sizeof(r));
        dst[i] = min + (r % span);
    }
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

// Test
int run_dataset_load(void)
{
    enum type_t data_type = DOUBLE;
    struct dataset features, labels;
    int ret1, ret2;
    int n_input = 7;
    int n_output = 1;

    const char *features_path = "/home/gic/Documents/Flank/migration-prediction-notebooks/352-nab_historic_prep_features.csv";
    const char *labels_path = "/home/gic/Documents/Flank/migration-prediction-notebooks/352-nab_historic_prep_labels.csv";

    /* initialize feature and label datasets from .csv files */
    if (dataset_from_csv(&features, features_path, ",", n_input, data_type, 0) < 0 ||
        dataset_from_csv(&labels, labels_path, ",", n_output, data_type, 0)) {
        return -1;
    }

    // struct norm_metadata *meta = dataset_normalize (&features);

    // // printf ("MIN\t\tRANGE\n");
    // // for (int i = 0; i < n_input; ++i) {
    // //   printf ("%.4f\t\t%.4f\n", meta->min[i], meta->range[i]);
    // // }
    // /* Create layers for the model */
    // struct layer input, hidden1, hidden2, hidden3, output;

    // /* initialize each layer */
    // init_layer (&input, n_input, 8, NONE);
    // init_layer (&hidden1, 8, 8, SIGMOID);
    // /*init_layer (&hidden2, 8, 8, SIGMOID);
    // init_layer (&hidden3, 8, 8, SIGMOID);*/
    // init_layer (&output, 8, n_output, SIGMOID);

    // struct layer hidden_layers[1] = {hidden1, /*hidden2, hidden3*/};

    // /* initialize model */
    // struct model m;
    // init_model (&m, 1, n_input, n_output, BCE, hidden_layers, &output, &input, 0, NULL);

    // struct dataset x_train, x_test, y_train, y_test;
    // const float learning_rate = 0.1;
    // const int epochs = 1;

    // int dataset_len = features.size;
    // int train_len = 0.7 * dataset_len;

    // x_train = dataset_slice (&features, 0, train_len, -1, -1);
    // y_train = dataset_slice (&labels, 0, train_len, -1, -1);
    // x_test = dataset_slice (&features, train_len, dataset_len, -1, -1);
    // y_test = dataset_slice (&labels, train_len, dataset_len, -1, -1);

    // train (&m, &x_train, &y_train, learning_rate, epochs);

    // model_to_kspace (meta, &m);
    return 0;
}

// ==================== End Load Dataset ====================

static int __init dataset_load_init(void)
{
    u64 t_start, t_stop, t_time = 0;
    int ret = 0;

    t_start = ktime_get_ns();

    if ((ret = run_dataset_load())) {
        return ret;
    }
    t_stop = ktime_get_ns();
    t_time = t_stop - t_start;

    PRINT("Total time: %d", t_time);
    return ret;
}

static void __exit dataset_load_fini(void)
{
}

module_init(dataset_load_init);
module_exit(dataset_load_fini);

MODULE_AUTHOR("Juan Diego Castro & Alvaro Guerrero");
MODULE_DESCRIPTION("A module for loading dataset");
MODULE_LICENSE("GPL");
MODULE_VERSION(
    __stringify(1) "." __stringify(0) "." __stringify(0) "."
                                                         "0");
