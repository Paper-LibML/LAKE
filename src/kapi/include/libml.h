#pragma once

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <sys/types.h>
#endif


typedef long long fixed_t;


enum type_t {
	CHAR,
	UCHAR,
	INT,
	LONG,
	LL,
	ULL,
	FLOAT,
	DOUBLE,
	LDOUBLE
};


struct matrix {
	int rows;							/* first dimension of matrix */
	int cols;							/* second dimension of matrix */
#ifdef CONFIG_FIXED_POINT_ML
	fixed_t *data;						/* matrix data (fixed point) */
#else
	double *data;						/* matrix data (floating point) */
#endif
};


struct dataset { 
	int size;
	int capacity;
	int n_columns;
	int iterator;
	struct matrix data;
	char **columns;
	enum type_t data_type;
};

struct norm_metadata {
	double* min;
	double* range;
};

enum act_func {
	RELU,
	SIGMOID,
	TANH,
	NONE
};

/* A layer object. Receives a vector of size "input" and outputs
 * a vector of size "output" after activation. Readjusts its weights
 * by applying "gradient" */
struct layer {
	int n_input;					/* size of input vector */
	int n_output;					/* size of output vector*/
	struct matrix bias;				/* bias vector */
	struct matrix weights;			/* weight matrix */
	struct matrix input;			/* input vector */
	struct matrix output;			/* output vector */
	struct matrix sensitivities;	/* sensitivities vector */
	enum act_func act;				/* activation function */
};

extern int dataset_from_csv(struct dataset *ds, char *filename, 
						    char *delim, int n_cols, enum type_t data_type, 
						    int headers);
extern struct norm_metadata *dataset_normalize (struct dataset *ds);

extern int init_layer(struct layer *l, int n_input, int n_output, enum act_func act);
