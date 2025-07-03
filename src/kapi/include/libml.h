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


extern int dataset_from_csv(struct dataset *ds, char *filename, 
						    char *delim, int n_cols, enum type_t data_type, 
						    int headers);
