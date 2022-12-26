#include <stdio.h>
#include <stdlib.h>

#define RAND_HIGH_RANGE0 .10
#define RAND_MIN_RANGE -0.10
#define INIT_BIAS 0.0
#define NET_BATCH_SIZE 3
#define NET_INOUT_LAYER_1_SIZE 4
#define NET_INOUT_LAYER_2_SIZE 5
#define NET_OUTPUT_LAYER_SIZE 3

typedef struct layer_dense_t
{
    double *weights;
    double *inputs;
    double *bias;
    int input_size;
    int output_size;
};

double dot_product(double *inputs, double *weights, double *bias, int input_size)
{
    int i = 0;
    
}
