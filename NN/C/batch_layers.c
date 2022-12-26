#include <stdio.h>
#include <stdlib.h>

#define RAND_HIGH_RANGE 0.10
#define RAND_MIN_RANGE -0.10
#define INIT_BIAS 0.0
#define NET_BATCH_SIZE 3
#define NET_INPUT_LAYER_1_SIZE 4  // can be replaced with (sizeof(var)/sizeof(double))
#define NET_HIDDEN_LAYER_2_SIZE 5 // can be replaced with (sizeof(var)/sizeof(double))
#define NET_OUTPUT_LAYER_SIZE 2   // can be replaced with (sizeof(var)/sizeof(double))

typedef struct
{
    double *weights;
    double *output;
    double *bias;
    int input_size;
    int output_size;
} layer_dense_t;

double dot_product(double *inputs, double *weights, double *bias, int input_size)
{
    int i = 0;
    double output = 0.0;
    for (i = 0; i < input_size; i++)
    {
        output += inputs[i] * weights[i];
    }
    output += *bias;
    return output;
}

void layer_outputs(double *input, double *weights, double *bias, int input_size, double *outputs, int output_size)
{
    int i = 0;
    int offset = 0.0;
    for (i = 0; i < output_size; i++)
    {
        outputs[i] = dot_product(input, weights + offset, &bias[i], input_size);
        offset += input_size;
    }
}

double rand_rage(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void init_layer(layer_dense_t *layer, int input_size, int output_size)
{

    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = (double *)malloc(sizeof(double) * output_size);
    if (layer->weights == NULL)
    {
        printf("weight mem err \n");
        return;
    }

    layer->bias = (double *)malloc(sizeof(double) * output_size);
    if (layer->bias == NULL)
    {
        printf("bias mem err \n");
        return;
    }

    layer->output = (double *)malloc(sizeof(double) * output_size);
    if (layer->output == NULL)
    {
        printf("output mem error\n");
        return;
    }

    int i = 0;
    for (i = 0; i < output_size; i++)
    {
        layer->bias[i] = INIT_BIAS;
    }

    for (i = 0; i < (input_size * output_size); i++)
    {
        layer->weights[i] = rand_rage(RAND_MIN_RANGE, RAND_HIGH_RANGE);
    }
}

// de-allocate layers memory
void deloc_layer(layer_dense_t *layer)
{
    if (layer->weights != NULL)
    {
        free(layer->weights);
    }

    if (layer->bias != NULL)
    {
        free(layer->bias);
    }
    if (layer->output != NULL)
    {
        free(layer->output);
    }
}

void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer)
{
    layer_outputs((previous_layer->output), next_layer->weights,
                  next_layer->bias, next_layer->input_size, next_layer->output, next_layer->output_size);
}

int main()
{
    srand(0);

    int i = 0;
    int j = 0;

    layer_dense_t X;
    layer_dense_t layer_1;
    layer_dense_t layer_2;

    double X_input[NET_BATCH_SIZE][NET_INPUT_LAYER_1_SIZE] = {
        {1.0, 2.0, 3.0, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8},
    };

    init_layer(&layer_1, NET_INPUT_LAYER_1_SIZE, NET_HIDDEN_LAYER_2_SIZE);
    init_layer(&layer_2, NET_HIDDEN_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

    for (i = 0; i < NET_BATCH_SIZE; i++)
    {
        X.output = &X_input[i][0];

        forward(&X, &layer_1);
        forward(&layer_1, &layer_2);

        printf("batch: %d layerY_output: ", i);

        for (j = 0; j < layer_2.output_size; j++)
        {
            printf("%f ", layer_2.output[j]);
        }
        printf("\n");
    }

    deloc_layer(&layer_1);
    deloc_layer(&layer_2);

    return 0;
}