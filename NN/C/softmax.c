#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RAND_HIGH_RANGE 0.10
#define RAND_MIN_RANGE -0.10
#define INIT_BIAS 0.0

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2 // Can be replaced with (sizeof(var)/sizeof(double))
#define NET_OUTPUT_LAYER_SIZE 5  // Can be replaced with (sizeof(var)/sizeof(double))

typedef void (*activation_callback)(double *output);

typedef struct
{
    double *weights;
    double *bias;
    double *output;
    int input_size;
    int output_size;
    activation_callback callback;
} layer_dense_t;

typedef struct
{
    double *x;
    double *y;
} spiral_data_t;

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

double activation_sigmoid(double x)
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

double activation_ReLU(double x)
{
    if (x < 0.0)
    {
        x = 0.0;
    }
    return x;
}

void activation1(double *output)
{
    *output = activation_ReLU(*output);
}

double uniform_distribution(double range_low, double range_high)
{
    double rng = rand() / (1.0 + RAND_MAX);
    double range = range_high - range_low + 1;
    double rng_scaled = (rng * range) + range_low;
    return rng_scaled;
}

void spiral_data(int points, int classes, spiral_data_t *data)
{

    data->x = (double *)malloc(sizeof(double) * points * classes * 2);
    if (data->x == NULL)
    {
        printf("data mem error\n");
        return;
    }
    data->y = (double *)malloc(sizeof(double) * points * classes);
    if (data->y == NULL)
    {
        printf("pionts mem error\n");
        return;
    }
    int ix = 0;
    int iy = 0;
    int class_number = 0;
    for (class_number = 0; class_number < classes; class_number++)
    {
        double r = 0;
        double t = class_number * 4;

        while (r <= 1 && t <= (class_number + 1) * 4)
        {
            // adding some randomness to t
            double random_t = t + uniform_distribution(-1.0, 1.0) * 0.2;

            // converting from polar to cartesian coordinates
            data->x[ix] = r * sin(random_t * 2.5);
            data->x[ix + 1] = r * cos(random_t * 2.5);

            data->y[iy] = class_number;

            // the below two statements achieve linspace-like functionality
            r += 1.0f / (points - 1);
            t += 4.0f / (points - 1);
            iy++;
            ix += 2; // increment index
        }
    }
}

void deloc_spiral(spiral_data_t *data)
{
    if (data->x != NULL)
    {
        free(data->x);
    }
    if (data->y != NULL)
    {
        free(data->y);
    }
}

void activation_softmax(layer_dense_t *output_layer)
{
    double sum = 0;
    double maxu = 0;
    int i = 0;

    maxu = output_layer->output[0];
    for (i = 0; i < output_layer->input_size; i++)
    {
        if (output_layer->output[i] > maxu)
        {
            maxu = output_layer->output[i];
        }
    }

    for (i = 0; i < output_layer->output_size; i++)
    {
        output_layer->output[i] = exp(output_layer->output[i] - maxu);
        sum += output_layer->output[i];
    }

    for (i = 0; i < output_layer->output_size; i++)
    {
        output_layer->output[i] = output_layer->output[i] / sum;
    }
}

double sum_softmax_layer_output(layer_dense_t *output_layer)
{
    double sum = 0.0;
    int i = 0;

    for (i = 0; i < output_layer->output_size; i++)
    {
        sum += output_layer->output[i];
    }

    return sum;
}

int main()
{
    srand(0);

    int i = 0;
    int j = 0;
    spiral_data_t X_data;
    layer_dense_t X;
    layer_dense_t layer1;
    // layer_dense_t layer_init;

    spiral_data(100, 3, &X_data);
    if (X_data.x == NULL)
    {
        printf("data null\n");
        return 0;
    }

    X.callback = NULL;
    layer1.callback = activation1;

    init_layer(&layer1, NET_INPUT_LAYER_1_SIZE, NET_OUTPUT_LAYER_SIZE);

    for (i = 0; i < NET_BATCH_SIZE; i++)
    {
        X.output = &X_data.x[i * 2];
        forward(&X, &layer1);

        printf("batch: %d layer1_output: ", i);
        for (j = 0; j < layer1.output_size; j++)
        {
            printf("%f ", layer1.output[j]);
        }
        printf("\n");
    }

    deloc_layer(&layer1);
    deloc_spiral(&X_data);
    return 0;
}
