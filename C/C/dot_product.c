#include <stdio.h>

#define NET_INPUT_LAYER_SIZE 4
#define NET_OUTPUT_LAYER_SIZE 3

double dot_product(double *input, double *weight, double *bias, int input_size)
{
    int i = 0;
    double output = 0.0;
    for (i = 0; i < input_size; i++)
    {
        output += input[i] * weight[i];
    }
    output += *bias;
    return output;
}

void layer_output(double *input, double *weighs, double *bias, int output_size, double *outputs, int input_size)
{
    int i = 0;
    int offset = 0;
    for (i = 0; i < output_size; i++)
    {
        outputs[i] = dot_product(input, weighs + offset, &bias[i], input_size);
        offset += input_size;
    }
}

int main(void)
{
    double input[NET_INPUT_LAYER_SIZE] = {1.0, 2.0, 3.0, 2.5};
    double weights[NET_OUTPUT_LAYER_SIZE][NET_INPUT_LAYER_SIZE] = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87},
    };

    double bias[NET_OUTPUT_LAYER_SIZE] = {2.0, 3.0, 0.5};

    double outputs[NET_OUTPUT_LAYER_SIZE] = {0.0};

    // the real action
    layer_output(&input[0], &weights[0][0], &bias[0], NET_INPUT_LAYER_SIZE, &outputs[0], NET_OUTPUT_LAYER_SIZE);
    printf("nur output: %f %f %f\n", outputs[0], outputs[1], outputs[2]);
}