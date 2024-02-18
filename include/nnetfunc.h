#ifndef nnetfunc
#define nnetfunc

#include "math.h"
#include "ndimarr.h"

float sigmoid_derivative(float value);
float sigmoid(float value);
float leaky_relu_derivative(float value);
float leaky_relu(float value);
float relu(float value);
void leaky_relu_cust(f32_arr *A, float a);
void softmax_derivative(f32_mat *A, f32_mat *res);
void softmax(f32_mat *A);
void swish(f32_mat *A, float b);

#endif

