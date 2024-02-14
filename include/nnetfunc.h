#ifndef nnetfunc
#define nnetfunc

#include "math.h"
#include "ndimarr.h"

float sigmoid(float value);
float leaky_relu(float value);
float relu(float value);
void leaky_relu_cust(f32_arr *A, float a);
void softmax(f32_arr *A);
void swish(f32_arr *A, float b);

#endif

