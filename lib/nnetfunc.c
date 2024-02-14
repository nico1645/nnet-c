#include "../include/nnetfunc.h"
#include <math.h>

float sigmoid(float value) { return 1.0 / (1 + expf(-value)); };

float relu(float value) {
  if (value <= 0)
    return 0;
  else
    return value;
};

float leaky_relu(float value) {
  if (value <= 0)
    return value * 0.02; // Standard values are between 0.01-0.03
  else
    return value;
};

void softmax(f32_arr *A) {
  float sum = 0;
  for (unsigned int i = 0; i < A->length; i++) {
    sum += expf(A->arr[i]);
  }
  for (unsigned int i = 0; i < A->length; i++) {
    A->arr[i] = expf(A->arr[i]) / sum;
  }
};

void swish(f32_arr *A, float b) {
  for (unsigned int i = 0; i < A->length; i++) {
    A->arr[i] = A->arr[i] / (1 + expf(-b * A->arr[i]));
  }
};

void leaky_relu_cust(f32_arr *A, float a) {
  for (unsigned int i = 0; i < A->length; i++) {
    if (A->arr[i] <= 0)
      A->arr[i] = A->arr[i] * a; // Standard values are between 0.01-0.03
  }
};
