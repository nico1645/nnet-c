#include "../include/nnetfunc.h"
#include <float.h>
#include <math.h>

float sigmoid(float value) { 
    if (value < 0) {
        return expf(value) / (1 + expf(value));
    } else {
        return 1.0 / (1 + expf(-value));
    }

};

float sigmoid_derivative(float value) {
  return sigmoid(value) * (1 - sigmoid(value));
};

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

float leaky_relu_derivative(float value) {
  if (value > 0)
    return 1;
  else
    return 0.02;
};

void softmax(f32_mat *A) {
  float biggest_num = FLT_MIN;
  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    if (A->matrix[i] > biggest_num)
      biggest_num = A->matrix[i];
  }
  float sum = 0;
  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    sum += expf(A->matrix[i]-biggest_num);
  }
  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    A->matrix[i] = expf(A->matrix[i]-biggest_num) / sum;
  }
};

void swish(f32_mat *A, float b) {
  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    A->matrix[i] = A->matrix[i] / (1 + expf(-b * A->matrix[i]));
  }
};

void leaky_relu_cust(f32_arr *A, float a) {
  for (unsigned int i = 0; i < A->length; i++) {
    if (A->arr[i] <= 0)
      A->arr[i] = A->arr[i] * a; // Standard values are between 0.01-0.03
  }
};
