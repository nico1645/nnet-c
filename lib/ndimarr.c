#include "../include/ndimarr.h"

float mat_at(f32_mat *A, unsigned int i, unsigned int j) {
  if (A->transposed == 1) {
    return A->matrix[j * A->cols + i];
  } else {
    return A->matrix[i * A->cols + j];
  }
};

void mat_set(f32_mat *A, float value, unsigned int i, unsigned int j) {
  if (A->transposed == 1) {
    A->matrix[j * A->cols + i] = value;
  } else {
    A->matrix[i * A->cols + j] = value;
  }
};

f32_mat *create_mnmat(float *A, int rows, int columns) {
  f32_mat *matrix = malloc(sizeof(f32_mat));
  if (matrix == NULL)
    return NULL;
  matrix->cols = columns;
  matrix->rows = rows;
  matrix->transposed = 0;
  matrix->matrix = A;
  return matrix;
};

f32_arr *create_narr(float *A, int length) {
  f32_arr *arr = malloc(sizeof(f32_mat));
  if (arr == NULL)
    return NULL;
  arr->length = length;
  arr->arr = A;
  return arr;
};

int hamard_prod(f32_mat *A, f32_mat *B) {
  if (A->rows != B->rows || A->cols != B->cols)
    return 1;
  for (unsigned int i = 0; i < A->rows * A->rows; i++) {
    A->matrix[i] = A->matrix[i] * A->matrix[i];
  }

  return 0;
}

f32_mat *mat_mul(f32_mat *A, f32_mat *B) {
  f32_mat *result = malloc(sizeof(f32_mat));
  float *matrix = malloc(sizeof(float) * A->rows * B->cols);
  if (result == NULL || matrix == NULL)
    return NULL;
  result->cols = B->cols;
  result->rows = A->rows;
  result->transposed = 0;
  if (A->transposed == 1 && B->transposed == 1) {
    for (unsigned int i = 0; i < A->rows; i++) {
      for (unsigned int j = 0; j < B->cols; j++) {
        float dot_prod = 0;
        for (unsigned int k = 0; k < A->cols; k++) {
          dot_prod += A->matrix[k * A->cols + i] * B->matrix[j * B->cols + k];
        }
        matrix[i * result->cols + j] = dot_prod;
      }
    }
  } else if (A->transposed == 1 && B->transposed == 0) {
    for (unsigned int i = 0; i < A->rows; i++) {
      for (unsigned int j = 0; j < B->cols; j++) {
        float dot_prod = 0;
        for (unsigned int k = 0; k < A->cols; k++) {
          dot_prod += A->matrix[k * A->cols + i] * B->matrix[k * B->cols + j];
        }
        matrix[i * result->cols + j] = dot_prod;
      }
    }
  } else if (B->transposed == 1 && A->transposed == 0) {
    for (unsigned int i = 0; i < A->rows; i++) {
      for (unsigned int j = 0; j < B->cols; j++) {
        float dot_prod = 0;
        for (unsigned int k = 0; k < A->cols; k++) {
          dot_prod += A->matrix[i * A->cols + k] * B->matrix[j * B->cols + k];
        }
        matrix[i * result->cols + j] = dot_prod;
      }
    }
  } else {
    for (unsigned int i = 0; i < A->rows; i++) {
      for (unsigned int j = 0; j < B->cols; j++) {
        float dot_prod = 0;
        for (unsigned int k = 0; k < A->cols; k++) {
          dot_prod += A->matrix[i * A->cols + k] * B->matrix[k * B->cols + j];
        }
        matrix[i * result->cols + j] = dot_prod;
      }
    }
  }
  result->matrix = matrix;
  return result;
};

int transpose(f32_mat *A) {
  A->transposed = (A->transposed + 1) % 2;
  int tmp = A->rows;
  A->rows = A->cols;
  A->cols = tmp;
  return 0;
};

int dot_prod(f32_arr *A, f32_arr *B, float *prod) {
  if (A->length != B->length)
    return 1;

  *prod = 0;
  for (unsigned int i = 0; i < A->length; i++) {
    // Maybe bound checks
    *prod += A->arr[i] * B->arr[i];
  }
  return 1;
};

int minus(f32_arr *A, f32_arr *B) {
  if (A->length != B->length)
    return 1;

  for (unsigned int i = 0; i < A->length; i++) {
    // Maybe add bound checks for under and overflow
    A->arr[i] = A->arr[i] - B->arr[i];
  }
  return 0;
};

int mat_minus(f32_mat *A, f32_mat *B) {
  if (A->cols != B->cols || A->rows != B->rows)
    return 1;

  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    // Maybe add bound checks
    A->matrix[i] = A->matrix[i] - B->matrix[i];
  }
  return 0;
};

int add(f32_arr *A, f32_arr *B) {
  if (A->length != B->length)
    return 1;

  for (unsigned int i = 0; i < A->length; i++) {
    // Maybe add bound checks for under and overflow
    A->arr[i] = A->arr[i] + B->arr[i];
  }
  return 0;
};

int mat_add(f32_mat *A, f32_mat *B) {
  if (A->cols != B->cols || A->rows != B->rows)
    return 1;

  for (unsigned int i = 0; i < A->rows * A->cols; i++) {
    // Maybe add bound checks
    A->matrix[i] = A->matrix[i] + B->matrix[i];
  }
  return 0;
};

f32_mat *arr_to_mat(f32_arr *A) {
  f32_mat *matrix = malloc(sizeof(f32_mat));
  if (matrix == NULL)
    return NULL;
  matrix->matrix = A->arr;
  matrix->transposed = 1;
  matrix->rows = A->length;
  matrix->cols = 1;
  return matrix;
};

f32_arr *mat_to_arr(f32_mat *A) {
  f32_arr *array = malloc(sizeof(f32_mat));
  if (array == NULL)
    return NULL;
  array->length = A->cols * A->rows;
  array->arr = A->matrix;
  return array;
};

void mat_print(f32_mat *A) {
  printf("[[");
  for (unsigned int i = 0; i < A->rows; i++) {
    for (unsigned int j = 0; j < A->cols; j++) {
      printf("%3.1f, ", mat_at(A, i, j));
    }
    if (i != A->rows - 1)
      printf("]\n[");
  }
  printf("]]\n");
};

void arr_print(f32_arr *A) {
  printf("[");
  for (unsigned int i = 0; i < A->length; i++) {
    printf("%3.1f, ", A->arr[i]);
  }
  printf("]\n");
}
