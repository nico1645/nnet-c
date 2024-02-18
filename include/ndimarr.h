#ifndef ndimarr
#define ndimarr

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  unsigned int length;
  float *arr;
} f32_arr;

// Matrix points to an array in row major order
typedef struct {
  unsigned int rows;
  unsigned int cols;
  unsigned int transposed;
  float *matrix;
} f32_mat;

void mat_func(f32_mat *A, float (*f)(float));
void arr_func(f32_arr *A, float (*f)(float));

void mat_fill_zeros(f32_mat *A);
void arr_fill_zeros(f32_arr *A);

// Make sure srand() is already initialized with a seed
void mat_fill_rand(f32_mat *A, float min, float max);
void arr_fill_rand(f32_arr *A, float min, float max);

// Operations are inplace and the result is stored in A. The return type can be
// used to check for errors.
int transpose(f32_mat *A);
int minus(f32_arr *A, f32_arr *B);
int mat_minus(f32_mat *A, f32_mat *B);
int add(f32_arr *A, f32_arr *B);
int mat_add(f32_mat *A, f32_mat *B);
int hadamard_prod(f32_mat *A, f32_mat *B);
int mat_scalar_mul(f32_mat *A, float a);
int mat_mul_inplace(f32_mat *A, f32_mat *B, f32_mat *result);

// Careful no bounds are being checked
float mat_at(f32_mat *A, unsigned int i, unsigned int j);
void mat_set(f32_mat *A, float value, unsigned int i, unsigned int j);

// The dot product is stored in prod. The return type can be used to check for
// errors.
int dot_prod(f32_arr *A, f32_arr *B, float *prod);

// Statically allocating new memory, and returning the newly created struct.
f32_mat *mat_mul(f32_mat *A, f32_mat *B);
f32_mat *create_mnmat(float *A, int rows, int columns);
f32_arr *create_narr(float *A, int length);

// The the matrix points to the same location as the array with rows = 1 and
// cols = arr.length.
f32_mat *arr_to_mat(f32_arr *A);
// The f32_arr points to the same data as the matrix array and the length is set
// to rows * columns.
f32_arr *mat_to_arr(f32_mat *A);

void mat_print(f32_mat *A);
void arr_print(f32_arr *A);

#endif
