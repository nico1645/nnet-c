
#include "../include/ndimarr.h"
#include "../include/nnetfunc.h"
#include "../include/nnetmodels.h"

int main(void) {
  srand(83742);
  float v1[] = {1, 2, 3, 4};
  float v2[] = {-4, 3, -2, 1};
  float m[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  f32_mat *m1 = create_mnmat(m, 4, 4);
  mat_print(m1);
  printf("Test ndimarr library: \n");
  f32_arr *arr1 = create_narr(v1, 4);
  f32_arr *arr2 = create_narr(v2, 4);
  float a = 100.000;
  dot_prod(arr1, arr2, &a);
  printf("1. dot_prod: %.2f\n", a);
  f32_mat *mat1 = arr_to_mat(arr1);
  f32_mat *mat2 = arr_to_mat(arr2);
  transpose(mat1);
  printf("2. transpose:\n");
  mat_print(mat2);
  printf("3. mat_mul:\n");
  f32_mat *res1 = mat_mul(mat1, mat2);
  mat_print(res1);
  transpose(mat1);
  transpose(mat2);
  f32_mat *res2 = mat_mul(mat1, mat2);
  transpose(m1);
  printf("4. mat_mul:\n");
  mat_print(res2);
  mat_fill_rand(res2, 0, 10);
  mat_func(res2, sigmoid);
  printf("5. mat_fill_rand:\n");
  mat_print(res2);
  mat_fill_zeros(res2);
  printf("6. mat_fill_zeros:\n");
  mat_print(res2);

  free(m1);
  free(arr1);
  free(arr2);
  free(mat1);
  free(mat2);
  free(res1->matrix);
  free(res2->matrix);
  free(res1);
  free(res2);

  unsigned int hlayers[] = {120, 120, 3};
  f32_model *model = model_create_ffnn(784, 10, hlayers, 3);
  if (model == NULL)
    printf("failed");
  printf("allocated\n");
  for (int i = 0; i < 1000; i++) {
        model_train_epoch(model, 0.01);
    }
model->input_layer->matrix[3] = 1;
  model_forward_propagate(model);
  printf("forward propagated\n");
  model_backward_propagate(model);
  printf("backward propagated\n");
  mat_print(&model->weights[0]);
  model_sgd(model, 0.01);
  printf("gradient descent\n");
  mat_print(model->output_layer);
  float b =compute_cross_entropy_loss(model->output_layer, model->label);
  printf("loss: %f\n", b);

  free_model(model);

  return 0;
}
