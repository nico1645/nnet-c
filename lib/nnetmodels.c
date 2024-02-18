#include "../include/nnetmodels.h"
#include <math.h>
#include <string.h>

f32_model *model_create_ffnn(unsigned int size_input_layer,
                             unsigned int size_output_layer,
                             unsigned int hidden_layers[],
                             unsigned int number_of_hlayers) {
  f32_model *model = malloc(sizeof(f32_model));
  if (model == NULL) {
    return NULL;
  }
  model->size = number_of_hlayers + 1;
  // Init input and output layers
  float *input_arr = malloc(sizeof(float) * size_input_layer);
  float *output_arr = malloc(sizeof(float) * size_output_layer);
  float *label_arr = malloc(sizeof(float) * size_output_layer);
  if (input_arr == NULL | output_arr == NULL) {
    return NULL;
  }
  f32_mat *input = create_mnmat(input_arr, 1, size_input_layer);
  transpose(input);
  f32_mat *output = create_mnmat(output_arr, 1, size_output_layer);
  transpose(output);
  f32_mat *label = create_mnmat(label_arr, 1, size_output_layer);
  transpose(label);
  model->input_layer = input;
  model->output_layer = output;
  model->label = label;

  // Init the rest
  f32_mat *weights = malloc(sizeof(f32_mat) * model->size);
  f32_mat *biases = malloc(sizeof(f32_mat) * model->size);
  f32_mat *activations = malloc(sizeof(f32_mat) * model->size);
  f32_mat *errors = malloc(sizeof(f32_mat) * model->size);
  f32_mat *dCdW = malloc(sizeof(f32_mat) * model->size);
  f32_mat *pre_activations = malloc(sizeof(f32_mat) * model->size);
  if (weights == NULL || biases == NULL || activations == NULL ||
      errors == NULL || dCdW == NULL || pre_activations == NULL)
    return NULL;

  for (unsigned int i = 0; i < model->size; i++) {
    if (i == model->size - 1) {
      float *arr = malloc(sizeof(float) * size_output_layer);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, size_output_layer);
      transpose(mat);
      pre_activations[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      transpose(mat);
      pre_activations[i] = *mat;
    }
    if (i == model->size - 1) {
      float *arr = malloc(sizeof(float) * size_output_layer);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, size_output_layer);
      transpose(mat);
      errors[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      transpose(mat);
      errors[i] = *mat;
    }
    if (i == model->size - 1) {
      float *arr = malloc(sizeof(float) * size_output_layer);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, size_output_layer);
      transpose(mat);
      activations[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      transpose(mat);
      activations[i] = *mat;
    }
    if (i == model->size - 1) {
      float *arr = malloc(sizeof(float) * size_output_layer);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, size_output_layer);
      mat_fill_zeros(mat);
      transpose(mat);
      biases[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      mat_fill_zeros(mat);
      transpose(mat);
      biases[i] = *mat;
    }
    if (i == 0) {
      float *arr = malloc(sizeof(float) * size_input_layer * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], size_input_layer);
      mat_fill_rand(mat, -1, 1);
      weights[i] = *mat;
    } else if (i == model->size - 1) {
      float *arr =
          malloc(sizeof(float) * size_output_layer * hidden_layers[i - 1]);
      f32_mat *mat = create_mnmat(arr, size_output_layer, hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_rand(mat, -1, 1);
      weights[i] = *mat;
    } else {
      float *arr =
          malloc(sizeof(float) * hidden_layers[i - 1] * hidden_layers[i]);
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_rand(mat, -1, 1);
      weights[i] = *mat;
    }
    if (i == 0) {
      float *arr = malloc(sizeof(float) * size_input_layer * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], size_input_layer);
      dCdW[i] = *mat;
    } else if (i == model->size - 1) {
      float *arr =
          malloc(sizeof(float) * size_output_layer * hidden_layers[i - 1]);
      f32_mat *mat = create_mnmat(arr, size_output_layer, hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      dCdW[i] = *mat;
    } else {
      float *arr =
          malloc(sizeof(float) * hidden_layers[i - 1] * hidden_layers[i]);
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      dCdW[i] = *mat;
    }
  }
  model->weights = weights;
  model->biases = biases;
  model->activations = activations;
  model->errors = errors;
  model->biases = biases;
  model->pre_activations = pre_activations;
  model->dCdW = dCdW;

  return model;
};

void free_model(f32_model *model) {
  // Free matrices inside arrays
  for (unsigned int i = 0; i < model->size; i++) {
    free(model->errors[i].matrix);
    free(model->biases[i].matrix);
    free(model->weights[i].matrix);
    free(model->activations[i].matrix);
    free(model->pre_activations[i].matrix);
    free(model->dCdW[i].matrix);
  }

  // Free arrays
  free(model->errors);
  free(model->biases);
  free(model->weights);
  free(model->activations);
  free(model->pre_activations);
  free(model->dCdW);

  // Free input/output layers and label
  free(model->output_layer->matrix);
  free(model->output_layer);
  free(model->input_layer->matrix);
  free(model->input_layer);
  free(model->label->matrix);
  free(model->label);

  // Free model itself
  free(model);
};

void model_forward_propagate(f32_model *model) {
  for (unsigned int i = 0; i < model->size; i++) {
    if (i == 0) {
      int err = mat_mul_inplace(&model->weights[i], model->input_layer,
                                &model->pre_activations[i]);
      mat_add(&model->pre_activations[i], &model->biases[i]);
      for (unsigned int j = 0;
           j < model->pre_activations[i].rows * model->pre_activations[i].cols;
           j++) {
        model->activations[i].matrix[j] = model->pre_activations[i].matrix[j];
      };
      mat_func(&model->activations[i], leaky_relu);
    } else if (i == model->size - 1) {
      int err = mat_mul_inplace(&model->weights[i], &model->activations[i - 1],
                                &model->pre_activations[i]);
      mat_add(&model->pre_activations[i], &model->biases[i]);
      for (unsigned int j = 0;
           j < model->pre_activations[i].rows * model->pre_activations[i].cols;
           j++) {
        model->activations[i].matrix[j] = model->pre_activations[i].matrix[j];
      };
      softmax(&model->activations[i]);
      for (unsigned int j = 0;
           j < model->activations[i].rows * model->activations[i].cols; j++) {
        model->output_layer->matrix[j] = model->activations[i].matrix[j];
      };
    } else {
      int err = mat_mul_inplace(&model->weights[i], &model->activations[i - 1],
                                &model->pre_activations[i]);
      mat_add(&model->pre_activations[i], &model->biases[i]);
      for (unsigned int j = 0;
           j < model->pre_activations[i].rows * model->pre_activations[i].cols;
           j++) {
        model->activations[i].matrix[j] = model->pre_activations[i].matrix[j];
      };
      mat_func(&model->activations[i], leaky_relu); // leaky_relu
    }
  }
};

void model_backward_propagate(f32_model *model) {
  for (int i = (int)model->size - 1; i >= 0; i--) {
    if (i == (int)model->size - 1) {
      for (unsigned int j = 0; j < model->label->cols * model->label->rows;
           j++) {
        model->errors[i].matrix[j] = -model->label->matrix[j];
      }
      mat_add(&model->errors[i], &model->activations[i]);
    } else {
      transpose(&model->weights[i + 1]);
      mat_mul_inplace(&model->weights[i + 1], &model->errors[i + 1],
                      &model->errors[i]);
      transpose(&model->weights[i + 1]);
      mat_func(&model->pre_activations[i], leaky_relu_derivative);
      hamard_prod(&model->errors[i], &model->pre_activations[i]);
    }
  }

  for (unsigned int i = 0; i < model->size; i++) {
    if (i == 0) {
      transpose(model->input_layer);
      mat_mul_inplace(&model->errors[i], model->input_layer, &model->dCdW[i]);
      transpose(model->input_layer);
    } else {
      transpose(&model->activations[i - 1]);

      int err = mat_mul_inplace(&model->errors[i], &model->activations[i - 1],
                                &model->dCdW[i]);
      if (err == 1)
        printf("ERR 3");
      transpose(&model->activations[i - 1]);
    }
  }
};

void model_sgd(f32_model *model, float learning_rate) {
  f32_mat *dCdW = model->dCdW;
  f32_mat *errors = model->errors;
  f32_mat *weights = model->weights;
  f32_mat *biases = model->biases;
  for (unsigned int i = 0; i < model->size; i++) {
    mat_scalar_mul(&dCdW[i], learning_rate);
    mat_scalar_mul(&errors[i], learning_rate);
    mat_minus(&weights[i], &dCdW[i]);
    mat_minus(&biases[i], &errors[i]);
  }
};

float compute_cross_entropy_loss(f32_mat *output_layer, f32_mat *true_labels) {
  float loss = 0;
  float epsilon = 1e-9;
  for (unsigned int i = 0; i < true_labels->cols * true_labels->rows; i++) {
    if (fabsf(output_layer->matrix[i] - epsilon) < epsilon) {
      loss += true_labels->matrix[i] * logf(epsilon);
    } else {
      loss += true_labels->matrix[i] * logf(output_layer->matrix[i]);
    }
  }

  return -loss;
};

void model_train_epoch(f32_model *model, float learning_rate) {
  model_forward_propagate(model);
  model_backward_propagate(model);
  model_sgd(model, learning_rate);
};
