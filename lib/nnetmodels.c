#include "../include/nnetmodels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const struct OPTIMIZER OPTIMIZER = {&model_sgd, &model_adam, &model_sgd_momentum};

f32_model *model_create_ffnn(unsigned int size_input_layer,
                             unsigned int size_output_layer,
                             unsigned int hidden_layers[],
                             unsigned int number_of_hlayers) {
  f32_model *model = malloc(sizeof(f32_model));
  if (model == NULL) {
    return NULL;
  }
  model->size = number_of_hlayers + 1;
  model->step = 0;
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
  f32_mat *m_errors = malloc(sizeof(f32_mat) * model->size);
  f32_mat *v_errors = malloc(sizeof(f32_mat) * model->size);
  f32_mat *dCdW = malloc(sizeof(f32_mat) * model->size);
  f32_mat *m_dCdW = malloc(sizeof(f32_mat) * model->size);
  f32_mat *v_dCdW = malloc(sizeof(f32_mat) * model->size);
  f32_mat *pre_activations = malloc(sizeof(f32_mat) * model->size);
  if (weights == NULL || biases == NULL || activations == NULL ||
      errors == NULL || m_errors == NULL || v_errors == NULL 
      || dCdW == NULL || m_dCdW == NULL || v_dCdW == NULL || pre_activations == NULL)
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
      mat_fill_zeros(mat);
      m_errors[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      transpose(mat);
      mat_fill_zeros(mat);
      m_errors[i] = *mat;
    }
    if (i == model->size - 1) {
      float *arr = malloc(sizeof(float) * size_output_layer);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, size_output_layer);
      transpose(mat);
      mat_fill_zeros(mat);
      v_errors[i] = *mat;
    } else {
      float *arr = malloc(sizeof(float) * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, 1, hidden_layers[i]);
      transpose(mat);
      mat_fill_zeros(mat);
      v_errors[i] = *mat;
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
      mat_fill_rand(mat, -0.3, 0.3);
      weights[i] = *mat;
    } else if (i == model->size - 1) {
      float *arr =
          malloc(sizeof(float) * size_output_layer * hidden_layers[i - 1]);
      f32_mat *mat = create_mnmat(arr, size_output_layer, hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_rand(mat, -0.3, 0.3);
      weights[i] = *mat;
    } else {
      float *arr =
          malloc(sizeof(float) * hidden_layers[i - 1] * hidden_layers[i]);
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_rand(mat, -0.3, 0.3);
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
    if (i == 0) {
      float *arr = malloc(sizeof(float) * size_input_layer * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], size_input_layer);
      mat_fill_zeros(mat);
      m_dCdW[i] = *mat;
    } else if (i == model->size - 1) {
      float *arr =
          malloc(sizeof(float) * size_output_layer * hidden_layers[i - 1]);
      f32_mat *mat = create_mnmat(arr, size_output_layer, hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_zeros(mat);
      m_dCdW[i] = *mat;
    } else {
      float *arr =
          malloc(sizeof(float) * hidden_layers[i - 1] * hidden_layers[i]);
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_zeros(mat);
      m_dCdW[i] = *mat;
    }
    if (i == 0) {
      float *arr = malloc(sizeof(float) * size_input_layer * hidden_layers[i]);
      if (arr == NULL)
        return NULL;
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], size_input_layer);
      mat_fill_zeros(mat);
      v_dCdW[i] = *mat;
    } else if (i == model->size - 1) {
      float *arr =
          malloc(sizeof(float) * size_output_layer * hidden_layers[i - 1]);
      f32_mat *mat = create_mnmat(arr, size_output_layer, hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_zeros(mat);
      v_dCdW[i] = *mat;
    } else {
      float *arr =
          malloc(sizeof(float) * hidden_layers[i - 1] * hidden_layers[i]);
      f32_mat *mat = create_mnmat(arr, hidden_layers[i], hidden_layers[i - 1]);
      if (arr == NULL)
        return NULL;
      mat_fill_zeros(mat);
      v_dCdW[i] = *mat;
    }
  }
  model->weights = weights;
  model->biases = biases;
  model->activations = activations;
  model->errors = errors;
  model->m_errors = m_errors;
  model->v_errors = v_errors;
  model->biases = biases;
  model->pre_activations = pre_activations;
  model->dCdW = dCdW;
  model->m_dCdW = m_dCdW;
  model->v_dCdW = v_dCdW;
  

  return model;
};

void free_model(f32_model *model) {
  // Free matrices inside arrays
  for (unsigned int i = 0; i < model->size; i++) {
    free(model->errors[i].matrix);
    free(model->m_errors[i].matrix);
    free(model->v_errors[i].matrix);
    free(model->biases[i].matrix);
    free(model->weights[i].matrix);
    free(model->activations[i].matrix);
    free(model->pre_activations[i].matrix);
    free(model->dCdW[i].matrix);
    free(model->m_dCdW[i].matrix);
    free(model->v_dCdW[i].matrix);
  }

  // Free arrays
  free(model->errors);
  free(model->biases);
  free(model->weights);
  free(model->activations);
  free(model->pre_activations);
  free(model->dCdW);

  // Free input/output layers and label
  mat_free(model->output_layer);
  mat_free(model->input_layer);
  mat_free(model->label);

  // Free model itself
  free(model);
};

void model_forward_propagate(f32_model *model) {
  for (unsigned int i = 0; i < model->size; i++) {
    if (i == 0) {
      mat_mul_inplace(&model->weights[i], model->input_layer,
                                &model->pre_activations[i]);
      mat_add(&model->pre_activations[i], &model->biases[i]);
      for (unsigned int j = 0;
           j < model->pre_activations[i].rows * model->pre_activations[i].cols;
           j++) {
        model->activations[i].matrix[j] = model->pre_activations[i].matrix[j];
      };
      mat_func(&model->activations[i], leaky_relu);
    } else if (i == model->size - 1) {
      mat_mul_inplace(&model->weights[i], &model->activations[i - 1],
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
      mat_mul_inplace(&model->weights[i], &model->activations[i - 1],
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
      hadamard_prod(&model->errors[i], &model->pre_activations[i]);
    }
  }

  for (unsigned int i = 0; i < model->size; i++) {
    if (i == 0) {
      transpose(model->input_layer);
      mat_mul_inplace(&model->errors[i], model->input_layer, &model->dCdW[i]);
      transpose(model->input_layer);
    } else {
      transpose(&model->activations[i - 1]);

      mat_mul_inplace(&model->errors[i], &model->activations[i - 1],
                                &model->dCdW[i]);
      transpose(&model->activations[i - 1]);
    }
  }
};

void model_sgd(f32_model *model, float learning_rate) {
  for (unsigned int i = 0; i < model->size; i++) {
    f32_mat *dCdW = &model->dCdW[i];
    f32_mat *errors = &model->errors[i];
    f32_mat *weights = &model->weights[i];
    f32_mat *biases = &model->biases[i];
    mat_scalar_mul(dCdW, learning_rate);
    mat_scalar_mul(errors, learning_rate);
    mat_minus(weights, dCdW);
    mat_minus(biases, errors);
  }
  model->step++;
};

void model_adam(f32_model *model, float learning_rate) {
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-6;
  unsigned int t = model->step;

  // Iterate over each layer or parameter group
  for (unsigned int i = 0; i < model->size; i++) {
      f32_mat *dCdW = &model->dCdW[i];
      f32_mat *errors = &model->errors[i];
      f32_mat *weights = &model->weights[i];
      f32_mat *biases = &model->biases[i];
      f32_mat *m_dCdW = &model->m_dCdW[i];
      f32_mat *v_dCdW = &model->v_dCdW[i];
      f32_mat *m_errors = &model->m_errors[i];
      f32_mat *v_errors = &model->v_errors[i];

      // Update biased first moment estimate (m_dCdW and m_errors)
      mat_scalar_mul(m_dCdW, beta1);
      mat_scalar_mul(m_errors, beta1);

      f32_mat *tmp_dCdW_copy = mat_deep_copy(dCdW);
      f32_mat *tmp_errors_copy = mat_deep_copy(errors);

      mat_scalar_mul(tmp_dCdW_copy, 1.0f - beta1);
      mat_scalar_mul(tmp_errors_copy, 1.0f - beta1);

      mat_add(m_dCdW, tmp_dCdW_copy);
      mat_add(m_errors, tmp_errors_copy);

      mat_free(tmp_dCdW_copy);
      mat_free(tmp_errors_copy);

      // Update biased second raw moment estimate (v_dCdW and v_errors)
      mat_scalar_mul(v_dCdW, beta2);
      mat_scalar_mul(v_errors, beta2);

      tmp_dCdW_copy = mat_deep_copy(dCdW);
      tmp_errors_copy = mat_deep_copy(errors);

      mat_scalar_pow(tmp_dCdW_copy, 2);
      mat_scalar_pow(tmp_errors_copy, 2);

      mat_scalar_mul(tmp_dCdW_copy, 1.0f - beta2);
      mat_scalar_mul(tmp_errors_copy, 1.0f - beta2);

      mat_add(v_dCdW, tmp_dCdW_copy);
      mat_add(v_errors, tmp_errors_copy);

      mat_free(tmp_dCdW_copy);
      mat_free(tmp_errors_copy);

      // Compute bias-corrected first moment estimate (m_dCdW_hat and m_errors_hat)
      f32_mat *m_dCdW_hat = mat_deep_copy(m_dCdW);
      mat_scalar_div(m_dCdW_hat, 1.0f - powf(beta1, t + 1));

      f32_mat *m_errors_hat = mat_deep_copy(m_errors);
      mat_scalar_div(m_errors_hat, 1.0f - powf(beta1, t + 1));

      // Compute bias-corrected second raw moment estimate (v_dCdW_hat and v_errors_hat)
      f32_mat *v_dCdW_hat = mat_deep_copy(v_dCdW);
      mat_scalar_div(v_dCdW_hat, 1.0f - powf(beta2, t + 1));

      f32_mat *v_errors_hat = mat_deep_copy(v_errors);
      mat_scalar_div(v_errors_hat, 1.0f - powf(beta2, t + 1));

      // Update weights and biases
      f32_mat *sqrt_v_dCdW_hat = mat_deep_copy(v_dCdW_hat);
      mat_sqrt(sqrt_v_dCdW_hat);
      mat_scalar_add(sqrt_v_dCdW_hat, epsilon);

      f32_mat *sqrt_v_errors_hat = mat_deep_copy(v_errors_hat);
      mat_sqrt(sqrt_v_errors_hat);
      mat_scalar_add(sqrt_v_errors_hat, epsilon);

      f32_mat *update_weights = mat_deep_copy(m_dCdW_hat);
      hadamard_div(update_weights, sqrt_v_dCdW_hat);
      mat_scalar_mul(update_weights, learning_rate);
      mat_minus(weights, update_weights);

      f32_mat *update_biases = mat_deep_copy(m_errors_hat);
      hadamard_div(update_biases, sqrt_v_errors_hat);
      mat_scalar_mul(update_biases, learning_rate);
      mat_minus(biases, update_biases);

      // Free temporary matrices
      mat_free(m_dCdW_hat);
      mat_free(m_errors_hat);
      mat_free(v_dCdW_hat);
      mat_free(v_errors_hat);
      mat_free(sqrt_v_dCdW_hat);
      mat_free(sqrt_v_errors_hat);
      mat_free(update_weights);
      mat_free(update_biases);
  }

  model->step++;
};

void model_sgd_momentum(f32_model *model, float learning_rate) {
    float momentum = 0.9f;

    for (unsigned int i = 0; i < model->size; i++) {
      f32_mat *dCdW = &model->dCdW[i];
      f32_mat *errors = &model->errors[i];
      f32_mat *weights = &model->weights[i];
      f32_mat *biases = &model->biases[i];
      f32_mat *v_dCdW = &model->v_dCdW[i];
      f32_mat *v_errors = &model->v_errors[i];

      // Update momentum vectors for weights
      mat_scalar_mul(v_dCdW, momentum);
      f32_mat *tmp_dCdW = mat_deep_copy(dCdW);
      mat_scalar_mul(tmp_dCdW, learning_rate);
      mat_add(v_dCdW, tmp_dCdW);

      // Update momentum vectors for biases
      mat_scalar_mul(v_errors, momentum);
      f32_mat *tmp_errors = mat_deep_copy(errors);
      mat_scalar_mul(tmp_errors, learning_rate);
      mat_add(v_errors, tmp_errors);

      // Update weights and biases
      mat_minus(weights, v_dCdW);
      mat_minus(biases, v_errors);

      // Free temporary matrices
      mat_free(tmp_dCdW);
      mat_free(tmp_errors);
    }
  model->step++;
}

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

void model_train_item(f32_model *model, float learning_rate, void (*optim_func)(f32_model*,float)) {
  model_forward_propagate(model);
  model_backward_propagate(model);
  for (unsigned int i = 0; i < model->size; i++ ) {
    mat_clip_low(&model->dCdW[i], -5);
    mat_clip_high(&model->dCdW[i], 5);
    mat_clip_low(&model->errors[i], -5);
    mat_clip_high(&model->errors[i], 5);
  }
  optim_func(model, learning_rate);
};
