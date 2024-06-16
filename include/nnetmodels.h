#ifndef nnetmodels
#define nnetmodels

#include "ndimarr.h"
#include "nnetfunc.h"

typedef struct {
  f32_mat *input_layer;
  f32_mat *output_layer;
  unsigned int size; // Number of layers excluding the input_layer
  f32_mat *biases;   // biases[i] are the biases of layer i
  f32_mat *weights;  // weigths[i] are the weights from layer i to i+1 with j
                     // rows and k columns where j is the number of neurons in
                     // i+1 and k is the number of neurons in i
  f32_mat *activations; // activations[i] is the activations vector of layer i
  f32_mat *errors; // error[i] is the error vector of layer i. Derivative of the
                   // cost function w.r.t. weights
  f32_mat *dCdW;   // Derivative of the cost function w.r.t. weights
  f32_mat *pre_activations; // the z vector before activation function
  f32_mat *label; // The true label of the training data
  // Needed for some optimizers
  f32_mat *m_dCdW; // First moment vector for weights
  f32_mat *v_dCdW; // Second moment vector for weights
  f32_mat *m_errors; // First moment vector for biases
  f32_mat *v_errors; // Second moment vector for biases
  unsigned int step;
} f32_model;

struct OPTIMIZER {
    void (*SGD)(f32_model*,float);
    void (*ADAM)(f32_model*,float);
    void (*SGD_MOMENTUM)(f32_model*,float);
};

extern const struct OPTIMIZER OPTIMIZER;

// The model is created and the biases are set to 0 and the weights are set
// randomly, the size of hidden_layers is the number of the hidden_layers an the
// individual integers the size of each layer
f32_model *model_create_ffnn(unsigned int size_input_layer,
                             unsigned int size_output_layer,
                             unsigned int hidden_layers[],
                             unsigned int number_of_hlayers);

// forward propogates using the input_layer and calculating all activations and
// the output_layer
void model_forward_propagate(f32_model *model);

// backward propogates and calculating the errors based on the activations,
// further the derivatives of the cost function w.r.t weights and biases
void model_backward_propagate(f32_model *model);

float compute_mse_loss(f32_arr *input_layer, f32_arr *true_labels);
float compute_cross_entropy_loss(f32_mat *output_layer, f32_mat *true_labels);

void model_sgd(f32_model *model, float learning_rate);
void model_adam(f32_model *model, float learning_rate);
void model_sgd_momentum(f32_model *model, float learning_rate);

void model_train_item(f32_model *model, float learning_rate, void (*optim_func)(f32_model*,float));

void free_model(f32_model *model);

#endif
