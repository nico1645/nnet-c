#include "../include/nnetmodels.h"
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

#define TRAINING_DATA 60000
#define CONTROL_DATA 10000
#define TEST_DATA 10000
#define BITS_PER_BYTE 8
#define IMAGE_SIZE 784 // 28x28 pixels

void load_mnist_images(const char *filename,
                       unsigned char images[TRAINING_DATA][IMAGE_SIZE]) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }

  // Read IDX header
  unsigned char header[16];
  fread(header, sizeof(unsigned char), 16, file);

  // Read image data
  for (int i = 0; i < TRAINING_DATA; ++i) {
    fread(images[i], sizeof(unsigned char), IMAGE_SIZE, file);
  }

  fclose(file);
}

void load_mnist_control_images(const char *filename,
                               unsigned char images[CONTROL_DATA][IMAGE_SIZE]) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }

  // Read IDX header
  unsigned char header[16];
  fread(header, sizeof(unsigned char), 16, file);

  // Read image data
  for (int i = 0; i < CONTROL_DATA; ++i) {
    fread(images[i], sizeof(unsigned char), IMAGE_SIZE, file);
  }

  fclose(file);
}

void load_mnist_control_labels(const char *filename,
                               unsigned char labels[CONTROL_DATA]) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }

  // Read IDX header
  unsigned char header[8];
  fread(header, sizeof(unsigned char), 8, file);

  // Read label data
  fread(labels, sizeof(unsigned char), CONTROL_DATA, file);

  fclose(file);
}

void load_mnist_labels(const char *filename,
                       unsigned char labels[TRAINING_DATA]) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }

  // Read IDX header
  unsigned char header[8];
  fread(header, sizeof(unsigned char), 8, file);

  // Read label data
  fread(labels, sizeof(unsigned char), TRAINING_DATA, file);

  fclose(file);
}

int main(void) {
  srand(time(NULL));
  unsigned char images[TRAINING_DATA][IMAGE_SIZE];
  unsigned char labels[TRAINING_DATA];
  unsigned char control_images[CONTROL_DATA][IMAGE_SIZE];
  unsigned char control_labels[CONTROL_DATA];
  const char *filename =
      "/Users/nba/projects/nnet/mnist/train-images.idx3-ubyte";
  const char *filename2 =
      "/Users/nba/projects/nnet/mnist/train-labels.idx1-ubyte";
  const char *filename3 =
      "/Users/nba/projects/nnet/mnist/t10k-images.idx3-ubyte";
  const char *filename4 =
      "/Users/nba/projects/nnet/mnist/t10k-labels.idx1-ubyte";
  load_mnist_images(filename, images);
  load_mnist_labels(filename2, labels);
  load_mnist_control_images(filename3, control_images);
  load_mnist_labels(filename4, control_labels);

  // Example: Print the pixel values of the first image
 /*   for (int i = 0; i < IMAGE_SIZE; ++i) {
      printf("%3u ", images[2][i]);
      if ((i + 1) % 28 == 0) {
        printf("\n");
      }
    }
    printf("!!!!!!!!!!!!!!!!!%3u\n", labels[2]);
  */    

  unsigned int hlayers[] = {300};
  f32_model *model; //= model_create_ffnn(784, 10, hlayers, 2);
  // loading the training data one example after the other

  model = model_create_ffnn(IMAGE_SIZE, 10, hlayers, 1);

  if (model == NULL)
    printf("Allocaiton failed");

  // Process the chunk of data here
  float *input = model->input_layer->matrix;
  float *output = model->label->matrix;
  float learning_rate = 0.0001;

    for (int z = 0; z < 8; z++) {

  for (int i = 0; i < TRAINING_DATA; i++) {
    for (unsigned int j = 0; j < 10; j++) {
      if (j == (unsigned int)(labels[i])) {
        output[j] = 1;
      } else {
        output[j] = 0;
      }
    }

    for (unsigned int j = 0; j < IMAGE_SIZE; j++) {
      input[j] = images[i][j];
    }
    model_train_item(model, learning_rate, OPTIMIZER.SGD_MOMENTUM);

    if (i % 5000 == 0) {
      printf("%d\n", i);
    }
  }

  int correct = 0;

  for (int i = 0; i < CONTROL_DATA; i++) {
    for (unsigned int j = 0; j < 10; j++) {
      if (j == (unsigned int)(control_labels[i])) {
        output[j] = 1;
      } else {
        output[j] = 0;
      }
    }

    for (unsigned int j = 0; j < IMAGE_SIZE; j++) {
      input[j] = control_images[i][j];
    }
    model_forward_propagate(model);
    unsigned int index = 0;
    for (unsigned int j = 1; j < model->label->cols * model->label->rows; j++) {
      if (model->output_layer->matrix[j] > model->output_layer->matrix[j - 1]) {
        index = j;
      }
    }
    if ((unsigned int)(control_labels[i]) == index)
      correct++;
  }
  printf("CORRECT: %d out of 10'000\n", correct);
  printf("Precentage: %f\n", correct * 100.0 / (float)CONTROL_DATA);
  }

  free_model(model);

  return 0;
}
