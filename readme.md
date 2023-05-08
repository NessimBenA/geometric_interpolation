# Deep Learning Model for Binary Classification Optimization

This notebook demonstrates the optimization of a deep learning model for binary classification using PyTorch. The main objective is to adjust the final layer's weights to control the classification area of one of the classes. The notebook is divided into several sections, each focusing on specific parts of the model implementation and optimization process, including dataset creation, model architecture, and model optimization using Optuna.

## Table of Contents

1. [Dataset Creation](#Dataset-Creation)
2. [Model Architecture](#Model-Architecture)
3. [Training the Model](#Training-the-Model)
4. [Area Calculation for Class 1](#Area-Calculation-for-Class-1)
5. [Neuron Projection](#Neuron-Projection)
6. [Interpolation Function](#Interpolation-Function)
7. [Final Layer Weight Modification](#Final-Layer-Weight-Modification)
8. [Optimization with Optuna](#Optimization-with-Optuna)
9. [Visualizing Classification](#Visualizing-Classification)

### Dataset Creation

This section describes the creation of a synthetic dataset for binary classification using the `make_blobs` function from the `sklearn.datasets` package. The dataset consists of 1000 samples, each with two features, and two target classes.

### Model Architecture

The model architecture is based on a simple feed-forward neural network implemented using PyTorch. The network consists of four linear layers followed by ReLU activation functions, with the final layer outputting a two-dimensional vector for the binary classification task.

### Training the Model

The training process involves setting up the loss function (cross-entropy loss) and the optimizer (Adam) for the model. The model is trained for a specified number of epochs, with the loss and accuracy reported after each epoch.

### Area Calculation for Class 1

This section covers the calculation of the area of class 1 in the output space of the model. The function `area_of_class1` computes the area by creating a grid of points, evaluating the model on these points, and counting the number of points classified as class 1.

### Neuron Projection

The `neuron_projection` function computes the projection of the neurons in the final layer of the model. It returns the projected endpoint of the neuron, the projection of the final layer, the length of the projection, and the non-negative indexes of the final layer's weights.

### Interpolation Function

The interpolation function, `interpfun`, creates a cubic interpolation of the neuron projections using the `interp1d` function from the `scipy.interpolate` package. This function can be used to modify the final layer's weights, as shown in the next section.

### Final Layer Weight Modification

The `final_layer_weight_modifier` function modifies the weights of the final layer by applying the cubic interpolation function to the neuron projections. The modified weights are returned as a new weight tensor.

### Optimization with Optuna

Optuna is a hyperparameter optimization library for machine learning models. In this notebook, Optuna is used to find the optimal modification value for the final layer's weights. The objective function minimizes the absolute difference between the area of class 1 and the target area (80% of the original area).

### Visualizing Classification

The `visualize_classification` function plots the decision boundary of the model along with the dataset points. This visualization helps in understanding how well the model has learned to classify the data and the effect of the optimization on the classification area.

## Usage

To run the notebook, make sure you have the necessary packages installed, including `torch`, `numpy`, `matplotlib`, `sklearn`, `scipy`, and `optuna`. You can install these packages using `pip



# Deep Learning Model for Decision Boundary Manipulation

This notebook demonstrates the manipulation of a deep learning model's decision boundary using PyTorch. The main objective is to adjust the final layer's weights to control the classification area of one of the classes. The notebook is divided into several sections, each focusing on specific parts of the model implementation and optimization process, including loading the model, preprocessing the image, and model optimization using Optuna.

## Table of Contents

1. [Loading the Model and Preprocessing the Image](#Loading-the-Model-and-Preprocessing-the-Image)
2. [Visualizing the Segmented Image](#Visualizing-the-Segmented-Image)
3. [Area Calculation for Class 1](#Area-Calculation-for-Class-1)
4. [Neuron Projection](#Neuron-Projection)
5. [Interpolation Function](#Interpolation-Function)
6. [Final Layer Weight Modification](#Final-Layer-Weight-Modification)
7. [Optimization with Optuna](#Optimization-with-Optuna)
8. [Visualizing Results](#Visualizing-Results)

### Loading the Model and Preprocessing the Image

This section describes loading a DeepLabV3 ResNet50 model and preprocessing an input image. The model is obtained from the PyTorch Hub, and the image is preprocessed using the `transforms` module from the `torchvision` package.

### Visualizing the Segmented Image

The `plot_img` function visualizes the segmented image obtained from the deep learning model. The segmented image is overlayed on top of the original image to provide better context.

### Area Calculation for Class 1

This section covers the calculation of the area of class 1 in the output space of the model. The function `area_of_class1` computes the area by evaluating the model on the input image and counting the number of pixels classified as class 1.

### Neuron Projection

The `neuron_projection` function computes the projection of the neurons in the final layer of the model. It returns the projected endpoint of the neuron, the projection of the final layer, the length of the projection, and the non-negative indexes of the final layer's weights.

### Interpolation Function

The interpolation function, `interpfun`, creates a cubic interpolation of the neuron projections using the `interp1d` function from the `scipy.interpolate` package. This function can be used to modify the final layer's weights, as shown in the next section.

### Final Layer Weight Modification

The `final_layer_weight_modifier` function modifies the weights of the final layer by applying the cubic interpolation function to the neuron projections. The modified weights are returned as a new weight tensor.

### Optimization with Optuna

Optuna is a hyperparameter optimization library for machine learning models. In this notebook, Optuna is used to find the optimal modification value for the final layer's weights. The objective function minimizes the absolute difference between the area of class 1 and the target area (91% of the original area).

### Visualizing Results

The `plot_img` function is called again to visualize the segmented image after the decision boundary manipulation. This visualization helps in understanding the effect of the optimization on the classification area.

## Usage

To run the notebook, make sure you have the necessary packages installed, including `torch`, `torchvision`, `numpy`, `matplotlib`, `scipy`, and `optuna`. You can install these packages using `pip`.