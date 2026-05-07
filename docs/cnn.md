# Convolutional Neural Networks (CNNs)

A Convolutional Neural Network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics.

## 1. Definition

A convolutional neural network (CNN) is a type of **feedforward neural network** that learns features via filter (or kernel) optimization. It is the de-facto standard for deep learning-based approaches to **computer vision** and image processing. CNNs are designed to process data with a grid-like topology (such as images) and are known for their ability to prevent vanishing and exploding gradients through regularization methods like shared weights.

## 2. Architecture

The architecture of a CNN consists of an **input layer**, multiple **hidden layers**, and an **output layer**. The hidden layers typically perform convolutions, followed by other operations like pooling and normalization. CNN architectures exploit three key properties:

*   **3D Volume of Neurons:** Layers have neurons arranged in width, height, and depth. For example, an input image might have dimensions of 32x32x3 (width, height, and RGB channels).
*   **Local Connectivity:** Unlike traditional neural networks where every neuron is connected to all neurons in the previous layer, CNN neurons are connected only to a small region of the previous layer, known as the **receptive field**.
*   **Shared Weights:** The same filter is used across the entire visual field. This significantly reduces the number of parameters and grants the network **translational equivariance** (the ability to recognize a feature regardless of its position in the image).

## 3. Layers

A typical CNN architecture consists of several types of layers stacked together:

*   **Convolutional Layer:** The core building block of a CNN. It performs convolution operations using learnable filters (kernels) to create **feature maps** (or activation maps). These filters detect specific features like edges, textures, or more complex shapes in deeper layers.
*   **Activation Function (ReLU):** Usually follows a convolutional layer to introduce non-linearity into the model. The Rectified Linear Unit (ReLU) is the most common choice.
*   **Pooling Layer:** Used for non-linear down-sampling to reduce the spatial dimensions of the representation. This decreases the number of parameters and computation in the network, helping to control overfitting. Common types include:
    *   **Max Pooling:** Takes the maximum value from a portion of the feature map.
    *   **Average Pooling:** Takes the average value.
*   **Fully Connected (FC) Layer:** Connects every neuron in one layer to every neuron in another, similar to a traditional multilayer perceptron. These layers are typically placed at the end of the network to perform final classification based on the high-level features extracted by the preceding convolutional and pooling layers.

---
*Source: Summarized from Wikipedia (Convolutional Neural Network)*
