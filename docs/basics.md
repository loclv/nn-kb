# Artificial Neural Network Basics

Artificial Neural Networks (ANNs) are computational models inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain.

## 1. Introduction

In machine learning, a **neural network (NN)**—also known as an **artificial neural network (ANN)**—is a computational model inspired by the structure and function of biological neural networks. It consists of connected units called **artificial neurons**, which loosely model the neurons in a brain. 

Each neuron receives signals from others, processes them using a non-linear function, and sends a signal (typically a real number) to further connected neurons. These connections, or **edges**, have "weights" that increase or decrease the strength of the signal, allowing the network to learn and recognize complex patterns.

## 2. Architecture (Organization)

Neural networks are typically organized into multiple **layers**:

*   **Input Layer:** Receives external data.
*   **Output Layer:** Produces the final result.
*   **Hidden Layers:** Intermediate layers between the input and output.
*   **Connectivity:** Traditionally, networks are **fully connected** (every neuron in one layer connects to every neuron in the next). However, some specialized architectures like **Convolutional Neural Networks (CNNs)** use partial connectivity to process spatial data.
*   **Information Flow:** In **feedforward networks**, information moves only in one direction (input to output). In **recurrent networks**, connections can loop back, allowing the network to maintain a state or "memory."

## 3. Learning

The goal of learning is to adjust the **weights** of the network to minimize the difference between predicted and actual results. This is primarily achieved through:

*   **Empirical Risk Minimization:** Optimizing parameters to reduce the "loss" or error in a dataset.
*   **Backpropagation:** The core algorithm used to calculate the gradient of the loss function, allowing weights to be adjusted to compensate for errors found during training.
*   **Hyperparameters:** Configurable settings like **learning rate** (the size of corrective steps) and **batch size** that are set before training begins.
*   **Learning Paradigms:**
    *   **Supervised Learning:** Learning from paired inputs and desired outputs (e.g., classification).
    *   **Unsupervised Learning:** Finding hidden structures in unlabeled data (e.g., clustering).
    *   **Reinforcement Learning:** Learning through trial and error based on feedback from an environment.

---
*Source: Summarized from Wikipedia (Artificial Neural Network)*
