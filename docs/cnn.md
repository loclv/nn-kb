# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs or ConvNets) are the dominant architecture for processing grid-structured data — primarily images and video. Their design is inspired by the organization of the animal visual cortex and exploits spatial locality, shared weights, and hierarchical feature extraction.

## 1. Core Concepts

### 1.1 Motivation

Standard fully connected networks do not scale to high-resolution images. A single 224×224 RGB image has 150,528 input values; a fully connected first layer with 4,096 neurons would require ~600 million parameters. CNNs address this through:

- **Local Connectivity:** Each neuron attends only to a small **receptive field** in the input, not every pixel.
- **Parameter Sharing:** The same filter (kernel) is applied across all spatial positions — this is the **convolution** operation.
- **Hierarchical Representations:** Early layers detect low-level features (edges, colors); deeper layers detect complex patterns (faces, objects).

### 1.2 The Convolution Operation

A **convolution** slides a learnable filter $\mathbf{w}$ of size $k \times k$ across the input feature map $\mathbf{x}$, computing the dot product at each position:

$$
(\mathbf{x} * \mathbf{w})[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \mathbf{x}[i+m, j+n] \cdot \mathbf{w}[m, n]
$$

The result is a **feature map** (or activation map) that indicates where in the image the filter's learned pattern appears.

Key parameters:
- **Stride ($s$):** How many pixels the filter shifts per step. Stride > 1 down-samples the feature map.
- **Padding:** Adding zeros around the border to control the output size. "Same" padding preserves spatial dimensions; "valid" padding does not.

## 2. Layer Types

### 2.1 Convolutional Layer

The core building block. A layer typically applies $C_{out}$ different filters, producing $C_{out}$ feature maps. Each filter learns to detect a different feature (e.g., horizontal edges, curves).

### 2.2 Activation (ReLU)

After every convolution, a **ReLU** activation ($\max(0, x)$) is applied element-wise to introduce non-linearity and create sparse representations.

### 2.3 Pooling Layer

Reduces spatial resolution and provides local translation invariance:

| Type | Operation | Effect |
|---|---|---|
| **Max Pooling** | Takes the maximum in each region (e.g., 2×2) | Retains strongest feature activations |
| **Average Pooling** | Averages values in each region | Smoother, sometimes used in classification head |
| **Global Avg Pooling** | Averages each entire feature map to a single value | Replaces FC layers; used in ResNet, MobileNet |

### 2.4 Fully Connected (FC) / Dense Layer

Placed at the end of the network to perform final classification. Takes the flattened feature maps and maps to class logits, which are then passed through **Softmax** for probabilities.

### 2.5 Batch Normalization

Ioffe & Szegedy (2015) ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167)) introduced **Batch Normalization (BN)**, which normalizes activations within each mini-batch. BN is typically inserted after each convolutional layer and before the activation function. It:
- Stabilizes and accelerates training.
- Allows higher learning rates.
- Acts as a regularizer, often replacing Dropout in CNNs.

### 2.6 Dropout

Randomly zeros activations during training to prevent co-adaptation (Hinton et al., 2012 — [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)). Typically applied after FC layers.

## 3. Landmark Architectures

### 3.1 LeNet-5 (1998)

**Authors:** LeCun, Bottou, Bengio, Haffner.

The first successful CNN deployed in production (for handwritten digit recognition on MNIST). Introduced the fundamental `Conv → Pool → Conv → Pool → FC` pattern that still underpins modern architectures.

### 3.2 AlexNet (2012)

**Authors:** Krizhevsky, Sutskever, Hinton.

Won the ImageNet LSVRC-2012 competition with a top-5 error of 15.3% — over 10 percentage points better than the runner-up. Key innovations:
- First use of **ReLU** activations in a large-scale CNN (faster than tanh).
- **Local Response Normalization (LRN)** (a precursor to BN).
- **Dropout** in FC layers to prevent overfitting.
- Trained on **two GPUs** in parallel.

### 3.3 VGGNet (2014)

**Authors:** Simonyan & Zisserman — [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

> *"We investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting."*

Demonstrated that **network depth** is a critical component of performance. Used a very simple homogeneous architecture of **3×3 convolution filters** stacked to depths of 16–19 layers. Achieved 1st place in localization and 2nd in classification at ILSVRC 2014. The VGG-16 and VGG-19 models remain widely used as feature extractors.

### 3.4 ResNet (2015)

**Authors:** He, Zhang, Ren, Sun — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

> *"Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."*

ResNet introduced **skip connections** (residual connections) that allow gradients to flow directly through many layers, solving the **vanishing gradient problem** for very deep networks. A **residual block** computes:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

where $\mathcal{F}$ is the residual mapping to be learned. By making the layers learn a residual $\mathcal{F}(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$ instead of the full mapping $H(\mathbf{x})$, training becomes dramatically easier.

**Key results:**
- Trained networks with **152 layers** — 8× deeper than VGG but with lower complexity.
- Achieved **3.57% top-5 error** on ImageNet — better than human-level performance (~5%).
- Won 1st place in ILSVRC & COCO 2015 across classification, detection, and segmentation.

### 3.5 Modern Efficient Architectures

| Architecture | Key Idea | Year |
|---|---|---|
| **GoogLeNet / Inception** | Parallel filters of different sizes (Inception module) | 2014 |
| **MobileNet** | Depthwise separable convolutions for mobile devices | 2017 |
| **EfficientNet** | Compound scaling of depth, width, and resolution | 2019 |
| **Vision Transformer (ViT)** | Applies Transformer attention to image patches | 2020 |

## 4. Applications

- **Computer Vision:** Image classification, object detection (YOLO, Faster R-CNN), semantic segmentation (U-Net, DeepLab).
- **Medical Imaging:** Tumor detection, radiology report generation.
- **Video Understanding:** Action recognition, optical flow estimation.
- **Natural Language Processing:** 1D CNNs for text classification and sequence modeling.
- **Audio:** Speech recognition (using spectrograms as 2D inputs).

---

## References

- LeCun, Y., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Hinton, G. E., et al. (2012). Improving neural networks by preventing co-adaptation of feature detectors. [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
