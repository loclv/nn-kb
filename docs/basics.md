# Artificial Neural Network Basics

Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. This document covers the fundamental building blocks, learning algorithms, regularization techniques, and training strategies that underpin modern deep learning.

## 1. Introduction

An **Artificial Neural Network (ANN)** consists of interconnected units called **neurons** (or nodes), loosely modeling the neurons in a biological brain. Each neuron receives a weighted sum of inputs, applies a non-linear **activation function**, and passes the result to subsequent neurons via weighted **edges**.

The modern deep learning era is generally traced to the work of Rumelhart, Hinton & Williams (1986) who popularized **backpropagation**, and LeCun et al. (1989) who applied it to convolutional networks. Since then, hardware advances (GPUs) and large datasets have enabled networks of extraordinary scale.

## 2. Architecture

### 2.1 Layer Types

| Layer | Role |
|---|---|
| **Input Layer** | Receives raw data (pixels, tokens, sensor values). |
| **Hidden Layers** | Extract increasingly abstract features. More layers = deeper network. |
| **Output Layer** | Produces predictions (class probabilities, regression values, etc.). |

Networks with many hidden layers are called **deep neural networks (DNNs)**. Depth is critical — Bengio et al. (2009) demonstrated that deep architectures can represent certain functions exponentially more efficiently than shallow ones.

### 2.2 Connectivity Patterns

- **Fully Connected (Dense):** Every neuron in layer $l$ connects to every neuron in layer $l+1$. Rich expressiveness but high parameter count.
- **Locally Connected (CNN-style):** Neurons connect only to a local spatial region. See [CNN docs](cnn.md).
- **Recurrent (RNN-style):** Edges loop back in time, giving the network a hidden "memory" state. See [RNN docs](rnn.md).
- **Sparse / Attention-based:** Connections are dynamically weighted. Used in [Transformers](transformer.md).

## 3. Activation Functions

Activation functions introduce **non-linearity**, enabling networks to learn complex mappings. Without them, stacking layers would be equivalent to a single linear transformation.

| Function | Formula | Notes |
|---|---|---|
| **Sigmoid** | $\sigma(x) = 1/(1+e^{-x})$ | Outputs in (0,1); suffers from vanishing gradients. |
| **Tanh** | $\tanh(x)$ | Zero-centered; still vanishes for large inputs. |
| **ReLU** | $\max(0, x)$ | Fast, sparse activations; "dying ReLU" problem. Introduced broadly by Glorot et al. (2011). |
| **Leaky ReLU** | $\max(\alpha x, x)$ | Fixes dying ReLU by allowing small negative slope $\alpha$. |
| **GELU** | $x \cdot \Phi(x)$ | Gaussian Error Linear Unit; used in BERT and GPT. Smoother than ReLU. |
| **SwiGLU** | $\text{Swish}_1(xW + b) \otimes (xV + c)$ | Used in Llama and PaLM; combines Gated Linear Unit with Swish. (Shazeer, 2020). |
| **Softmax** | $e^{x_i}/\sum e^{x_j}$ | Normalizes outputs to a probability distribution; used in the output layer for classification. |

## 4. Loss Functions

The **loss function** $\mathcal{L}$ measures the gap between the model's prediction $\hat{y}$ and the ground truth $y$. Training minimizes $\mathcal{L}$.

| Task | Loss Function | Formula |
|---|---|---|
| Binary Classification | Binary Cross-Entropy | $-[y \log \hat{y} + (1-y)\log(1-\hat{y})]$ |
| Multi-class Classification | Categorical Cross-Entropy | $-\sum_c y_c \log \hat{y}_c$ |
| Regression | Mean Squared Error (MSE) | $\frac{1}{n}\sum (y - \hat{y})^2$ |
| Regression (robust) | Mean Absolute Error (MAE) | $\frac{1}{n}\sum |y - \hat{y}|$ |

## 5. Backpropagation & Optimization

### 5.1 Backpropagation

**Backpropagation** (Rumelhart et al., 1986) is an efficient algorithm for computing the gradient of the loss $\mathcal{L}$ with respect to every weight in the network using the chain rule of calculus. Given the gradient, weights are updated by **gradient descent**:

$$
w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

where $\eta$ is the **learning rate**.

### 5.2 Optimizers

| Optimizer | Key Idea | Reference |
|---|---|---|
| **SGD** | Updates weights using a mini-batch gradient. Simple but slow to converge. | — |
| **SGD + Momentum** | Accumulates a velocity vector to dampen oscillations and accelerate convergence. | Polyak, 1964 |
| **RMSprop** | Divides gradient by an exponential moving average of squared gradients; adapts learning rate per parameter. | Hinton, 2012 |
| **Adam** | Combines momentum (first moment) and RMSprop (second moment); highly popular default optimizer. | Kingma & Ba, 2014 ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) |
| **AdamW** | Adam with decoupled weight decay; often better for Transformers. | Loshchilov & Hutter, 2019 |

**Adam** is defined by:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}
$$
$$
w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### 5.3 Learning Rate Schedulers

| Scheduler | Key Idea |
|---|---|
| **Step Decay** | Drops learning rate by a factor every $N$ epochs. |
| **Cosine Annealing** | Decays $\eta$ following a cosine curve; often used with **warmup** (starting small and increasing). |
| **Cyclic LR** | Oscillates $\eta$ between boundaries; helps escape saddle points. |

## 6. Regularization

Regularization techniques prevent **overfitting** — memorizing training data rather than generalizing.

### 6.1 Dropout

Introduced by Hinton et al. (2012) ([arXiv:1207.0580](https://arxiv.org/abs/1207.0580)), **Dropout** randomly "drops" (sets to zero) a fraction $p$ of neurons during each training step. This prevents neurons from co-adapting and acts as an ensemble of exponentially many networks. At inference, all neurons are active but weights are scaled by $(1-p)$.

### 6.2 Batch Normalization

Introduced by Ioffe & Szegedy (2015) ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167)), **Batch Normalization (BN)** normalizes the inputs of each layer to have zero mean and unit variance across the training mini-batch. Key benefits:
- Allows much **higher learning rates**.
- Reduces sensitivity to **weight initialization**.
- Acts as a **regularizer**, often reducing the need for Dropout.
- Dramatically accelerates training (same accuracy in 14× fewer steps on ImageNet).

### 6.3 RMSNorm

Introduced by Zhang & Sennrich (2019) ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467)), **Root Mean Square Layer Normalization** is a computationally efficient variant of LayerNorm that only rescales inputs without re-centering them (removing the mean-subtraction step):

$$
\bar{a}_i = \frac{a_i}{\sqrt{\frac{1}{n}\sum a_j^2 + \epsilon}} \cdot \gamma_i
$$

RMSNorm is the standard normalization for modern LLMs like **Llama 3**, Gopher, and Chinchilla, offering similar performance to LayerNorm with 10–40% faster computation.

### 6.4 L1 / L2 Weight Regularization

- **L2 (Weight Decay):** Penalizes large weights by adding $\lambda \|w\|^2$ to the loss. Encourages small, diffuse weights.
- **L1 (Lasso):** Adds $\lambda \|w\|_1$. Encourages **sparsity** — drives many weights to exactly zero.

## 7. Learning Paradigms

| Paradigm | Description | Example |
|---|---|---|
| **Supervised Learning** | Learns from labeled (input, target) pairs. | Image classification, language translation. |
| **Unsupervised Learning** | Discovers structure in unlabeled data. | Clustering, autoencoders, GANs. |
| **Self-Supervised Learning** | Generates supervision signal from the data itself. | BERT's masked language modeling (Devlin et al., 2018). |
| **Reinforcement Learning** | Learns from rewards via trial-and-error. | AlphaGo, robotic control. |
| **Transfer Learning** | Pre-trains on large data; fine-tunes on a target task. | Fine-tuning GPT or BERT for downstream NLP tasks. |

## 8. Key Milestones

| Year | Paper | Contribution |
|---|---|---|
| 1986 | Rumelhart et al. | Popularized backpropagation |
| 1989 | LeCun et al. | Convolutional networks for handwriting recognition |
| 1997 | Hochreiter & Schmidhuber | Long Short-Term Memory (LSTM) |
| 2012 | Hinton et al. | Dropout regularization |
| 2014 | Kingma & Ba | Adam optimizer ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) |
| 2015 | Ioffe & Szegedy | Batch Normalization ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167)) |
| 2017 | Vaswani et al. | Transformer / Attention Is All You Need ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) |
| 2018 | Devlin et al. | BERT ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) |
| 2020 | Brown et al. | GPT-3 (175B parameters) |
| 2022 | Hoffmann et al. | Chinchilla / Scaling Laws ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) |
| 2024 | Meta AI | Llama 3 (Open weights state-of-the-art) |

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*, 533–536.
- Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. *AISTATS*.
- Hinton, G. E., et al. (2012). Improving neural networks by preventing co-adaptation of feature detectors. [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Bengio, Y., et al. (2009). Learning deep architectures for AI. *Foundations and Trends in Machine Learning*.
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Shazeer, N. (2020). GLU Variants Improve Transformer. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts (Cosine Annealing). [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
