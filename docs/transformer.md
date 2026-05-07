# Transformer Architectures

The Transformer is a deep learning architecture introduced by Vaswani et al. (2017) that relies **entirely on attention mechanisms**, discarding recurrence and convolution. It is the backbone of modern Large Language Models (LLMs) including BERT, GPT-3, and beyond.

## 1. Motivation & Origin

**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

> *"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."*

**Key improvements over RNNs:**

| Property | RNN | Transformer |
|---|---|---|
| **Parallelism** | Sequential (timestep by timestep) | Fully parallel across positions |
| **Long-range dependencies** | Struggles (vanishing gradients) | Direct attention between any two positions |
| **Training speed** | Slow on long sequences | Much faster on modern hardware |
| **Interpretability** | Opaque hidden state | Attention weights are visualizable |

**Results on WMT 2014:**
- English-to-German: **28.4 BLEU** — over 2 BLEU improvement over the best prior result.
- English-to-French: **41.8 BLEU** — new single-model state-of-the-art after only 3.5 days on 8 GPUs.

## 2. Attention Mechanisms

### 2.1 Scaled Dot-Product Attention

The fundamental operation takes three matrices as input — **Queries (Q)**, **Keys (K)**, and **Values (V)**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- $Q$, $K$, $V$ are linear projections of the input.
- The dot product $QK^T$ gives a score for every (query, key) pair.
- Dividing by $\sqrt{d_k}$ (dimension of keys) prevents vanishing gradients in softmax.
- Softmax converts scores to a probability distribution (attention weights).
- The output is the weighted sum of values.

### 2.2 Multi-Head Attention

Instead of performing a single attention function, the Transformer computes $h$ parallel attention operations ("heads"), each with its own learned projection matrices:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

This allows the model to **jointly attend to information from different representation subspaces** at different positions simultaneously.

### 2.3 Self-Attention vs Cross-Attention

| Type | Q/K/V Source | Use |
|---|---|---|
| **Self-Attention** | All from the same sequence | Each token attends to all other tokens in the same sequence |
| **Cross-Attention** | Q from decoder, K/V from encoder | Decoder attends to the encoder's output |

## 3. Architecture

The standard Transformer follows an **Encoder-Decoder** structure:

```
Input Tokens
     │
[Embedding + Positional Encoding]
     │
  Encoder Stack (N× layers)
  ├── Multi-Head Self-Attention
  ├── Add & Norm (Residual)
  ├── Feed-Forward Network
  └── Add & Norm (Residual)
     │
  Decoder Stack (N× layers)
  ├── Masked Multi-Head Self-Attention
  ├── Add & Norm
  ├── Cross-Attention (Q←decoder, KV←encoder)
  ├── Add & Norm
  ├── Feed-Forward Network
  └── Add & Norm
     │
[Linear + Softmax]
     │
Output Probabilities
```

### 3.1 Positional Encoding

Since the model has no inherent notion of position (unlike RNNs), **positional encodings** are added to the input embeddings. The original paper used sinusoidal functions:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

Modern variants (like RoPE and ALiBi) use learned or relative positional encodings that generalize better to longer contexts.

### 3.2 Feed-Forward Network (FFN)

Each encoder/decoder layer contains a position-wise FFN applied independently to each position:

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

The inner dimension $d_{ff}$ is typically 4× the model dimension $d_{model}$.

### 3.3 Residual Connections & Layer Normalization

Each sub-layer uses a **residual connection** (He et al., 2015 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) followed by **Layer Normalization**:

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

## 4. Landmark Models Built on Transformers

### 4.1 BERT

**Authors:** Devlin, Chang, Lee, Toutanova (2018) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

> *"BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."*

BERT is an **encoder-only** Transformer pre-trained on two tasks:
1. **Masked Language Modeling (MLM):** 15% of tokens are masked; the model predicts them using bidirectional context.
2. **Next Sentence Prediction (NSP):** Predicts whether two sentences are consecutive.

**Results:** Set new state-of-the-art on 11 NLP benchmarks including GLUE (80.5%), SQuAD v1.1 F1 (93.2%), and SQuAD v2.0 F1 (83.1%).

**Key variant — BERT-Large:** 24 layers, 1024 hidden units, 16 attention heads, 340M parameters.

### 4.2 GPT-3

**Authors:** Brown et al. (2020) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

> *"Scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches."*

GPT-3 is a **decoder-only** autoregressive Transformer trained on next-token prediction. Key facts:
- **175 billion parameters** — 10× larger than any previous non-sparse model.
- Demonstrates remarkable **few-shot learning**: given a few examples in the prompt (without gradient updates), it achieves competitive performance on many NLP tasks.
- Architecture: 96 layers, $d_{model} = 12288$, 96 attention heads.

GPT-3 led directly to ChatGPT and modern instruction-following LLMs.

### 4.3 Other Important Variants

| Model | Type | Key Feature |
|---|---|---|
| **GPT-2** | Decoder-only | Zero-shot text generation; "too dangerous to release" |
| **T5** | Encoder-Decoder | Frames every NLP task as text-to-text |
| **RoBERTa** | Encoder-only | Optimized BERT training (longer, larger batches, no NSP) |
| **Vision Transformer (ViT)** | Encoder-only | Applies Transformer to image patches for CV |
| **GPT-4** | Decoder-only | Multimodal; frontier model |
| **LLaMA** | Decoder-only | Open-weights efficient Transformer family |

## 5. Decoder-Only vs Encoder-Only vs Encoder-Decoder

| Architecture | Masking | Primary Use |
|---|---|---|
| **Encoder-only** (BERT) | Bidirectional self-attention | Understanding tasks (classification, NER, QA) |
| **Decoder-only** (GPT) | Causal (masked) self-attention | Generation tasks (text completion, chat) |
| **Encoder-Decoder** (T5, original) | Encoder: bidirectional; Decoder: causal | Seq-to-seq (translation, summarization) |

---

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- He, K., et al. (2015). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
