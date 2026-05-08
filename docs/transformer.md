# Transformer Architectures

The Transformer is a deep learning architecture introduced by Vaswani et al. (2017) that relies **entirely on attention mechanisms**, discarding recurrence and convolution. It is the backbone of modern Large Language Models (LLMs) including BERT, GPT-3, and beyond.

## 1. Motivation & Origin

**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017) â€” [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

> *"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."*

**Key improvements over RNNs:**

| Property | RNN | Transformer |
|---|---|---|
| **Parallelism** | Sequential (timestep by timestep) | Fully parallel across positions |
| **Long-range dependencies** | Struggles (vanishing gradients) | Direct attention between any two positions |
| **Training speed** | Slow on long sequences | Much faster on modern hardware |
| **Interpretability** | Opaque hidden state | Attention weights are visualizable |

**Results on WMT 2014:**
- English-to-German: **28.4 BLEU** â€” over 2 BLEU improvement over the best prior result.
- English-to-French: **41.8 BLEU** â€” new single-model state-of-the-art after only 3.5 days on 8 GPUs.

## 2. Self-Attention

Self-attention is the core mechanism that allows each token to directly attend to every other token in the sequence. This enables the Transformer to capture long-range dependencies in a single layer, unlike RNNs which require many recurrent steps.

The full deep-dive â€” including step-by-step computation, a numerical worked example, multi-head attention, positional encodings (sinusoidal, RoPE, ALiBi), and efficient variants (FlashAttention, Sparse Transformer) â€” is documented in **[Self-Attention Mechanism](self-attention.md)**.

### 2.1 The Core Formula

Given query ($Q$), key ($K$), and value ($V$) matrices projected from the input:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- The dot product $QK^T$ scores how much each token should attend to every other.
- Scaling by $\sqrt{d_k}$ prevents softmax saturation.
- The output is a context-sensitive weighted sum of the value vectors.

### 2.2 Multi-Head Attention

The Transformer runs $h$ parallel attention heads, each projecting with its own learned weights:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

The original paper uses $h = 8$ heads with $d_k = d_v = 64$ ($d_{model} = 512$). Each head can specialize in different types of relationships (syntax, coreference, position, semantics).

### 2.3 Attention Types

| Type | Q/K/V Source | Use |
|---|---|---|
| **Encoder Self-Attention** | Encoder input | Bidirectional; every token attends to all others. |
| **Decoder Masked Self-Attention** | Decoder input (past only) | Causal; tokens cannot attend to future positions. |
| **Decoder Cross-Attention** | Q from decoder, K/V from encoder | Decoder "reads" the full encoded source. |

## 3. Architecture

The standard Transformer follows an **Encoder-Decoder** structure:

```
Input Tokens
     â”‚
[Embedding + Positional Encoding]
     â”‚
  Encoder Stack (NÃ— layers)
  â”œâ”€â”€ Multi-Head Self-Attention
  â”œâ”€â”€ Add & Norm (Residual)
  â”œâ”€â”€ Feed-Forward Network
  â””â”€â”€ Add & Norm (Residual)
     â”‚
  Decoder Stack (NÃ— layers)
  â”œâ”€â”€ Masked Multi-Head Self-Attention
  â”œâ”€â”€ Add & Norm
  â”œâ”€â”€ Cross-Attention (Qâ†decoder, KVâ†encoder)
  â”œâ”€â”€ Add & Norm
  â”œâ”€â”€ Feed-Forward Network
  â””â”€â”€ Add & Norm
     â”‚
[Linear + Softmax]
     â”‚
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

The inner dimension $d_{ff}$ is typically 4Ã— the model dimension $d_{model}$.

### 3.3 Residual Connections & Layer Normalization

Each sub-layer uses a **residual connection** (He et al., 2015 â€” [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) followed by **Layer Normalization**:

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

## 4. Landmark Models Built on Transformers

### 4.1 BERT

**Authors:** Devlin, Chang, Lee, Toutanova (2018) â€” [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

> *"BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."*

BERT is an **encoder-only** Transformer pre-trained on two tasks:
1. **Masked Language Modeling (MLM):** 15% of tokens are masked; the model predicts them using bidirectional context.
2. **Next Sentence Prediction (NSP):** Predicts whether two sentences are consecutive.

**Results:** Set new state-of-the-art on 11 NLP benchmarks including GLUE (80.5%), SQuAD v1.1 F1 (93.2%), and SQuAD v2.0 F1 (83.1%).

**Key variant â€” BERT-Large:** 24 layers, 1024 hidden units, 16 attention heads, 340M parameters.

### 4.2 GPT-3

**Authors:** Brown et al. (2020) â€” [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

> *"Scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches."*

GPT-3 is a **decoder-only** autoregressive Transformer trained on next-token prediction. Key facts:
- **175 billion parameters** â€” 10Ã— larger than any previous non-sparse model.
- Demonstrates remarkable **few-shot learning**: given a few examples in the prompt (without gradient updates), it achieves competitive performance on many NLP tasks.
- Architecture: 96 layers, $d_{model} = 12288$, 96 attention heads.

GPT-3 led directly to ChatGPT and modern instruction-following LLMs.

### 4.3 Llama 3 & 3.1

**Authors:** Meta AI (2024) — [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)

Llama 3 is a family of highly capable, open-weights dense Transformers. The **405B** version is the first open-weights model to rival frontier closed models.

- **Architecture:** Standard decoder-only Transformer with several modern improvements:
    - **Grouped-Query Attention (GQA)** for efficient inference.
    - **RoPE** positional embeddings.
    - **RMSNorm** and **SwiGLU** activation.
- **Scaling:** Trained on **15 trillion tokens** (7.5× more than Llama 2).
- **Llama 3.1:** Extended context to **128K tokens** using specialized RoPE scaling and fine-tuning.

### 4.4 Gemini 1.5 / 2.5

**Authors:** Google DeepMind (2023–2025) — [Gemini Technical Reports](https://arxiv.org/abs/2312.11805)

Gemini is a family of natively **multimodal** models (Ultra, Pro, Nano).

- **Natively Multimodal:** Unlike models that use "adapters" to connect vision encoders to LLMs, Gemini was trained from the start across text, image, audio, and video.
- **Extreme Context:** Gemini 1.5 Pro introduced a **1M to 2M token context window**, using Ring Attention and efficient KV cache management.
- **Reasoning:** Gemini 2.5 ("Thinking" models) incorporates reinforcement learning and inference-time search to solve complex math and coding problems.

### 4.5 Mistral & Mixtral (SMoE)

**Authors:** Jiang et al. (2023–2024) — [Mistral 7B](https://arxiv.org/abs/2310.06825), [Mixtral 8x7B](https://arxiv.org/abs/2401.04088)

Mistral introduced high-performance efficiency to the 7B class, while Mixtral popularized **Sparse Mixture-of-Experts (SMoE)** for LLMs.

- **Sliding Window Attention (SWA):** Mistral 7B uses a windowed attention mechanism to handle long sequences with $O(n \cdot \text{window})$ complexity.
- **SMoE:** Mixtral 8x7B has 47B total parameters but only uses **13B active parameters** per token, providing GPT-3.5 level performance with much faster inference.

## 5. Mixture-of-Experts (MoE)

Sparse MoE (Shazeer et al., 2017; Fedus et al., 2021) replaces the dense FFN layer with multiple "expert" networks. A **router** network learns to send each token to the most relevant 1 or 2 experts.

$$
\text{MoE}(x) = \sum_{i=1}^k G(x)_i E_i(x)
$$

- $G(x)$: The gating (router) network.
- $E_i(x)$: The $i$-th expert.
- $k$: Number of active experts (typically 1 or 2).

**Key Advantage:** Allows scaling model capacity (total parameters) without a proportional increase in training or inference compute.

## 6. Decoder-Only vs Encoder-Only vs Encoder-Decoder

| Model | Type | Key Feature |
|---|---|---|
| **GPT-2** | Decoder-only | Zero-shot text generation; "too dangerous to release" |
| **T5** | Encoder-Decoder | Frames every NLP task as text-to-text |
| **RoBERTa** | Encoder-only | Optimized BERT training (longer, larger batches, no NSP) |
| **Vision Transformer (ViT)** | Encoder-only | Applies Transformer to image patches for CV |
| **GPT-4** | Decoder-only | Multimodal; frontier model |
| **LLaMA 3** | Decoder-only | Open-weights frontier model; 405B parameters |
| **Mixtral 8x7B** | SMoE | Efficient sparse architecture; 8 experts |
| **DeepSeek-V3** | SMoE | Frontier efficiency; Multi-head Latent Attention (MLA) |

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
- Child, R., et al. (2019). Generating Long Sequences with Sparse Transformers. [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Jiang, A. Q., et al. (2023). Mistral 7B. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Jiang, A. Q., et al. (2024). Mixtral of Experts. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Meta AI (2024). The Llama 3 Herd of Models. [Technical Report](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- Google DeepMind (2023). Gemini: A Family of Highly Capable Multimodal Models. [arXiv:2312.11805](https://arxiv.org/abs/2312.11805)
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, W., et al. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Press, O., Smith, N. A., & Lewis, M. (2021). Train Short, Test Long: Attention with Linear Biases (ALiBi). [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
