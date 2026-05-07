# Self-Attention Mechanism

Self-attention is the core innovation of the Transformer architecture (Vaswani et al., 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)). It allows each position in a sequence to directly relate to every other position with a constant number of operations, enabling rich contextual representations without any recurrence or convolution.

> See also: [Transformer Architectures](transformer.md) for the full encoder-decoder architecture, BERT, and GPT-3.

---

## 1. Intuition: What is Attention?

Consider the sentence: *"The animal didn't cross the street because **it** was too tired."*

A human reader immediately resolves *"it"* to *"The animal"* (not *"the street"*) through context. Self-attention gives the model the same ability: when processing *"it"*, the model can directly attend to *"animal"* with a high weight, giving it the contextual meaning it needs.

---

## 2. From Token Embeddings to Q, K, V

Given a sequence of $n$ tokens, each represented as a $d_{model}$-dimensional embedding vector, self-attention begins by projecting each token's embedding into three distinct vector spaces using learned weight matrices:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

where:
- $X \in \mathbb{R}^{n \times d_{model}}$ — the input embedding matrix (one row per token).
- $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$ — learned projection matrices for queries and keys.
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$ — learned projection matrix for values.

**Conceptual roles:**

| Vector | Analogy | Meaning |
|---|---|---|
| **Query (Q)** | A search query | *"What am I looking for?"* — represents the current token's need. |
| **Key (K)** | A document index tag | *"What do I contain?"* — represents each token's "label" for retrieval. |
| **Value (V)** | Document content | *"What do I actually communicate?"* — the information to aggregate. |

---

## 3. Step-by-Step Computation

**Step 1 — Compute raw attention scores.**

Take the dot product of each query against every key. This produces an $n \times n$ score matrix $S$:

$$
S = QK^T \quad (S_{ij} \text{ = similarity of token } i \text{ to token } j)
$$

A high score $S_{ij}$ means token $i$ "cares about" token $j$ a lot.

**Step 2 — Scale.**

Divide by $\sqrt{d_k}$ to prevent extremely large dot products that push softmax into vanishing-gradient regions:

$$
S' = \frac{QK^T}{\sqrt{d_k}}
$$

*Why $\sqrt{d_k}$?* If Q and K are random vectors with zero mean and unit variance, their dot product has variance $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance.

**Step 3 — (Optional) Apply masking.**

For autoregressive (decoder) settings, apply a causal mask before softmax to prevent token $i$ from attending to future token $j > i$:

$$
S'_{ij} = \begin{cases} S'_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

After softmax, $-\infty$ becomes 0, ensuring future tokens receive zero attention weight.

**Step 4 — Softmax to get attention weights.**

Apply softmax row-wise to turn raw scores into a probability distribution:

$$
A = \text{softmax}(S') \quad (A \in \mathbb{R}^{n \times n}, \; \sum_j A_{ij} = 1)
$$

$A_{ij}$ is now the fraction of attention token $i$ allocates to token $j$.

**Step 5 — Weighted sum of values.**

Multiply the attention weights by the value matrix to get the self-attention output:

$$
\text{Attention}(Q, K, V) = A \cdot V \quad (\in \mathbb{R}^{n \times d_v})
$$

Each output vector is a **context-sensitive blend** of all value vectors, weighted by how relevant each token is.

**Full formula:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

---

## 4. Worked Numerical Example

For a 3-token sequence ("cat", "sat", "mat") with $d_k = 2$, suppose after projection:

```
Q = [[1, 0],    K = [[1, 0],    V = [[0.1, 0.9],
     [0, 1],         [0, 1],         [0.8, 0.2],
     [1, 1]]         [1, 1]]         [0.5, 0.5]]
```

**Score matrix** $QK^T / \sqrt{2}$:

```
         cat    sat    mat
cat   [[ 0.71, 0.00, 0.71],
sat    [ 0.00, 0.71, 0.71],
mat    [ 0.71, 0.71, 1.41]]
```

"mat" attends to itself most strongly (score 1.41) and equally to "cat" and "sat".

After softmax, "cat" row: `[0.50, 0.18, 0.32]` — "cat" pays most attention to itself, some to "mat", little to "sat".

---

## 5. Multi-Head Attention

A single attention head can only attend to one "type" of relationship at a time. **Multi-head attention** runs $h$ independent attention operations in parallel, each with its own $W^Q_i, W^K_i, W^V_i$:

$$
\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

The original paper uses $h = 8$ heads with $d_k = d_v = d_{model}/h = 64$. Each head can specialize in:
- **Syntactic relationships:** subject–verb agreement.
- **Coreference:** pronoun resolution ("it" → "animal").
- **Positional patterns:** attending to the previous or next token.
- **Semantic similarity:** attending to synonyms.

The outputs are concatenated ($\mathbb{R}^{n \times (h \cdot d_v)}$) and projected back to $d_{model}$ via $W^O$.

---

## 6. Self-Attention vs Cross-Attention

| Type | Q source | K, V source | Use |
|---|---|---|---|
| **Encoder Self-Attention** | Encoder input | Encoder input | Every token in the input attends to every other input token (bidirectional). |
| **Decoder Masked Self-Attention** | Decoder input | Decoder input (past only) | Each generated token attends only to previously generated tokens (causal mask). |
| **Decoder Cross-Attention** | Decoder state | Encoder output | The decoder attends to the full encoded source — this is the seq2seq "read" step. |

---

## 7. Computational Complexity

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(n^2)$ |
| RNN (per step) | $O(n \cdot d^2)$ | $O(d)$ |
| CNN (k-width) | $O(k \cdot n \cdot d^2)$ | $O(k \cdot n \cdot d)$ |

The $O(n^2)$ memory cost of storing the full attention matrix $A$ is the key bottleneck for long sequences. This has driven research into efficient attention variants (see §8).

---

## 8. Positional Encodings

Self-attention is **permutation-invariant** — if you shuffle the tokens, the attention weights shuffle correspondingly but the mechanism itself has no sense of order. Positional encodings inject position information.

### 8.1 Sinusoidal (Vaswani et al., 2017)

The original paper adds a fixed sinusoidal signal to each token embedding:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

Different dimensions oscillate at different frequencies, encoding both absolute position and relative distances via trigonometric identities: $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$.

### 8.2 Rotary Position Embedding (RoPE)

**Authors:** Su et al. (2021) — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

> *"RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation."*

RoPE rotates the Q and K vectors by a position-dependent angle before computing the dot product:

$$
\text{score}(q_m, k_n) = \langle R_m q, R_n k \rangle
$$

where $R_m$ is a rotation matrix for position $m$. The attention score naturally depends only on the **relative distance** $(m - n)$. RoPE is used in LLaMA, Mistral, Gemma, and most modern open-weight LLMs. Key advantages:
- **Length extrapolation:** Models can generalize to sequences longer than those seen during training.
- **Decaying bias:** Attention scores decrease as token distance increases (matches linguistic intuition).
- Compatible with **linear attention** extensions.

### 8.3 ALiBi (Attention with Linear Biases)

**Authors:** Press et al. (2021) — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

Rather than adding positional encodings to embeddings, ALiBi adds a fixed negative bias proportional to the distance between tokens:

$$
S'_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m \cdot (i - j)
$$

where $m$ is a head-specific slope. This penalizes attention to distant tokens without any positional embeddings, and generalizes well beyond training context length.

---

## 9. Efficient Attention Variants

Standard attention requires $O(n^2)$ memory, making it infeasible for very long contexts. Several approaches address this:

### 9.1 FlashAttention

**Authors:** Dao, Fu, Ermon, Rudra, Ré (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

> *"A missing principle is making attention algorithms IO-aware — accounting for reads and writes between levels of GPU memory."*

FlashAttention computes **exact** attention (no approximation) with $O(n)$ memory by **tiling** the computation to avoid materializing the full $n \times n$ attention matrix in high-bandwidth memory (HBM). Instead it keeps sub-tiles in fast on-chip SRAM.

**Results:**
- **15%** faster than MLPerf 1.1 record on BERT-large.
- **3×** speedup on GPT-2 (seq. length 1K).
- **2.4×** speedup on long-range arena (1K–4K tokens).
- Enables context lengths of 16K–64K tokens.

FlashAttention v2 and v3 are the de-facto attention implementation in PyTorch, HuggingFace Transformers, and most LLM inference engines.

### 9.2 Sparse Transformer

**Authors:** Child et al. (2019) — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)

Instead of every token attending to every other token, the Sparse Transformer factorizes the attention matrix so each token attends to only $O(\sqrt{n})$ other tokens, reducing complexity to $O(n\sqrt{n})$:

- **Strided attention:** Token $i$ attends to tokens $\{i-1, i-2, \ldots, i-l\}$ (local window) and $\{i, i-l, i-2l, \ldots\}$ (strided pattern).
- Two heads with different patterns together can attend to all positions in 2 hops.

**Results:** Modeled sequences of 16,000+ timesteps, set new state-of-the-art on Enwik8 and ImageNet-64.

### 9.3 Other Efficient Variants

| Method | Complexity | Key Idea |
|---|---|---|
| **Longformer** | $O(n)$ | Sliding window + global tokens. |
| **BigBird** | $O(n)$ | Combines random, window, and global attention. |
| **Performer** | $O(n \cdot d)$ | Approximates softmax attention with random feature maps (FAVOR+). |
| **Linear Attention** | $O(n \cdot d^2)$ | Kernel trick to avoid materializing $n \times n$ matrix. |
| **Mamba (SSM)** | $O(n)$ | State space models as a recurrent alternative to attention. |

---

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Child, R., et al. (2019). Generating Long Sequences with Sparse Transformers. [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press, O., Smith, N. A., & Lewis, M. (2021). Train Short, Test Long: Attention with Linear Biases (ALiBi). [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
