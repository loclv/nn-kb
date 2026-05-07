# Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks designed for **sequential data** — data where the order of elements carries meaning (time series, text, audio). Unlike feedforward networks, RNNs maintain an internal **hidden state** that is updated at every timestep, giving them a form of memory.

## 1. The Vanilla RNN

### 1.1 Formulation

At each timestep $t$, a vanilla RNN takes an input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and produces a new hidden state $\mathbf{h}_t$:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

Weights are **shared across all timesteps**. Training uses **Backpropagation Through Time (BPTT)**, which unrolls the network across timesteps.

### 1.2 The Vanishing Gradient Problem

During BPTT, gradients are multiplied by the weight matrix $\mathbf{W}_{hh}$ at every timestep. If singular values of $\mathbf{W}_{hh} < 1$, gradients **vanish exponentially**; if $> 1$, they **explode**. This makes learning long-range dependencies extremely difficult.

## 2. Long Short-Term Memory (LSTM)

**Authors:** Hochreiter & Schmidhuber (1997).

LSTM solves the vanishing gradient problem by introducing a **cell state** $\mathbf{c}_t$ — a "conveyor belt" that allows gradients to flow without vanishing. An LSTM unit contains four interacting gates:

| Component | Role |
|---|---|
| **Forget Gate** $\mathbf{f}_t$ | Decides what fraction of the old cell state to **erase**. |
| **Input Gate** $\mathbf{i}_t$ | Decides which new values to **write** to the cell state. |
| **Cell Candidate** $\tilde{\mathbf{c}}_t$ | The candidate values to add to the cell state. |
| **Output Gate** $\mathbf{o}_t$ | Controls which part of the cell state is **exposed** as hidden state $\mathbf{h}_t$. |

Cell state update (linear — prevents gradient vanishing):

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

## 3. Gated Recurrent Unit (GRU)

**Authors:** Cho et al. (2014) — [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)

> *"We found GRU to be comparable to LSTM."*

The GRU simplifies the LSTM by merging the cell state and hidden state, and combining the forget and input gates into a single **update gate**:

| Component | Role |
|---|---|
| **Update Gate** $\mathbf{z}_t$ | Balances how much of the **old hidden state** to keep vs. the new candidate. |
| **Reset Gate** $\mathbf{r}_t$ | Decides how much of the **previous hidden state** to use when computing the candidate. |

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

**GRU vs LSTM:** GRU has fewer parameters and trains faster; LSTM is generally preferred for very long sequences.

## 4. Attention Mechanism (Pre-Transformer)

**Authors:** Bahdanau, Cho & Bengio (2014) — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

> *"We propose to allow a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word."*

The classic encoder-decoder RNN compresses the entire input into a single fixed-length context vector — a **bottleneck** that degrades performance on long sequences. Bahdanau attention solves this by computing a **context vector** as a dynamic weighted sum of all encoder hidden states:

$$
\alpha_{ts} = \frac{\exp(e_{ts})}{\sum_{k} \exp(e_{tk})}, \quad \mathbf{c}_t = \sum_{s} \alpha_{ts} \mathbf{h}_s
$$

where $e_{ts} = a(\mathbf{s}_{t-1}, \mathbf{h}_s)$ is a learned **alignment score**. This was the direct precursor to the Transformer's self-attention. See [Transformer docs](transformer.md).

## 5. RNN Architectures

| Pattern | Use Case |
|---|---|
| **One-to-Many** | Image captioning |
| **Many-to-One** | Sentiment analysis |
| **Many-to-Many (encoder-decoder)** | Machine translation |
| **Bidirectional RNN** | Named entity recognition, BERT |
| **Deep (Stacked) RNN** | Speech recognition |

## 6. Applications

- **NLP:** Machine translation, sentiment analysis, text generation.
- **Speech:** Automatic speech recognition (ASR), text-to-speech (TTS).
- **Time Series:** Stock forecasting, weather prediction, anomaly detection.
- **Bioinformatics:** Protein structure prediction, genomic sequence modeling.
- **Music:** Melody generation, polyphonic music modeling.

---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780.
- Cho, K., et al. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
