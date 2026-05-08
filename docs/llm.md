# Large Language Models (LLMs)

Large Language Models are the dominant paradigm in artificial intelligence, scaling the Transformer architecture to billions of parameters and trillions of tokens. This document covers the fundamental principles of scaling, alignment, and optimization that define the modern LLM era.

---

## 1. Scaling Laws & Data Optimality

### 1.1 The Chinchilla Scaling Laws

**Authors:** Hoffmann et al. (2022) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

Before Chinchilla, many LLMs (like GPT-3) were "under-trained"—they had too many parameters for the amount of data they were shown. DeepMind's research established that for every doubling of compute budget, model size ($N$) and training data ($D$) should be scaled in equal proportions.

**Key Findings:**
- A compute-optimal model should have roughly **20 tokens per parameter**.
- **Chinchilla (70B)** outperformed Gopher (280B) despite being 4× smaller, simply because it was trained on 1.4T tokens.
- Modern models (like Llama 3) often deliberately **over-train** (using 50+ tokens per parameter) to maximize inference-time efficiency.

---

## 2. Post-Training & Alignment

Pre-training on the internet produces a powerful "base model" that excels at continuation but is not necessarily helpful or safe. Post-training aligns the model with human intent.

### 2.1 RLHF (Reinforcement Learning from Human Feedback)

**Authors:** Ouyang et al. (2022) — [InstructGPT](https://arxiv.org/abs/2203.02155)

The standard process for creating models like ChatGPT:
1. **SFT (Supervised Fine-Tuning):** Train the base model on high-quality (prompt, response) pairs written by humans.
2. **Reward Modeling:** Humans rank multiple model outputs; a separate "Reward Model" is trained to predict these human preferences.
3. **PPO (Proximal Policy Optimization):** The model is fine-tuned via RL to maximize the score from the Reward Model while staying close to the SFT model (KL penalty).

### 2.2 DPO (Direct Preference Optimization)

**Authors:** Rafailov et al. (2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

DPO simplifies alignment by removing the need for a separate Reward Model and complex RL (PPO). It directly optimizes the policy using a simple cross-entropy loss on preference pairs (chosen vs. rejected). DPO is now the preferred method for many open-source models due to its stability and simplicity.

---

## 3. Reasoning & Inference-Time Compute

### 3.1 Chain-of-Thought (CoT)

**Authors:** Wei et al. (2022) — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

Simply asking a model to *"Think step by step"* triggers Chain-of-Thought reasoning, where the model generates intermediate rationales before producing a final answer. This dramatically improves performance on multi-step tasks like math and symbolic reasoning.

### 3.2 Reinforcement Learning for Reasoning (DeepSeek-R1 style)

Modern reasoning models (like DeepSeek-R1 or OpenAI o1) use large-scale RL (often without initial SFT) to encourage the model to explore and self-correct during generation.
- **Verification:** The model is rewarded for reaching the correct answer (e.g., in math or code) rather than just mimicking human thought patterns.
- **Inference Scaling:** These models use more compute at *test time* (thinking longer) to solve harder problems, representing a new scaling law for inference.

---

## 4. KV Cache & Inference Optimization

### 4.1 The KV Cache

In autoregressive generation, each new token requires attending to all previous tokens. To avoid redundant computation, we store the Key and Value vectors for all past tokens in the **KV Cache**.
- **The Problem:** The KV cache grows linearly with sequence length and batch size, consuming massive GPU memory.

### 4.2 PagedAttention (vLLM)

**Authors:** Kwon et al. (2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

Inspired by virtual memory in operating systems, **PagedAttention** partitions the KV cache into non-contiguous "pages."
- **Efficiency:** Eliminates memory fragmentation and allows multiple requests to share common KV blocks (e.g., in multi-turn chat).
- **Result:** Up to **24×** higher throughput than standard HuggingFace implementations.

---

## 5. References

- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [Technical Report](https://github.com/deepseek-ai/DeepSeek-R1)
