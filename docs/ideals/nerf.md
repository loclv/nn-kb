# Neural Radiance Fields (NeRF)

This document collects research ideas for improving Neural Radiance Fields, focusing on generalization, dynamics, and semantic integration.

---

## 2.1 Generalizable NeRF via Cross-Scene Attention

**Problem:** Standard NeRF must be **re-optimized for every new scene**, taking 1–10 hours. This prevents deployment in robotics, AR/VR, or any application requiring rapid scene understanding.

**Idea:** Train a **generalizable NeRF** (e.g., building on PixelNeRF or IBRNet) using **cross-scene attention**: the model receives image features from 2–5 context views and uses Transformer cross-attention to answer arbitrary 5D radiance queries without per-scene optimization:

```
Query: (x, y, z, θ, φ)
Context: {(I_1, P_1), ..., (I_k, P_k)}   [images + camera matrices]
  ↓ Project query 3D point onto each image → image features
  ↓ Cross-attention over k image features
  ↓ Predict (c, σ) in one forward pass
```

**Why it could work:** IBRNet and MVSNeRF demonstrate that multi-view aggregation allows zero-shot generalization. The bottleneck is expressiveness — replacing their mean-pooling aggregator with learned attention should capture view-dependent occlusion better.

**Open questions:**
- Can the context encoder and query decoder be jointly pre-trained on a large dataset (e.g., Objaverse)?
- How does performance scale with the number of context views?

---

## 2.2 Dynamic NeRF with Learned Temporal Basis

**Problem:** Standard NeRF assumes a **static scene**. Dynamic scenes (humans moving, cars driving) are modeled by naive time-conditioned NeRFs but this struggles with fast motion.

**Idea:** Represent scene dynamics as a **low-dimensional temporal basis**. Factor the 4D scene into:
- A **canonical static NeRF** $F_\Theta(x, y, z)$.
- A **deformation field** $D_\Phi(x, y, z, t)$ that maps each frame's coordinates back to canonical space.
- Model $D_\Phi$ as a small MLP that predicts SE(3) rigid transforms per region, learned from a **temporal attention** over $k$ learned basis deformations.

**Why it could work:** Consistent with HyperNeRF and Nerfies, which showed that deformation fields + canonical NeRF outperform naive time-conditioning. Temporal basis factorization reduces the degrees of freedom and could work with very few frames.

---

## 2.3 Semantic-Aware NeRF: Fusing Language and Radiance

**Problem:** NeRF represents appearance (color) and geometry (density) but has no semantic understanding. Queries like "render only the car" or "segment the table" are impossible with vanilla NeRF.

**Idea:** Distill **CLIP or DINO features** ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)) into a NeRF by adding a semantic feature head $F_\Theta(x,y,z) \to \mathbf{f}_{sem}$ alongside the color/density heads. Render semantic feature maps using the same volume rendering equation, and supervise with features computed from 2D training views:

$$
\hat{\mathbf{f}}(\mathbf{r}) = \int T(t)\, \sigma\, \mathbf{f}_{sem}(t)\, dt
\quad \approx \quad \text{CLIP}(\text{patch at pixel})
$$

At inference, use the rendered feature map with text or image queries to perform zero-shot 3D segmentation.

**Why it could work:** LERF ([arXiv:2303.09553](https://arxiv.org/abs/2303.09553)) already shows this works well. An extension is to distill not just CLIP but also **depth features** (from Depth Anything) to improve geometry consistency.
