# 3D Gaussian Splatting (3DGS) Improvements

This document explores research ideas for advancing 3D Gaussian Splatting, specifically focusing on semantics, articulation, and compression.

---

## 3.1 Semantic 3D Gaussians for Panoptic Scene Understanding

**Problem:** 3DGS Gaussians carry only RGB color (via spherical harmonics). There is no semantic label per Gaussian, so you cannot ask "which Gaussians are the chair?"

**Idea:** Assign each Gaussian a **semantic feature vector** $\mathbf{s}_i \in \mathbb{R}^D$ (in addition to its color). Render semantic feature maps with the same alpha-compositing and supervise with 2D segmentation labels (from SAM or DINO):

$$
\hat{\mathbf{S}}(\text{pixel}) = \sum_{i} \mathbf{s}_i \alpha_i \prod_{j < i}(1 - \alpha_j)
$$

This enables **3D panoptic segmentation** by simply clustering Gaussians in feature space.

**Why it could work:** Feature Splatting ([arXiv:2404.01528](https://arxiv.org/abs/2404.01528)) demonstrates this direction. An extension is to use **open-vocabulary features** (CLIP or DINO) so you can query objects in natural language in real-time 3D.

---

## 3.2 Structured Gaussian Primitives for Articulated Objects

**Problem:** 3DGS cannot naturally represent **articulated objects** (humans, robotic arms) because its Gaussians are assumed to be independent — there is no notion of kinematic structure.

**Idea:** Organize 3D Gaussians into a **part hierarchy** where each rigid body part owns a set of Gaussians. Parameterize deformations via SE(3) transforms on each part:

$$
\mu_i(t) = R_k(t)\, \mu_i^{canonical} + T_k(t), \quad i \in \text{part}_k
$$

Learn the pose of each part $k$ over time from monocular video using an inverse kinematics constraint (bone length preservation).

**Why it could work:** PhysGaussian shows that coupling Gaussians with physics simulation is feasible. Extending this to learned articulation from video is a natural next step toward real-time avatar animation.

---

## 3.3 Compressed 3DGS via Hierarchical Codebook

**Problem:** A typical 3DGS scene requires **hundreds of millions of Gaussians**, consuming 1–6 GB of memory — impractical for mobile or edge devices.

**Idea:** Apply **vector quantization** (VQ-VAE style) to the Gaussian attribute vectors (position, covariance, color SH). Cluster Gaussians into $K$ prototypes and store only indices + a compact codebook:

```
Full Gaussians: N × (3 + 6 + 3 + 48) = N × 60 floats
Quantized:      N × log₂(K) bits  +  K × 60 floats codebook
```

Use a straight-through estimator to train the codebook end-to-end while optimizing scene reconstruction quality.

**Why it could work:** LightGaussian and Compact3D already show that Gaussian pruning + compression is feasible. Learned VQ codebooks could preserve semantically coherent groups of Gaussians better than hand-designed pruning.
