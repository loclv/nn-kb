# Research Ideas: Improving 3D World Understanding & Object Learning

This document collects open problems, unsolved challenges, and original research directions in learning from the 3D world. Each idea is grounded in the limitations of current approaches documented in [3d-learning.md](../3d-learning.md).

> **How to read this file:** This is a central index for various research areas. Each topic contains structured ideas (Problem, Idea, Why it could work, Open questions).

---

## Research Areas

- [**1. Point Cloud Learning**](point-cloud.md)
  - Dynamic Neighborhood Attention
  - Masked Autoencoding
  - Physics-Informed Descriptors
- [**2. Neural Radiance Fields (NeRF)**](nerf.md)
  - Generalizable NeRF
  - Dynamic NeRF (Temporal Basis)
  - Semantic-Aware NeRF
- [**3. 3D Gaussian Splatting (3DGS)**](gaussian-splatting.md)
  - Semantic 3D Gaussians
  - Structured Gaussian Primitives
  - Compressed 3DGS
- [**4. 3D Object Detection & Scene Understanding**](object-detection.md)
  - Language-Grounded (Open-Vocabulary) Detection
  - Temporal Object Memory (SSM)
  - Occupancy Prediction from Sparse Observations
- [**5. Cross-Modal and Foundation Models**](cross-modal.md)
  - 3D Foundation Models
  - LLM-Guided 3D Scene Editing

---

## 6. Priority Matrix

| Idea | Impact | Feasibility | Novelty |
|---|---|---|---|
| Dynamic neighborhood radii (§1.1) | Medium | High | Medium |
| Masked point cloud pre-training (§1.2) | High | High | Low* |
| Semantic 3D Gaussians (§3.1) | High | High | Low* |
| Generalizable NeRF via cross-attention (§2.1) | High | Medium | Medium |
| Language-grounded 3D detection (§4.1) | Very High | Medium | High |
| SSM temporal memory for detection (§4.2) | High | Medium | High |
| Occupancy self-supervised completion (§4.3) | High | Medium | High |
| 3D Foundation Model (§5.1) | Very High | Low | High |
| LLM-guided 3D scene editing (§5.2) | Very High | Medium | Medium |

> *Low novelty means the idea is already being actively explored — good for implementation, less so for a novel research contribution.

---

## 7. Recommended Starting Points

For a new researcher entering the field:

1. **Read first:** PointNet → PointNet++ → PCT → NeRF → 3DGS (in this order).
2. **Implement:** PointNet from scratch on ModelNet40. Then try Point-MAE pre-training.
3. **Experiment:** Start with **semantic 3D Gaussians** (§3.1) — it has clear evaluation protocol (3D segmentation mIoU), strong baselines (LERF, Feature Splatting), and is highly impactful for downstream tasks.
4. **Stretch goal:** SSM temporal memory for BEVFusion (§4.2) — novel, high-impact, and tractable with existing codebases.
