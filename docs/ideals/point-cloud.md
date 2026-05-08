# Point Cloud Learning

This document explores research ideas for improving point cloud understanding, specifically focusing on dynamic attention, self-supervised pre-training, and physics-informed descriptors.

---

## 1.1 Dynamic Neighborhood Attention with Learnable Radii

**Problem:** PointNet++ uses a fixed ball-query radius per layer. In scenes with varying density (e.g., LiDAR near and far objects), a fixed radius either under-samples sparse regions or over-samples dense ones.

**Idea:** Let the network predict a **per-point, per-layer radius** $r_i$ as a function of local density:

$$
r_i = f_\phi\!\left(\frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \|p_j - p_i\|^2\right)
$$

Use a soft grouping (Gaussian weighting by distance) so gradients flow back through the radius predictor.

**Why it could work:** Density-adaptive pooling is already shown to help in PointNet++ (MSG). Making the radius a learned, input-conditioned parameter takes this further — similar to how deformable convolutions ([arXiv:1703.06211](https://arxiv.org/abs/1703.06211)) replaced fixed-kernel CNNs.

**Open questions:**
- Does learnable radius overfit in small-data regimes?
- Can radii be constrained to maintain computational complexity?

---

## 1.2 Point Cloud Self-Supervised Pre-Training via Masked Autoencoding

**Problem:** 3D labeled datasets (e.g., ScanNet, ModelNet) are small compared to 2D image datasets. Models trained from scratch on these datasets generalize poorly.

**Idea:** Adopt the **Masked Autoencoder (MAE)** paradigm ([arXiv:2111.06377](https://arxiv.org/abs/2111.06377)) to point clouds:
- Randomly mask 70–80% of point patches.
- Train a Transformer encoder-decoder to reconstruct the missing geometry (XYZ coordinates or SDF values at masked locations).
- Fine-tune on downstream tasks with few labeled examples.

**Why it could work:** Point-MAE ([arXiv:2203.06604](https://arxiv.org/abs/2203.06604)) already validates this idea and shows strong transfer to classification and segmentation. Extending it to outdoor LiDAR scenes with scene-level pre-training is underexplored.

**Open questions:**
- What masking strategy is optimal — random point masking vs block masking vs height-stratum masking?
- How to handle the **scale invariance** problem (indoor vs outdoor scenes)?

---

## 1.3 Physics-Informed Point Clouds via Surface Normals

**Problem:** Current point cloud networks treat geometry as pure coordinates, ignoring differential geometry information (normals, curvatures) that humans use to reason about surfaces.

**Idea:** Augment point features with **second-order differential geometry descriptors** estimated from the local neighborhood:
- **Normal** $\mathbf{n}_i$ — surface orientation.
- **Principal curvatures** $\kappa_1, \kappa_2$ — how much the surface bends.
- Encode these as additional input channels to each token in PCT / PointNet++.

Alternatively, use a **geometric loss**: penalize the network if the predicted per-point features are not consistent with the local tangent plane.

**Why it could work:** Early handcrafted descriptors (FPFH, SHOT) that capture this information were state-of-the-art before deep learning. Injecting this structured prior should improve sample efficiency.
