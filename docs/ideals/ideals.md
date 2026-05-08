# Research Ideas: Improving 3D World Understanding & Object Learning

This document collects open problems, unsolved challenges, and original research directions in learning from the 3D world. Each idea is grounded in the limitations of current approaches documented in [3d-learning.md](docs/3d-learning.md).

> **How to read this file:** Each idea is structured as:
> - **Problem** — the current gap or limitation
> - **Idea** — a concrete research direction
> - **Why it could work** — the intuition or prior evidence
> - **Open questions** — what would need to be validated

---

## 1. Point Cloud Learning

### 1.1 Dynamic Neighborhood Attention with Learnable Radii

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

### 1.2 Point Cloud Self-Supervised Pre-Training via Masked Autoencoding

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

### 1.3 Physics-Informed Point Clouds via Surface Normals

**Problem:** Current point cloud networks treat geometry as pure coordinates, ignoring differential geometry information (normals, curvatures) that humans use to reason about surfaces.

**Idea:** Augment point features with **second-order differential geometry descriptors** estimated from the local neighborhood:
- **Normal** $\mathbf{n}_i$ — surface orientation.
- **Principal curvatures** $\kappa_1, \kappa_2$ — how much the surface bends.
- Encode these as additional input channels to each token in PCT / PointNet++.

Alternatively, use a **geometric loss**: penalize the network if the predicted per-point features are not consistent with the local tangent plane.

**Why it could work:** Early handcrafted descriptors (FPFH, SHOT) that capture this information were state-of-the-art before deep learning. Injecting this structured prior should improve sample efficiency.

---

## 2. Neural Radiance Fields (NeRF)

### 2.1 Generalizable NeRF via Cross-Scene Attention

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

### 2.2 Dynamic NeRF with Learned Temporal Basis

**Problem:** Standard NeRF assumes a **static scene**. Dynamic scenes (humans moving, cars driving) are modeled by naive time-conditioned NeRFs but this struggles with fast motion.

**Idea:** Represent scene dynamics as a **low-dimensional temporal basis**. Factor the 4D scene into:
- A **canonical static NeRF** $F_\Theta(x, y, z)$.
- A **deformation field** $D_\Phi(x, y, z, t)$ that maps each frame's coordinates back to canonical space.
- Model $D_\Phi$ as a small MLP that predicts SE(3) rigid transforms per region, learned from a **temporal attention** over $k$ learned basis deformations.

**Why it could work:** Consistent with HyperNeRF and Nerfies, which showed that deformation fields + canonical NeRF outperform naive time-conditioning. Temporal basis factorization reduces the degrees of freedom and could work with very few frames.

---

### 2.3 Semantic-Aware NeRF: Fusing Language and Radiance

**Problem:** NeRF represents appearance (color) and geometry (density) but has no semantic understanding. Queries like "render only the car" or "segment the table" are impossible with vanilla NeRF.

**Idea:** Distill **CLIP or DINO features** ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)) into a NeRF by adding a semantic feature head $F_\Theta(x,y,z) \to \mathbf{f}_{sem}$ alongside the color/density heads. Render semantic feature maps using the same volume rendering equation, and supervise with features computed from 2D training views:

$$
\hat{\mathbf{f}}(\mathbf{r}) = \int T(t)\, \sigma\, \mathbf{f}_{sem}(t)\, dt
\quad \approx \quad \text{CLIP}(\text{patch at pixel})
$$

At inference, use the rendered feature map with text or image queries to perform zero-shot 3D segmentation.

**Why it could work:** LERF ([arXiv:2303.09553](https://arxiv.org/abs/2303.09553)) already shows this works well. An extension is to distill not just CLIP but also **depth features** (from Depth Anything) to improve geometry consistency.

---

## 3. 3D Gaussian Splatting (3DGS) Improvements

### 3.1 Semantic 3D Gaussians for Panoptic Scene Understanding

**Problem:** 3DGS Gaussians carry only RGB color (via spherical harmonics). There is no semantic label per Gaussian, so you cannot ask "which Gaussians are the chair?"

**Idea:** Assign each Gaussian a **semantic feature vector** $\mathbf{s}_i \in \mathbb{R}^D$ (in addition to its color). Render semantic feature maps with the same alpha-compositing and supervise with 2D segmentation labels (from SAM or DINO):

$$
\hat{\mathbf{S}}(\text{pixel}) = \sum_{i} \mathbf{s}_i \alpha_i \prod_{j < i}(1 - \alpha_j)
$$

This enables **3D panoptic segmentation** by simply clustering Gaussians in feature space.

**Why it could work:** Feature Splatting ([arXiv:2404.01528](https://arxiv.org/abs/2404.01528)) demonstrates this direction. An extension is to use **open-vocabulary features** (CLIP or DINO) so you can query objects in natural language in real-time 3D.

---

### 3.2 Structured Gaussian Primitives for Articulated Objects

**Problem:** 3DGS cannot naturally represent **articulated objects** (humans, robotic arms) because its Gaussians are assumed to be independent — there is no notion of kinematic structure.

**Idea:** Organize 3D Gaussians into a **part hierarchy** where each rigid body part owns a set of Gaussians. Parameterize deformations via SE(3) transforms on each part:

$$
\mu_i(t) = R_k(t)\, \mu_i^{canonical} + T_k(t), \quad i \in \text{part}_k
$$

Learn the pose of each part $k$ over time from monocular video using an inverse kinematics constraint (bone length preservation).

**Why it could work:** PhysGaussian shows that coupling Gaussians with physics simulation is feasible. Extending this to learned articulation from video is a natural next step toward real-time avatar animation.

---

### 3.3 Compressed 3DGS via Hierarchical Codebook

**Problem:** A typical 3DGS scene requires **hundreds of millions of Gaussians**, consuming 1–6 GB of memory — impractical for mobile or edge devices.

**Idea:** Apply **vector quantization** (VQ-VAE style) to the Gaussian attribute vectors (position, covariance, color SH). Cluster Gaussians into $K$ prototypes and store only indices + a compact codebook:

```
Full Gaussians: N × (3 + 6 + 3 + 48) = N × 60 floats
Quantized:      N × log₂(K) bits  +  K × 60 floats codebook
```

Use a straight-through estimator to train the codebook end-to-end while optimizing scene reconstruction quality.

**Why it could work:** LightGaussian and Compact3D already show that Gaussian pruning + compression is feasible. Learned VQ codebooks could preserve semantically coherent groups of Gaussians better than hand-designed pruning.

---

## 4. 3D Object Detection & Scene Understanding

### 4.1 Language-Grounded 3D Detection (Open-Vocabulary)

**Problem:** Current 3D detectors (CenterPoint, BEVFusion) use a **fixed set of categories** defined at training time. Detecting "a person carrying a box" or "a damaged traffic cone" requires re-training from scratch.

**Idea:** Lift 2D open-vocabulary detection (e.g., GLIP, Grounding DINO) to 3D by:
1. Run open-vocabulary 2D detection on camera images.
2. Lift detections to 3D using **depth estimation + back-projection**.
3. Fuse with LiDAR geometry in BEV to refine 3D bounding boxes.
4. Encode text queries as CLIP embeddings and use them as classifier weights for the 3D detection head.

**Why it could work:** CLIP-BEV and GLIP-3D show strong open-vocabulary 2D→3D transfer. The key challenge is handling misalignment between image and LiDAR modalities for rare categories.

---

### 4.2 Temporal Object Memory with State Space Models (SSM)

**Problem:** Most 3D detectors process each frame **independently**. Temporal context (object velocity, acceleration, occlusion patterns) is only added as a post-hoc tracking step (Kalman filter), not learned end-to-end.

**Idea:** Integrate a **Mamba-style SSM** ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) as a temporal memory module in the BEV backbone. The SSM maintains a hidden state across frames, allowing the model to:
- Track objects through occlusion.
- Predict velocity directly from the hidden state.
- Reason about long-horizon behavior.

$$
h_t = A h_{t-1} + B x_t, \quad y_t = C h_t
$$

where $x_t$ is the current BEV feature map and $y_t$ is the temporally-refined feature.

**Why it could work:** StreamPETR and BEVDet4D show that temporal context improves 3D detection significantly. SSMs are more efficient than transformers for long sequences and can handle variable-length temporal windows.

---

### 4.3 Occupancy Prediction from Sparse Observations

**Problem:** 3D occupancy prediction (predicting which voxels are occupied) requires dense ground truth labels, which are expensive to collect. Current methods like TPVFormer and MonoScene need many labeled frames.

**Idea:** Frame occupancy prediction as a **self-supervised completion** task:
- During training, randomly mask 50% of LiDAR beams.
- Train the model to predict the occupancy of masked regions from unmasked observations.
- At test time, use the full LiDAR scan — the model learns to complete sparse observations from priors learned during masking.

This is analogous to Masked Autoencoders but applied to 3D occupancy grids rather than image patches.

**Why it could work:** Mask-based self-supervision has shown strong gains in 2D (MAE) and 3D point clouds (Point-MAE). Occupancy completion naturally teaches the model about shape priors and scene geometry without any human labels.

---

## 5. Cross-Modal and Foundation Model Ideas

### 5.1 3D Foundation Model via Multi-Task Pre-Training

**Problem:** Unlike 2D vision (where ViT pre-trained on ImageNet or CLIP is universally reused), there is **no universally accepted 3D foundation model**. Each 3D task trains a separate model from scratch.

**Idea:** Pre-train a single large Transformer on **multiple 3D tasks jointly**:

| Task | Input | Output |
|---|---|---|
| Object classification | Point cloud | Class label |
| Part segmentation | Point cloud | Per-point label |
| Scene reconstruction | Multi-view images | 3D Gaussian splats |
| Pose estimation | RGB-D image | 6-DoF pose |
| Occupancy prediction | LiDAR | 3D occupancy grid |

Use a **shared Transformer encoder** with task-specific lightweight decoders (decoder-only heads), similar to how T5 frames all NLP as text-to-text.

**Why it could work:** UniPAD and Uni3D show early evidence that multi-task 3D pre-training transfers better than single-task. The key design choice is the **tokenization** — how to represent point clouds, images, and voxels in a shared token space.

---

### 5.2 LLM-Guided 3D Scene Editing

**Problem:** Editing 3D scenes (moving objects, changing materials, adding new elements) requires expert CAD skills. Current NeRF/3DGS editing methods require manual mask annotations or per-edit optimization.

**Idea:** Use an LLM as a **scene editor planner**: given a natural language instruction ("move the red cup to the left shelf"), the LLM generates a structured edit plan:
```json
{
  "action": "translate",
  "object": "red_cup",
  "delta": [0.3, 0.0, 0.8],
  "verify": "render and check semantic consistency"
}
```
Execute the plan by:
1. **Segment** the object in 3DGS using semantic Gaussians (§3.1).
2. **Transform** the Gaussian subset rigidly.
3. **Inpaint** the hole left behind using a diffusion model conditioned on surrounding Gaussians.

**Why it could work:** Instruct-NeRF2NeRF shows LLM-guided NeRF editing is feasible. Combining it with the explicit, editable structure of 3DGS should make object-level edits faster and more precise.

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
