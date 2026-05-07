# Learning from the 3D World

This document covers neural network approaches for understanding, reconstructing, and generating 3D scenes and objects. The field spans implicit neural representations, explicit point cloud learning, novel-view synthesis, and 3D perception for autonomous systems.

---

## 1. 3D Data Representations

Before choosing a model, you must choose a **representation** for 3D geometry. Each has distinct tradeoffs:

| Representation | Storage | Resolution | Differentiable | Examples |
|---|---|---|---|---|
| **Voxel Grid** | $O(N^3)$ — dense | Fixed | Yes | Early 3D CNNs |
| **Point Cloud** | $O(N)$ — sparse | Variable | Partially | LiDAR scans |
| **Mesh** | $O(V+F)$ | Variable | With care | CAD models |
| **Implicit / SDF** | $O(1)$ parameters | Continuous | Yes | DeepSDF, NeRF |
| **3D Gaussian** | $O(K)$ splats | Adaptive | Yes | 3DGS |
| **Occupancy Grid** | $O(N^3)$ | Fixed | Yes | OccNet |

---

## 2. Point Cloud Learning

### 2.1 PointNet

**Authors:** Qi, Su, Mo, Guibas (2017) — [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)

> *"We design a novel type of neural network that directly consumes point clouds and well respects the permutation invariance of points in the input."*

**Problem:** Point clouds are **unordered** sets of $(x, y, z)$ coordinates. Standard CNNs assume a grid; standard RNNs assume a sequence — neither applies directly.

**Key idea:** Apply a shared MLP to each point independently, then aggregate with a **symmetric function** (global max-pooling) to produce a permutation-invariant global feature:

```
Input: N × 3 point cloud
  ↓ Shared MLP per point  [N × 3  →  N × 64  →  N × 1024]
  ↓ Global Max Pooling    [N × 1024  →  1024]
  ↓ MLP classifier        [1024  →  512  →  k classes]
```

**T-Net (Input Transform):** A mini-network predicts a $3 \times 3$ alignment matrix to canonicalize the input and a $64 \times 64$ feature-space transform to make features invariant to certain transformations.

**Theoretical result:** PointNet approximates any continuous set function, and its global descriptor is determined by a small **critical point set** (the "skeleton" of the shape). This explains robustness to outliers and missing points.

**Results:** State-of-the-art on ModelNet40 classification (89.2% accuracy), ShapeNet part segmentation, and S3DIS scene segmentation — while being much faster than voxel-based methods.

---

### 2.2 PointNet++

**Authors:** Qi, Yi, Su, Guibas (2017) — [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

> *"PointNet does not capture local structures induced by the metric space points live in, limiting its ability to recognize fine-grained patterns."*

**Problem:** PointNet ignores local neighborhoods — it treats each point independently before the global pooling, missing the hierarchical local features that CNNs exploit through receptive fields.

**Key idea:** Apply PointNet **recursively** on a hierarchical nested partition of the point cloud, analogous to how CNNs stack convolutional layers:

```
Set Abstraction Layer:
  1. Farthest Point Sampling  →  select M centroids
  2. Ball Query               →  find K neighbors within radius r
  3. Mini-PointNet            →  encode each local region into a feature vector
  ↓ Repeat for finer / coarser resolution
```

**Multi-scale grouping (MSG):** To handle **non-uniform density** (a common real-world problem with LiDAR), PointNet++ applies ball queries at multiple radii and concatenates the resulting features.

**Results:** Significantly outperforms PointNet on challenging benchmarks, including ScanNet 3D semantic segmentation with varying point densities.

---

### 2.3 Point Cloud Transformer (PCT)

**Authors:** Guo, Tang, Ma, Liu, Ding (2020) — [arXiv:2012.09688](https://arxiv.org/abs/2012.09688)

> *"PCT is inherently permutation invariant for processing a sequence of points, making it well-suited for point cloud learning."*

Applies the **Transformer's self-attention** (see [self-attention.md](self-attention.md)) directly to point clouds. The permutation-invariance of self-attention makes it naturally suited for the unordered point set domain.

**Architecture:**
- **Input Embedding:** Farthest point sampling + k-NN to extract local features as the input tokens.
- **Offset-Attention:** A modified self-attention that uses the *difference* between a point's attention output and its input as the learned offset — this emphasizes local differences and improves gradient flow.

$$
\text{Attn\_output} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V
$$
$$
\text{PCT\_output} = \text{Norm}(\text{Attn\_output} - \text{input})
$$

**Results:** State-of-the-art on ModelNet40 (93.2% accuracy) and ShapeNet part segmentation at the time of publication.

---

## 3. Neural Radiance Fields (NeRF)

### 3.1 Original NeRF

**Authors:** Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng (2020) — [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)

> *"We represent a scene using a fully-connected deep network, whose input is a single continuous 5D coordinate and whose output is the volume density and view-dependent emitted radiance at that spatial location."*

**Core concept:** Represent a scene as a continuous **neural radiance field** $F_\Theta$: a function from a 5D input (3D position + 2D viewing direction) to volume density and RGB color:

$$
F_\Theta : (x, y, z, \theta, \phi) \mapsto (\mathbf{c}, \sigma)
$$

where $\mathbf{c} = (r, g, b)$ is view-dependent color and $\sigma$ is volume density.

**Volume rendering:** To render a pixel, cast a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ through the scene. Integrate along the ray using the **classical volume rendering equation**:

$$
\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt
$$

$$
T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)
$$

$T(t)$ is the accumulated transmittance — the probability that the ray travels from $t_n$ to $t$ without hitting anything.

In practice, this integral is approximated with **stratified sampling** of $N$ points along the ray.

**Positional encoding:** Raw $(x,y,z)$ coordinates are poor inputs for MLPs representing high-frequency detail. NeRF applies a **positional encoding** (similar to the Transformer's):

$$
\gamma(p) = \left(\sin(2^0 \pi p),\, \cos(2^0 \pi p),\, \ldots,\, \sin(2^{L-1} \pi p),\, \cos(2^{L-1} \pi p)\right)
$$

This lifts the input to a higher-dimensional space where high-frequency functions are easier to learn.

**Training:** Minimize the photometric reconstruction loss between rendered and observed pixels:

$$
\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2
$$

No 3D supervision is needed — only 2D images with known camera poses.

**Limitations:**
- **Slow training & rendering:** Each query requires many MLP forward passes.
- **Per-scene optimization:** Must be re-trained for each new scene.
- **Requires known camera poses** (typically from COLMAP).

---

### 3.2 NeRF Variants & Extensions

| Paper | Key Contribution | arXiv |
|---|---|---|
| **NeRF--** | Joint optimization of camera poses and NeRF, removing the need for COLMAP. | [2102.07064](https://arxiv.org/abs/2102.07064) |
| **NeRF++** | Separates foreground/background into two nested NeRFs for unbounded 360° scenes. | [2010.07492](https://arxiv.org/abs/2010.07492) |
| **Mip-NeRF** | Renders conical frustums instead of rays to anti-alias multi-scale representations. | [2103.13415](https://arxiv.org/abs/2103.13415) |
| **Instant-NGP** | Multi-resolution hash encoding on a GPU grid; trains in seconds not hours. | [2201.05079](https://arxiv.org/abs/2201.05079) |
| **Zip-NeRF** | Combines Mip-NeRF 360's anti-aliasing with hash-grid efficiency. | [2304.06706](https://arxiv.org/abs/2304.06706) |
| **Block-NeRF** | Decomposes large-scale city-scale scenes into independently trained blocks. | [2202.05263](https://arxiv.org/abs/2202.05263) |

---

## 4. 3D Gaussian Splatting (3DGS)

**Authors:** Kerbl, Kopanas, Leimkühler, Drettakis (2023) — [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)

> *"We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (≥ 30 fps) novel-view synthesis at 1080p resolution."*

**Core idea:** Replace the implicit MLP of NeRF with an **explicit set of 3D Gaussians**, each parameterized by:
- **Position** $\mu \in \mathbb{R}^3$
- **Covariance** $\Sigma \in \mathbb{R}^{3\times3}$ (anisotropic ellipsoid, decomposed as $\Sigma = RSS^TR^T$)
- **Opacity** $\alpha \in [0, 1]$
- **Color / Spherical Harmonics** coefficients (view-dependent appearance)

**Rasterization (Splatting):** Rather than ray marching, 3DGS *projects* 3D Gaussians onto the 2D image plane as 2D Gaussians and alpha-composites them front-to-back:

$$
C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j < i} (1 - \alpha_j)
$$

This is **massively parallelizable** on GPU — no per-pixel ray marching required.

**Adaptive density control:**
- **Densification:** Split/clone Gaussians in under-reconstructed regions (where gradients are large).
- **Pruning:** Remove transparent or oversized Gaussians.

**NeRF vs 3DGS comparison:**

| Property | NeRF | 3D Gaussian Splatting |
|---|---|---|
| Representation | Implicit MLP | Explicit 3D Gaussians |
| Rendering | Ray marching (slow) | Rasterization (fast) |
| Real-time (1080p) | No | Yes (≥30 fps) |
| Memory | Low (network weights) | High (millions of Gaussians) |
| Editability | Difficult | Easier (move/remove Gaussians) |
| Training time | Hours | 30–45 min |

---

## 5. 3D Object Detection

### 5.1 CenterNet (Objects as Points)

**Authors:** Zhou, Wang, Krähenbühl (2019) — [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)

> *"We model an object as a single point — the center point of its bounding box."*

Rather than enumerating anchor boxes, **CenterNet** detects objects by finding the **center point** of each bounding box on a heatmap, then regressing all other properties (width, height, **depth, 3D orientation**, pose) from that center. This extends seamlessly from 2D to **3D object detection** on the KITTI benchmark using monocular or LiDAR inputs.

**3D extension:**
- Regresses **absolute depth** from a single center feature.
- Uses separate heads for 3D bounding box dimensions and orientation (yaw angle).
- Achieves competitive results on KITTI 3D detection at real-time speed.

---

### 5.2 BEVFusion

**Authors:** Liu et al. (MIT HAN Lab, 2022) — [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)

> *"We break the deeply-rooted convention of point-level fusion with BEVFusion, an efficient and generic multi-task multi-sensor fusion framework."*

**Problem:** Modern autonomous driving systems need to fuse **camera** (rich semantics) and **LiDAR** (precise geometry) data. Prior approaches (point-level fusion) project camera features onto the LiDAR point cloud, discarding spatial semantic density.

**BEVFusion unifies all sensors in the Bird's-Eye View (BEV) space:**

```
Camera Images  →  LSS View Transform  →  Camera BEV Features
LiDAR Points   →  Voxelization + 3D CNN  →  LiDAR BEV Features
                                               ↓
                               BEV Fusion & Task Heads
                           (Detection / Segmentation / etc.)
```

**BEV Pooling optimization:** The view transformation (camera → BEV) was the main bottleneck. BEVFusion introduces a **precomputation + optimized CUDA kernel** that reduces this latency by **40×**.

**Results on nuScenes:**
- 3D Object Detection: **+1.3% mAP** and **NDS** over prior state-of-the-art.
- BEV Map Segmentation: **+13.6% mIoU**.
- **1.9× lower computation cost** vs prior methods.

---

## 6. Implicit Neural Representations (INR)

Implicit Neural Representations encode 3D geometry as the **zero level-set of a function** $f_\Theta : \mathbb{R}^3 \to \mathbb{R}$, where $f_\Theta(\mathbf{x}) = 0$ defines the surface.

### 6.1 Signed Distance Functions (SDF)

An SDF assigns each 3D point the **signed distance to the nearest surface**:
- $f(\mathbf{x}) < 0$ — inside the object.
- $f(\mathbf{x}) = 0$ — on the surface.
- $f(\mathbf{x}) > 0$ — outside the object.

The surface can be recovered via **Marching Cubes** at the $f=0$ isosurface.

### 6.2 Occupancy Networks

**Authors:** Mescheder et al. (2019) — [arXiv:1812.03828](https://arxiv.org/abs/1812.03828)

A network $f_\Theta : \mathbb{R}^3 \times \mathcal{Z} \to [0, 1]$ predicts the probability that a 3D point is inside an object, conditioned on a shape latent code $\mathbf{z}$. The surface is extracted at the $f = 0.5$ decision boundary.

### 6.3 Key INR Properties

| Property | Description |
|---|---|
| **Continuous** | Represents geometry at arbitrary resolution — no voxelization artifacts. |
| **Compact** | Network weights (a few MB) vs voxel grids (GBs at high res). |
| **Differentiable** | Enables gradient-based optimization directly on geometry. |
| **Generalizable** | Can be conditioned on shape codes to represent a whole class of objects. |

---

## 7. Key Papers at a Glance

| Year | Paper | Topic | arXiv |
|---|---|---|---|
| 2017 | PointNet | Point cloud classification & segmentation | [1612.00593](https://arxiv.org/abs/1612.00593) |
| 2017 | PointNet++ | Hierarchical local feature learning on point sets | [1706.02413](https://arxiv.org/abs/1706.02413) |
| 2019 | Occupancy Networks | Implicit 3D shape representation | [1812.03828](https://arxiv.org/abs/1812.03828) |
| 2019 | CenterNet (Objects as Points) | Anchor-free 2D/3D object detection | [1904.07850](https://arxiv.org/abs/1904.07850) |
| 2020 | NeRF | Neural radiance fields for view synthesis | [2003.08934](https://arxiv.org/abs/2003.08934) |
| 2020 | Point Cloud Transformer (PCT) | Transformer for point clouds | [2012.09688](https://arxiv.org/abs/2012.09688) |
| 2021 | NeRF-- | NeRF with unknown camera poses | [2102.07064](https://arxiv.org/abs/2102.07064) |
| 2022 | BEVFusion | Multi-sensor BEV fusion for autonomous driving | [2205.13542](https://arxiv.org/abs/2205.13542) |
| 2023 | 3D Gaussian Splatting | Real-time radiance field rendering | [2308.04079](https://arxiv.org/abs/2308.04079) |

---

## References

- Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
- Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)
- Guo, M., et al. (2020). PCT: Point Cloud Transformer. [arXiv:2012.09688](https://arxiv.org/abs/2012.09688)
- Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- Wang, Z., et al. (2021). NeRF--: Neural Radiance Fields Without Known Camera Parameters. [arXiv:2102.07064](https://arxiv.org/abs/2102.07064)
- Zhang, K., et al. (2020). NeRF++: Analyzing and Improving Neural Radiance Fields. [arXiv:2010.07492](https://arxiv.org/abs/2010.07492)
- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
- Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as Points. [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)
- Liu, Z., et al. (2022). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)
- Mescheder, L., et al. (2019). Occupancy Networks: Learning 3D Reconstruction in Function Space. [arXiv:1812.03828](https://arxiv.org/abs/1812.03828)
