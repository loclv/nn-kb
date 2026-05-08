# 3D Object Detection & Scene Understanding

This document collects research ideas for improving 3D object detection and occupancy prediction, focusing on open-vocabulary detection, temporal memory, and self-supervised completion.

---

## 4.1 Language-Grounded 3D Detection (Open-Vocabulary)

**Problem:** Current 3D detectors (CenterPoint, BEVFusion) use a **fixed set of categories** defined at training time. Detecting "a person carrying a box" or "a damaged traffic cone" requires re-training from scratch.

**Idea:** Lift 2D open-vocabulary detection (e.g., GLIP, Grounding DINO) to 3D by:
1. Run open-vocabulary 2D detection on camera images.
2. Lift detections to 3D using **depth estimation + back-projection**.
3. Fuse with LiDAR geometry in BEV to refine 3D bounding boxes.
4. Encode text queries as CLIP embeddings and use them as classifier weights for the 3D detection head.

**Why it could work:** CLIP-BEV and GLIP-3D show strong open-vocabulary 2D→3D transfer. The key challenge is handling misalignment between image and LiDAR modalities for rare categories.

---

## 4.2 Temporal Object Memory with State Space Models (SSM)

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

## 4.3 Occupancy Prediction from Sparse Observations

**Problem:** 3D occupancy prediction (predicting which voxels are occupied) requires dense ground truth labels, which are expensive to collect. Current methods like TPVFormer and MonoScene need many labeled frames.

**Idea:** Frame occupancy prediction as a **self-supervised completion** task:
- During training, randomly mask 50% of LiDAR beams.
- Train the model to predict the occupancy of masked regions from unmasked observations.
- At test time, use the full LiDAR scan — the model learns to complete sparse observations from priors learned during masking.

This is analogous to Masked Autoencoders but applied to 3D occupancy grids rather than image patches.

**Why it could work:** Mask-based self-supervision has shown strong gains in 2D (MAE) and 3D point clouds (Point-MAE). Occupancy completion naturally teaches the model about shape priors and scene geometry without any human labels.
