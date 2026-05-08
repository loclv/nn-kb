# Cross-Modal and Foundation Model Ideas

This document collects research ideas for 3D foundation models and LLM-guided scene understanding.

---

## 5.1 3D Foundation Model via Multi-Task Pre-Training

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

## 5.2 LLM-Guided 3D Scene Editing

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
