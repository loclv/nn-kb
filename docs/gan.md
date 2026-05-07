# Generative Adversarial Networks (GANs)

Generative Adversarial Networks are a class of generative models introduced by Goodfellow et al. (2014). Rather than directly learning a data distribution, GANs learn through an adversarial game between two competing neural networks.

## 1. The Adversarial Framework

**Authors:** Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio (2014) — [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

> *"We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G."*

### 1.1 The Two Players

| Network | Input | Output | Goal |
|---|---|---|---|
| **Generator (G)** | Random noise vector $\mathbf{z} \sim p_z$ | Fake sample $G(\mathbf{z})$ | Fool D into classifying fake samples as real |
| **Discriminator (D)** | A sample $\mathbf{x}$ (real or fake) | Probability $D(\mathbf{x}) \in [0,1]$ that it is real | Correctly distinguish real from fake |

### 1.2 The Minimax Objective

The training objective is a **minimax game**:

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]
$$

- D is trained to **maximize** this objective (correctly classify real and fake).
- G is trained to **minimize** it — equivalently, to maximize $\log D(G(\mathbf{z}))$ (the non-saturating heuristic).

### 1.3 Theoretical Optimum

Goodfellow et al. prove that the **global optimum** is reached when:
- G perfectly recovers the real data distribution: $p_g = p_{data}$
- D can no longer distinguish real from fake: $D(\mathbf{x}) = 0.5$ everywhere

## 2. Training Dynamics & Challenges

### 2.1 Mode Collapse

The generator may learn to produce a very limited variety of samples (e.g., only generating one class of digit), ignoring the full diversity of the real distribution.

### 2.2 Training Instability

The minimax objective is notoriously difficult to balance. If D becomes too strong, G receives vanishing gradients and cannot improve; if G becomes too strong, D gets confused.

### 2.3 Wasserstein GAN (WGAN)

Arjovsky et al. (2017) proposed replacing the JS-divergence-based loss with the **Wasserstein distance (Earth Mover's Distance)**:

$$
\min_G \max_{D \in \mathcal{D}_L} \mathbb{E}_{\mathbf{x}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))]
$$

where $D$ is constrained to be 1-Lipschitz (via gradient penalty or weight clipping). WGAN provides more stable training and meaningful loss curves.

## 3. Landmark GAN Architectures

| Model | Key Contribution | Year |
|---|---|---|
| **DCGAN** | Replaced FC layers with strided convolutions; stable architecture guidelines for convolutional GANs. | 2015 |
| **Conditional GAN (cGAN)** | Conditions both G and D on class labels $y$, enabling controlled generation. | 2014 |
| **Pix2Pix** | Image-to-image translation using conditional GAN with a U-Net generator. | 2017 |
| **CycleGAN** | Unpaired image-to-image translation using cycle consistency loss. | 2017 |
| **StyleGAN / StyleGAN2** | Introduced style-based generator and mapping network; produces photorealistic faces (FFHQ). | 2019/2020 |
| **BigGAN** | Large-scale class-conditional GAN; high-fidelity ImageNet generation. | 2018 |
| **ProGAN** | Progressive growing of both G and D from low to high resolution. | 2018 |

## 4. Applications

- **Image Synthesis:** Photorealistic face generation (StyleGAN), art creation.
- **Image-to-Image Translation:** Photo colorization, satellite-to-map, day-to-night (Pix2Pix, CycleGAN).
- **Data Augmentation:** Generating synthetic training data to address class imbalance.
- **Super-Resolution:** Upscaling low-resolution images (SRGAN).
- **Drug Discovery:** Generating novel molecular structures.
- **Video Synthesis:** Deepfake generation and detection.

## 5. GANs vs. Other Generative Models

| Model | Approach | Strength | Weakness |
|---|---|---|---|
| **GAN** | Adversarial game | Sharp, high-quality samples | Training instability, mode collapse |
| **VAE** | Variational Inference | Stable training, latent space structure | Blurry samples |
| **Diffusion Models** | Iterative denoising | State-of-the-art quality & diversity | Slow inference |
| **Autoregressive (GPT)** | Next-token prediction | Strong coherence for sequential data | Slow generation; no easy latent space |

---

## References

- Goodfellow, I., et al. (2014). Generative Adversarial Networks. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- Radford, A., et al. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN). [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Karras, T., et al. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN). [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
