Diffusion Autoencoders are Scalable Image Tokenizers

Yinbo Chen  $^{1}$  Rohit Girdhar  $^{2}$  Xiaolong Wang  $^{1}$  Sai Saketh Rambhatla  $^{2}$  Ishan Misra  $^{2}$

# Abstract

Tokenizing images into compact visual representations is a key step in learning efficient and high-quality image generative models. We present a simple diffusion tokenizer (DiTo) that learns compact visual representations for image generation models. Our key insight is that a single learning objective, diffusion L2 loss, can be used for training scalable image tokenizers. Since diffusion is already widely used for image generation, our insight greatly simplifies training such tokenizers. In contrast, current state-of-the-art tokenizers rely on an empirically found combination of heuristics and losses, thus requiring a complex training recipe that relies on non-trivially balancing different losses and pretrained supervised models. We show design decisions, along with theoretical grounding, that enable us to scale DiTo for learning competitive image representations. Our results show that DiTo is a simpler, scalable, and self-supervised alternative to the current state-of-the-art image tokenizer which is supervised. DiTo achieves competitive or better quality than state-of-the-art in image reconstruction and downstream image generation tasks. Project page and code: https://yinboc.github.io/dito/.

# 1. Introduction

Image representations play an important role in the visual generative modeling of images and videos (Rombach et al., 2022; Podell et al., 2024; Yu et al., 2022; Dai et al., 2023; Girdhar et al., 2023; Blattmann et al., 2023). Since visual data is high dimensional, a dominant paradigm for generative visual models is to first compress the input pixel space into a compact latent representation, then perform generative modeling in the latent space (Rombach et al., 2022; Yu et al., 2022), and finally decompress the latent space back to pixel space. These compact latents have both theoretical and practical benefits. Compact latents make the generative

$^{1}$ UC San Diego  $^{2}$ GenAI, Meta. Correspondence to: Yinbo Chen <yic026@ucsd.edu>.

Preprint.

![img-0.jpeg](img-0.jpeg)
(a) GAN-LPIPS Tokenizer (GLPTo)

![img-1.jpeg](img-1.jpeg)
(b) Diffusion Tokenizer (DiTo)
Figure 1: Diffusion tokenizer (DiTo) is a diffusion autoencoder with an ELBO objective (e.g., Flow Matching). The input image  $\pmb{x}$  is passed into the encoder  $E$  to obtain the latent representation, i.e., 'tokens'  $\pmb{z}$ , a decoder  $D$  then learns the distribution  $p(\pmb{x}|\pmb{z})$  with the diffusion objective.  $E$  and  $D$  are jointly trained from scratch. In contrast, prior work (a) relies on a combination of losses, heuristics, and pretrained models to learn.

task easier as the lower dimensional representations remove nuisance factors of variation often present in the raw input signal. The latents also allow for smaller generative models yielding both training and inference speed-ups.

We focus on the 'tokenizers' used to learn the latent representations (tokens) for image generation. We study the tokenizers commonly used in state-of-the-art image generation methods (Rombach et al., 2022; Peebles &amp; Xie, 2023; Karras et al., 2024; Podell et al., 2024), which compress the images into continuous latent variables that are further used for learning a latent diffusion generative model. The image reconstruction quality of the tokenizers directly affects the quality of the generative model and thus, studying and improving the tokenizers is of increasing importance.

The most widely used tokenizer, GAN-LPIPS tokenizer (GLPTo) (Rombach et al., 2022; Peebles &amp; Xie, 2023; Karras et al., 2024; Podell et al., 2024), can be viewed as a supervised autoencoder that uses a combination of losses - L1, LPIPS (Zhang et al., 2018) (supervised), and GAN (Goodfellow et al., 2020) to reconstruct the image (see Figure 1). While effective, GLPTo is not ideal yet: (i) the combination</yic026@ucsd.edu>

![img-2.jpeg](img-2.jpeg)
Figure 2: Comparison of GAN-LPIPS tokenizer (GLPTo) and diffusion tokenizer (DiTo). GLPTo uses a weighted combination of L1, LPIPS, and GAN loss, while DiTo only uses a diffusion L2 loss. Despite the simplicity, we observe that when being scaled up, DiTo is competitive to or better than GLPTo for reconstruction, as shown in the examples (at 256 pixel resolution).

of several losses requires tuning weights for each of the individual losses; (ii) L1 and LPIPS losses do not correctly model a probabilistic reconstruction, while it is non-trivial to scale up GANs; and (iii) the LPIPS loss is a heuristic that requires a supervised deep network feature space for image reconstruction. In practice, the GLPTo reconstructions are prone to have artifacts for structured visual input e.g., text and symbols, and high-frequency image regions as shown in Figure 2. These artifacts translate into the image generation model learned on this latent space (Rombach et al., 2022; Chen et al., 2024). Inspired by these observations, we ask the question: does the image tokenizer training have to be so complex and rely on supervised models?

Diffusion models are a theoretically sound (Kingma &amp; Gao, 2024; Dhariwal &amp; Nichol, 2021) and practically scalable (Podell et al., 2024; Polyak et al., 2024) technique for probabilistic modeling of images. However, the theory and practice of using them for learning representations useful for image generation remains underexplored. In this work, we show that a single diffusion loss can be used to build scalable image tokenizers. Our 'Diffusion Tokenizer' (DiTo), illustrated in Figure 1, is trained with a single diffusion L2 loss. At inference, given the latent  $z$ , the decoder reconstructs the image from the latent with a diffusion sampler.

We show design choices that allow us to train and scale DiTo yielding competitive or better representations than the GLPTo. We connect our training to the recent Evidence Lower Bound (ELBO) theory (Kingma &amp; Gao, 2024) of

diffusion models, and use an ELBO objective (Flow Matching (Lipman et al., 2023)) for the diffusion decoder which makes our learned representations maximize the ELBO of the likelihood of the input image, for which we observe the practical benefits. Furthermore, we propose noise synchronization, which aims to synchronize the noising process in the latent space to the pixels space, and allows DiTo's latent representation to be more useful for downstream image generation models.

Beyond its simplicity, DiTo achieves competitive or better quality than GLPTo for image reconstruction, especially for small text, symbols, and structured visual parts. We also find that image generation models trained on DiTo latent representations are competitive to or outperform those trained on GLPTo representations. DiTo can easily be scaled up by increasing the size of the model without requiring any further tuning of loss hyperparameters. We find both the visual quality and reconstruction faithfulness to the input image get significantly improved when scaling up the model. Our ablations further suggest that the effectiveness of DiTo lies in jointly learning a latent representation and a decoder for probabilistic reconstruction.

# 2. Related Work

Diffusion models. Diffusion models are initially proposed and derived as maximizing the evidence lower-bound (ELBO) of data-likelihood in the early work (Sohl-Dickstein et al., 2015). Later works (Nichol &amp; Dhariwal, 2021; Karras

et al., 2022)* improve various aspects of the initial diffusion model, including architecture, noise schedule, prediction type, and timestep weighting, and connect the theory to score-based generative models *(Song and Ermon, 2019; Song et al., 2021b)*, making many of them no longer follow the derivation in the initial work. When being scaled-up, diffusion models beat GANs for image synthesis *(Dhariwal and Nichol, 2021)*, and achieve success for various probabilistic modeling tasks, in particular for text-to-image *(Nichol et al., 2021; Ramesh et al., 2022; Rombach et al., 2022; Betker et al., 2023; Podell et al., 2024)* and text-to-video *(Girdhar et al., 2023)*. The sampling of diffusion models requires iterative denoising, recent efforts are made towards a faster sampler *(Song et al., 2021a; Lu et al., 2022)* or distilling the diffusion model to a one-step generator *(Song and Dhariwal, 2024; Yin et al., 2024b; Xie et al., 2024; Salimans et al., 2024; Yin et al., 2024a; Lu and Song, 2024)*. The recently proposed flow matching *(Lipman et al., 2023)* can be also viewed as a diffusion process with a specific simple noise schedule and $\bm{v}$-prediction *(Salimans and Ho, 2022)* as the training objective.

Image tokenizers. Image tokenizers are autoencoders that convert images to latent representations that can be reconstructed back. Generative models are then usually trained on the latent representations, including autoregressive models for discrete latents *(Esser et al., 2021)*, and diffusion models (or autoregressive diffusion *(Li et al., 2024)*) on continuous latents *(Rombach et al., 2022)*. While diffusion tokenizer is applicable to both types of latents, we focus on continuous latents in this work. A continuous latent space is commonly used by recent state-of-the-art visual generative models *(Peebles and Xie, 2023; Karras et al., 2024; Podell et al., 2024)*, which is obtained by a GAN-LPIPS tokenizer *(Rombach et al., 2022)* (GLPTo). It uses a combination of L1, LPIPS *(Zhang et al., 2018)*, and GAN *(Goodfellow et al., 2020)* loss for image reconstruction, which is an empirical recipe for reconstruction that is also commonly used in super-resolution *(Ledig et al., 2017; Wang et al., 2018, 2021)*. After obtaining the latent space, a latent diffusion model can be trained with UNet *(Rombach et al., 2022)* or Transformer *(Peebles and Xie, 2023)*.

Diffusion autoencoders. The use of a diffusion objective for training image tokenizers remains largely underexplored. Early works *(Preechakul et al., 2022; Pandey et al., 2022)* jointly train an encoder and a diffusion decoder to represent an image as a single latent vector and a noise map for reconstruction. Promising results are shown on simple datasets, while the diffusion autoencoders are mainly used for face attribute editing, and they were not connected to the ELBO objectives in recent work *(Kingma and Gao, 2024)*. DALL-E 3 *(Betker et al., 2023)* trains a diffusion decoder to decode from the frozen latent space of the GLPTo, and distill the diffusion decoder to one-step with consistency model *(Song and Dhariwal, 2024)* for efficiency. Würstchen *(Pernias et al., 2024)* trains a diffusion autoencoder to further compress the frozen latent space of a GLPTo. Concurrent to our work, SWYCC *(Birodkar et al., 2024)* uses a diffusion model to refine a coarse prediction supervised by LPIPS loss in a joint training. $\bm{\epsilon}$-VAE *(Zhao et al., 2024)* trains the autoencoder with LPIPS, GAN, and diffusion loss. Both works show that diffusion loss can be helpful in autoencoder training.

Self-supervised representation learning. Our work is also related to the research in self-supervised representation learning *(He et al., 2020; Chen et al., 2020; Misra and Maaten, 2020; He et al., 2022; Caron et al., 2021; Oord et al., 2018; Donahue et al., 2016; Grill et al., 2020; Bao et al., 2022)*. In particular, our work leverages the long line of research into methods that leverage an autoencoder style reconstruction loss *(Masci et al., 2011; Ranzato et al., 2007; Vincent et al., 2008; He et al., 2022; Salakhutdinov and Hinton, 2009; Bao et al., 2022)*. While many of these methods are focused on representation learning for downstream recognition tasks, we focus on downstream generation tasks. We believe studying unified representations for both generation and recognition is a strong research direction for the future.

## 3 Preliminaries

Score-based models. Most of the recent state-of-the-art diffusion models are based on the theory of score-based generative models *(Song et al., 2021b)*. A diffusion process *(Sohl-Dickstein et al., 2015; Ho et al., 2020)* gradually adds noise to data and finally makes it indistinguishable from pure Gaussian noise. Formally, given a $D$-dimensional random variable $\bm{x}_{0}\in\mathbb{R}^{D}$ that represents the data, the noise schedule is defined by $\alpha_{t},\sigma_{t}$, such that

$q(\bm{x}_{t}|\bm{x}_{0})=\mathcal{N}(\alpha_{t}\bm{x}_{0},\sigma_{t}^{2}\bm{I}),\quad t\in[0,1].$ (1)

A typical design is to let $\alpha_{t}$ decrease from $\alpha_{0}=1$ to $\alpha_{1}=0$, and let $\sigma_{t}$ increase from $\sigma_{0}=0$ to $\sigma_{1}=1$, so that $\bm{x}_{1}\sim\mathcal{N}(\bm{0},\bm{I})$ is a standard normal distribution.

Diffusion models learn to estimate the score function $\nabla_{\bm{x}}\log q(\bm{x}_{t})$ *(Ho et al., 2020; Song et al., 2020)* for all noise levels $t$. To estimate the score function, a neural network $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ is trained typically with the denoising score matching objective *(Ho et al., 2020)*

$\mathcal{L}(\bm{x}_{0})=\mathbb{E}_{t,\bm{\epsilon}}\big{[}||\bm{\epsilon}_{\theta}(\bm{x}_{t},t)-\bm{\epsilon}||_{2}^{2}\big{]},$ (2)

where $\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})$, $\bm{x}_{t}=\alpha_{t}\bm{x}_{0}+\sigma_{t}\bm{\epsilon}$. After training, $\nabla_{\bm{x}}\log q(\bm{x}_{t})\approx-\bm{\epsilon}_{\theta}(\bm{x}_{t},t)/\sigma_{t}$. A sample of $\bm{x}_{0}$ can be generated by first sampling $\bm{x}_{1}$ and then iteratively reversing the diffusion process with the estimated score function using an SDE or ODE solver.

Connection to ELBO. The original diffusion loss *(Sohl-Dickstein et al., 2015)* is derived by maximizing the evidence lower bound (ELBO) of the log-likelihood of data. In practice, later works *(Ho et al., 2020; Nichol and Dhariwal, 2021; Karras et al., 2022)* modified the implementation including noise schedule, prediction type, and timestep weighting for improving the visual quality.

These modifications can be viewed as reweighting the loss for denoising tasks at different log signal-to-noise ratios (SNR) $\lambda_{t}=\log(\alpha_{t}^{2}/\sigma_{t}^{2})$:

$\mathcal{L}(\bm{x}_{0})=\frac{1}{2}\int_{\lambda}w(\lambda)\mathbb{E}_{\bm{\epsilon}}\big{[}||\bm{\epsilon}_{\theta}(\bm{x}_{t(\lambda)},t(\lambda))-\bm{\epsilon}||_{2}^{2}\big{]}\ \mathrm{d}\lambda.$ (3)

While the reweighted variants still learn the correct score function that allows sampling, many of them no longer follow the original derivation of ELBO maximization for the data. Kingma et al. *(Kingma and Gao, 2024)* shows certain conditions under which diffusion losses are equivalent to maximizing an ELBO objective with data augmentation:

$\mathcal{L}(\bm{x}_{0})=\mathbb{E}_{p_{w}(t)}[\mathcal{L}_{t}(\bm{x}_{0})]+\text{constant},$ (4)

where $p_{w}(t)=\frac{\mathrm{d}}{\mathrm{d}t}w(\lambda_{t})$ is a distribution, assuming $w(\lambda_{t})$ is normalized such that $w(\lambda_{1})=1$, and

$\mathcal{L}_{t}(\bm{x}_{0})$ $=D_{KL}(q(\bm{x}_{t...1}|\bm{x}_{0})||p_{\theta}(\bm{x}_{t...1}))$ (5)
$\geq D_{KL}(q(\bm{x}_{t}|\bm{x}_{0})||p_{\theta}(\bm{x}_{t}))$ (6)
$=-\mathbb{E}_{q(\bm{x}_{t}|\bm{x}_{0})}[\log p_{\theta}(\bm{x}_{t})]+\text{constant}.$ (7)

The diffusion objective is ELBO maximization if $p_{w}(t)=\frac{\mathrm{d}}{\mathrm{d}t}w(\lambda_{t})\geq 0$.

We base the theory of our diffusion tokenizers on the diffusion models with ELBO objectives, such as Flow Matching *(Lipman et al., 2023; Albergo and Vanden-Eijnden, 2022; Liu et al., 2022)* as shown in Kingma et al. *(Kingma and Gao, 2024)*, which we detail in approach.

## 4 Approach

Our goal is to learn compressed latent representations of images that can be used for training latent-space image generation models. This compression is learned via a tokenizer that can compress the image from pixel space to latent space (tokens) and decompress it from latent space to pixel space. More formally, given an input image $\bm{x}$ in pixel space, it is passed into an encoder $E$ to obtain the compact latent representation or tokens $\bm{z}$. The latent $\bm{z}$ is used as the condition for a diffusion decoder $D$ that models the distribution $p(\bm{x}|\bm{z})$. An overview of our diffusion tokenizer (DiTo) is shown in Figure 1.

During training, a noisy image $\bm{x}_{t}$ is constructed by adding noise to $\bm{x}$ with the forward diffusion process at random time $t\in[0,1]$, then the diffusion network $D$ takes both $\bm{x}_{t}$ and $\bm{z}$ as input and is supervised by the Flow Matching objective. At test time, given a latent representation $\bm{z}$, the reconstruction image in pixel space can be decoded by first sampling Gaussian noise $\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})$, and then iteratively “denoising” it with reverse diffusion process conditioned on $\bm{z}$. $E$ and $D$ are jointly trained from scratch to learn the latent representation and conditional decoding together.

Training objective. We follow Flow Matching *(Lipman et al., 2023; Albergo and Vanden-Eijnden, 2022; Liu et al., 2022)* that is shown *(Kingma and Gao, 2024)* to be an ELBO maximization diffusion objective. The noise schedule is defined as

$\alpha_{t}=1-t,\quad\sigma_{t}=\sigma_{\min}+t\cdot(1-\sigma_{\min}),$ (8)

where $\sigma_{\min}=10^{-5}$. The diffusion network $D$ uses $\bm{v}$-prediction *(Salimans and Ho, 2022; Lipman et al., 2023)* that is trained with the objective

$\mathcal{L}(\bm{x})=\mathbb{E}_{t,\bm{\epsilon}}\big{[}||D(\bm{x}_{t},t,\bm{z})-\big{(}(1-\sigma_{\min})\bm{\epsilon}-\bm{x}\big{)}||_{2}^{2}\big{]}.$ (9)

The time $t$ is uniformly sampled in $[0,1]$.

Simple implementation. Our implementation only uses a single L2 loss (Equation 9). Thus, unlike GLPTo, it does not require access to pretrained discriminative models to compute LPIPS loss, or training an extra GAN discriminator in an adversarial game. Since we use a single loss, our method does not need a combinatorial search for loss weight rebalancing in contrast to GLPTo. We also observe that discarding the variational KL regularization loss for $\bm{z}$ in GLPTo has negligible impact on DiTo. Finally, DiTo is a self-supervised technique, unlike GLPTo that relies on pretrained supervised discriminative models in LPIPS.

Theoretical justification. A scalable autoencoder typically requires a principled objective. We connect the finding from Kingma et al. *(Kingma and Gao, 2024)* to our diffusion autoencoder to show its theoretical basis. Given the recent results *(Kingma and Gao, 2024)*, our choice of the Flow Matching training objective can be interpreted as learning to compress the image $\bm{x}$ into a latent $\bm{z}$ while maximizing the ELBO $\mathbb{E}_{q(\bm{x}_{t}|\bm{x})}[\log p_{D}(\bm{x}_{t}|\bm{z})]$. That is, $\bm{z}$ is learned to maximize the log probability density of the input $\bm{x}$ augmented at all noise levels $t$ in the expectation. The widely used $\bm{\epsilon}$-prediction (with cosine schedule) *(Nichol and Dhariwal, 2021)* and EDM *(Karras et al., 2022)* are shown *(Kingma and Gao, 2024)* not in this ELBO form and may not directly maximize the log probability density of the input. We study the effects of these objectives in our experiments and observe the practical benefits of the ELBO objectives.

Noise synchronization. We propose an additional regularization on the DiTo’s latent representations $\bm{z}$ that facilitates

training the latent diffusion model on top of them for image generation. When these latents  $z$  are used to train the latent diffusion model, they are noised as  $z_{t}$ . While clean variables  $z_{0}$  are supervised to contain rich information for reconstruction by the diffusion decoder, the noising process from  $t = 0$  to 1 on  $z_{t}$  may potentially destroy the information too quickly or slowly in an uncontrolled way.

To make the diffusion path for the latent variable  $z$  more smooth, we try to synchronize the noising process on the latent  $z$  to the pixel space  $x$ . The idea is to encourage the noisy  $z_{\tau}$  to maximize the ELBO for the noise images  $x_{\tau \ldots 1}$  (Equation (7)). Specifically, during the DiTo training, after obtaining  $z = E(x)$ , we augment  $z_{\tau} = \alpha_{\tau}z + \sigma_{\tau}\epsilon$  with probability  $p = 0.1$  for a random time  $\tau \in [0,1]$ , then use the diffusion decoder to compute the denoising loss with  $t$  sampled in  $[\tau,1]$ . Intuitively, it encourages  $z_{\tau}$  to help denoising  $\{x_t \mid t \in [\tau,1]\}$ , where larger  $\tau$  corresponds to denoising at higher noise levels, which are for more global and lower-frequency information.

# 4.1. Implementation Details

We describe the architecture and training hyperparameters for our diffusion tokenizers.

Architecture. The encoder  $E$  follows the standard convolutional encoder used in Stable Diffusion (LDM (Rombach et al., 2022)) and SDXL (Podell et al., 2024), with the configuration that has a spatial downsampling factor 8, and 4 channels for the latent. The decoder  $D$  is a convolutional UNet with timestep conditioning that follows Consistency Decoder (Song et al., 2023). The  $z$ -condition of the diffusion model is implemented by nearest upsampling  $z$  and concatenation to  $x_{t}$  as the input to the decoder. While the original autoencoder in LDM (Rombach et al., 2022) applies a KL loss on the latent as in a variational autoencoder, we remove it and simply use a LayerNorm (Ba et al., 2016) on  $z$ , which eliminates the burden to balance an additional KL loss (see Appendix B).

Training. Both the encoder and diffusion decoder are jointly trained from scratch. We use AdamW (Loshchilov &amp; Hutter, 2019) optimizer, with constant learning rate 0.0001,  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$ , weight decay 0.01. By default, diffusion tokenizers are trained for 300K iterations with batch size 64. We refer to more details in Appendix A.2.

Inference. We choose the Euler ODE solver for simplicity, and use 50 steps to sample from the diffusion decoder  $D$ .

# 5. Experiments

Dataset. We use the ImageNet (Deng et al., 2009) dataset, which is large-scale and contains diverse real-world images, to train and evaluate our models and baselines for both

|   | Model | rFID@5K  |
| --- | --- | --- |
|  Supervised | (Rombach et al., 2022) | 4.37  |
|   |  GLPTo-B | 4.39  |
|   |  GLPTo-L | 4.05  |
|   |  GLPTo-XL | 4.14  |
|   |  DiTo-B (+LPIPS) | 4.13  |
|   |  DiTo-XL (+LPIPS) | 3.53  |
|  Self-supervised | DiTo-B | 8.91  |
|   |  DiTo-L | 8.75  |
|   |  DiTo-XL | 7.95  |

Table 1: Comparison for image reconstruction on ImageNet. While DiTo-XL shows a higher FID metric, it achieves better visual quality than GLPTo-XL (Fig. 2,4). When adding the supervised LPIPS loss (already used in GLPTo) to explicitly match deep network features, DiTo's FID outperforms GLPTo.

image reconstruction and generation. We post-process the dataset such that faces in the images are blurred. By default, images are resized to be at 256 pixel resolution for the shorter side. For tokenizer training, we apply random crop and horizontal flip as data augmentation. Images are center-cropped for evaluation.

Baselines. We compare to the standard tokenizer used in LDM (Rombach et al., 2022), which we refer to as GLPTo. It is widely used in recent state-of-the-art visual generative models (Peebles &amp; Xie, 2023; Karras et al., 2024; Rombach et al., 2022; Podell et al., 2024). The tokenizer uses L1, LPIPS, and GAN loss for reconstruction. For a fair comparison, we train GLPTo using the same training data and the same architecture that matches the number of parameters to the corresponding DiTo model (see Appendix A.2). The GLPTo downsamples by a factor of 8 and produces a latent  $z$  of size  $4 \times 32 \times 32$ .

Models. Since the main difference of DiTo compared to the baselines is the diffusion decoder, we fix the encoder as the encoder in LDM (Rombach et al., 2022) with a downsampling factor 8 by default, and evaluate several variants of the diffusion decoder in different sizes, the settings are denoted as DiTo-B, DiTo-L, and DiTo-XL with 162.8M, 338.5M, 620.9M parameters in the decoder respectively. The architecture details are provided in Appendix A.1. The same as GLPTo, DiTo's  $z$  is of the size  $4 \times 32 \times 32$ .

Automatic evaluation metrics. We evaluate the commonly used Fréchet Inception Distance (FID) (Heusel et al., 2017) for both the reconstruction and generation. The reconstruction FID (rFID) is computed between a set of input images and their corresponding reconstructed images by the tokenizer. The generation FID (gFID) is computed between randomly generated images and the dataset images. For computation efficiency, we use a fixed set of 5K images from ImageNet validation set to evaluate rFID (which we

![img-3.jpeg](img-3.jpeg)
Figure 3: Scalability of diffusion tokenizers. When increasing the number of trainable parameters in the diffusion decoder from DiTo-B, DiTo-L, to DiTo-XL in the joint training, we observe that the image reconstruction quality keeps improving for structures and textures. Both the visual quality and reconstruction faithfulness are improved when scaling up the diffusion tokenizer.

observe to be stable, while it is typically higher than FID with 50K samples, see Appendix C). We evaluate gFID with 50K samples.

Human evaluation. Recent work shows that automated metrics for evaluating visual generation do not correlate well with human judgment (Girdhar et al., 2023; Podell et al., 2024; Borji, 2019; 2022; Jayasumana et al., 2024). Thus, we also collect human preferences to compare our method and baselines. To compare the two models, we set up a side-by-side evaluation task where humans pick the preferred result. We provide the details in Appendix A.4.

# 5.1. Image reconstruction

We compare the reconstruction quality of DiTo and the baseline GLPTo. Reconstruction quality directly measures the ability of the tokenizer to learn compact latent representations (tokens) that can reconstruct the image. DiTo is trained without noise synchronization (Section 4) by default as we measure the reconstruction quality in this section.

The qualitative results are shown in Figure 2. Despite using a simpler loss, we observe that DiTo shows a better reconstruction quality than GLPTo, especially for regular visual structures, symbols, and text, as shown in the example images. A potential reason might be that the GLPTo relies on the heuristic LPIPS loss that matches the deep network features of the reconstructed image. While it is good for random textures, it may be not accurate enough for structured details. DiTo has principled probabilistic modeling (ELBO) for decoding images, and thus can learn to compress the common patterns, including visual structures and text appearance by compressing images using the self-supervised

![img-4.jpeg](img-4.jpeg)
Figure 4: Comparison for human preference of image reconstructions. Models are compared to GLPTo at the same scale. When being scaled up, we observe that DiTo's (without perceptual loss) visual quality significantly improves and outperforms GLPTo in human preference.

# reconstruction loss.

A quantitative comparison is shown in Table 1. DiTo has a higher reconstruction FID than the GLPTo. FID is computed using distance in a supervised deep network feature space. We hypothesize that the LPIPS loss heuristic plays an important role in the GLPTo to achieve a low FID as it explicitly matches supervised deep network features for the reconstruction and the ground truth. Based on this hypothesis, we train a variant of DiTo that uses an additional LPIPS loss (see Appendix E). Note that LPIPS loss is typically necessary for stability and visual quality in GLPTo training, while it is optional for DiTo. We observe that the supervised variant of DiTo with LPIPS loss achieves lowest FID while controlling for model size, i.e., DiTo-B with LPIPS outperforms a similarly sized GLPTo-B and DiTo-XL with LPIPS

![img-5.jpeg](img-5.jpeg)
Figure 5: Comparison of training objectives in diffusion tokenizers. The frozen  $z$  space is from a GLPTo-B. We observe that when jointly training the encoder and diffusion decoder, ELBO diffusion objectives (flow matching,  $v$ -pred with cosine schedule) can learn good latent representation  $z$ , while other objectives may have color shift in the reconstruction (colors are good given a frozen  $z$  space).

|  Latent encoder | gFID@50K | rFID@5K (Autoencoding)  |
| --- | --- | --- |
|  GLPTo-XL | 7.49 | 4.14  |
|  DiTo-XL | 7.57 | 7.95  |
|  DiTo-XL (w/ noise sync.) | 6.29 | 8.65  |

Table 2: Training image generation models on the latent representations from DiTo and GLPTo. We train DiT models and compare the image generations. We observe that the latent representations from DiTo lead to competitive image generations. Our proposed noise synchronization further improves the generation quality and outperforms the generations using a GLPTo.

also outperforms GLPTo-XL.

Scalability. We study the scalability of DiTo on the three variants - DiTo-B, DiTo-L, and DiTo-XL, where we nearly double the decoder size across each model while keeping the encoder architecture unchanged. A qualitative comparison of the image reconstructions by these models is shown in Figure 3. We observe that both the image reconstruction quality and the reconstruction faithfulness keep improving as the model is scaled up. The improvements of scaling are also confirmed by the reduction in reconstruction FID in Table 1, where the rFID smoothly reduces with model size. However, as shown in Table 1, FID is affected by the supervised LPIPS loss, and many recent works report that it is not aligned with visual quality (Borji, 2019; 2022; Jayasumana et al., 2024). Thus, we use human evaluations to compare the self-supervised DiTo and the supervised GLPTo.

We conduct a side-by-side human evaluation of the image reconstructions from these models and report the preference rate in Figure 4, where a preference greater than  $50\%$  indicates that a model 'wins' over the other. At sizes of B (162.8M) and L (338.5M), the supervised GLPTo's image reconstructions are preferred over those of DiTo. However, when further scaling up to XL (620.9M), we observe

that self-supervised DiTo-XL's reconstructions are preferred over the GLPTo-XL. Qualitatively, we observed that the quality of GLPTo gets mostly saturated when scaling up the decoder and the failure cases are not significantly improved. In contrast, we observed many reconstruction details keep improving for DiTo with the decoder size. This result also shows that DiTo is a scalable, simpler, and self-supervised alternative to GLPTo.

Finally, we note that while evaluating reconstructions is meaningful, in the next step, the representations from DiTo and GLPTo are used to train image generation models. We evaluate how useful these representations are for image generation in Section 5.2.

# 5.2. Image generation

We compare the performance of training a latent diffusion image generation model on the learned latent representation  $z$  from either DiTo or GLPTo. We follow DiT (Peebles &amp; Xie, 2023) and use DiT-XL/2 as the latent diffusion model for class-conditioned image generation on the ImageNet dataset (see more details in Appendix A.3). We compare the image generations from the resulting DiT models in Table 2 and draw several observations.

A DiT trained using DiTo without noise synchronization achieves competitive FID to a DiT trained using GLPTo suggesting that the latent image representations of DiTo are suitable for downstream image generation tasks. Note that when compared in Table 1, DiTo has a higher reconstruction FID than GLPTo with a larger gap. It suggests that the low FID advantage achieved by explicitly matching deep features may not be fully inherited in the image generation stage. A DiT trained on DiTo with noise synchronization achieves the best performance, even outperforming GLPTo in FID. This result confirms the effectiveness of DiTo as a

![img-6.jpeg](img-6.jpeg)
Figure 6: Effectiveness of the latent representation vs. decoder. We train a DiTo decoder-only on a frozen latent space from GLPTo and observe that the reconstruction results are more similar to using a GLPTo decoder (notice similar errors on the visual text reconstruction). These reconstructions are qualitatively different compared to an end-to-end trained DiTo's reconstructions. This suggests that the effectiveness of DiTo comes from jointly learning a powerful decoder and a latent representation.

tokenizer for image generation.

# 5.3. Ablations and Analysis

We now present ablations of our design choices and analyze the key components of DiTo. We follow the same experimental setup as in Section 5.1.

Training objectives. As described in Sections 3 and 4, our DiTo uses a Flow Matching objective which can be viewed as an ELBO maximization for image reconstruction. In contrast, as shown in (Kingma &amp; Gao, 2024), the widely used diffusion implementations such as  $\epsilon$ -prediction (with cosine noise schedule) and EDM (Karras et al., 2022) are not ELBO objectives. We now study the impact of this by training three variants of DiTo and changing the training objective only. We show the examples of the reconstructions in Figure 5. Using the ELBO objectives of Flow Matching and  $\pmb{v}$ -prediction (with cosine schedule, which is also an ELBO objective) yields image reconstructions that are more faithful to the input image. The non-ELBO objectives of  $\epsilon$ -prediction and EDM yield reconstructions sometimes with a noticeable loss of faithfulness, e.g., color shift. To further investigate this, we start with a pretrained GLPTo encoder and keep it frozen while learning diffusion decoders from scratch with the different training objectives. We observe that the image reconstructions do not have such obvious

color shift, suggesting that the non-ELBO objectives can 'decode' correctly but may lead to learning sub-optimal latent representations. A potential reason might be that the non-ELBO objectives have a non-monotonic weight function  $w(\lambda)$  for different log SNR ratios, which makes some terms contribute negatively in Equation (4), and leads to training noise or bias for reconstruction.

Effectiveness of the latent representation vs. decoder. We now study whether the effectiveness of DiTo vs. GLPTo mainly comes from the decoder's powerful probabilistic modeling or from jointly learning both a powerful latent  $z$  and the decoder. We train a DiTo decoder-only on a frozen latent space from a GLPTo and compare the reconstructions to the GLPTo in Figure 6. We observe that both reconstructions look qualitatively similar, and have the same error modes around visual text reconstruction. When compared with reconstructions from an end-to-end DiTo, we observe qualitative differences, e.g., the visual text reconstruction is clearer. This suggests that DiTo's effectiveness lies in jointly learning a powerful latent  $z$  that is helpful to the probabilistic reconstruction objective of the decoder.

# 6. Conclusion and Discussion

We showed that diffusion autoencoders with proper design choices can be scalable tokenizers for images. Our diffusion tokenizer (DiTo) is simple, and theoretically justified compared to prior state-of-the-art GLPTo. DiTo training is self-supervised compared to the supervised training (LPIPS) from GLPTo. Compared to GLPTo, we observe that DiTo's learned latent representations achieve better image reconstruction, and enable better downstream image generation models. We also observed that DiTo is easier to scale and its performance improves significantly with scale.

There are several directions to be further explored for diffusion tokenizers. Our work only explored learning tokenizers for a downstream image generation task. We believe learning tokenizers that work well for both recognition and generation tasks will greatly simplify model training. We also believe content-aware tokenizers that can encode the spatially variable information density in images will likely lead to higher compression. Finally, this paper only studies diffusion tokenizers for images. We believe extending this concept to video, audio, and other continuous signals will unify and simplify training.

# Social Impact

Our method is developed for research purpose, any real world usage requires considering more aspects. DiTo is an image tokenizer, the reconstructed image is perceptually similar but not exactly the same as the input image. The generative diffusion decoder and latent diffusion model may learn unintentional bias present in the dataset statistics.

References

- Albergo et al. (2022) Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.
- Ba et al. (2016) Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization, 2016. URL https://arxiv.org/abs/1607.06450.
- Bao et al. (2022) Bao, H., Dong, L., and Wei, F. BEiT: Bert pre-training of image transformers. In ICLR, 2022.
- Betker et al. (2016) Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn.openai.com/papers/dall-e-3.pdf, 2(3):8, 2023.
- Birodkar et al. (2023) Birodkar, V., Barcik, G., Lyon, J., Ioffe, S., Minnen, D., and Dillon, J. V. Sample what you cant compress, 2024. URL https://arxiv.org/abs/2409.02529.
- Blattmann et al. (2019) Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S. W., Fidler, S., and Kreis, K. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- Borji et al. (2024) Borji, A. Pros and cons of gan evaluation measures. Computer vision and image understanding, 179:41–65, 2019.
- Borji et al. (2021) Borji, A. Pros and cons of gan evaluation measures: New developments. Computer Vision and Image Understanding, 215:103329, 2022.
- Caron et al. (2024) Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In ICCV, 2021.
- Chen et al. (2020) Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In ICML, 2020.
- Chen et al. (2024) Chen, Y., Wang, O., Zhang, R., Shechtman, E., Wang, X., and Gharbi, M. Image neural field diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8007–8017, 2024.
- Dai et al. (2023) Dai, X., Hou, J., Ma, C.-Y., Tsai, S., Wang, J., Wang, R., Zhang, P., Vandenhende, S., Wang, X., Dubey, A., et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.
- Deng et al. (2009) Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.
- Dhariwal et al. (2021) Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- Donahue et al. (2016) Donahue, J., Krahenbühl, P., and Darrell, T. Adversarial feature learning. In ICLR, 2016.
- Esser et al. (2021) Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.
- Girdhar et al. (2023) Girdhar, R., Singh, M., Brown, A., Duval, Q., Azadi, S., Rambhatla, S. S., Shah, A., Yin, X., Parikh, D., and Misra, I. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.
- Goodfellow et al. (2020) Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.
- Grill et al. (2020) Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020.
- He et al. (2022) He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.
- He et al. (2022) He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. Masked autoencoders are scalable vision learners. In CVPR, 2022.
- Heusel et al. (2022) Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- Ho et al. (2024) Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- Jayasumana et al. (2020) Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., and Kumar, S. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9307–9315, 2024.
- Karras et al. (2022) Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=k7FuTOWMOc7.

Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24174–24184, 2024.
- Kingma et al. (2024) Kingma, D. and Gao, R. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36, 2024.
- Ledig et al. (2017) Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al. Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4681–4690, 2017.
- Li et al. (2019) Li, T., Tian, Y., Li, H., Deng, M., and He, K. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024.
- Lipman et al. (2023) Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.
- Liu et al. (2022) Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- Loshchilov and Hutter (2022) Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.
- Lu and Song (2021) Lu, C. and Song, Y. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- Lu et al. (2020) Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- Masci et al. (2022) Masci, J., Meier, U., Cires, D., and Schmidhuber, J. Stacked convolutional auto-encoders for hierarchical feature extraction. In ICANN, pp. 52–59, 2011.
- Misra and Maaten (2021) Misra, I. and Maaten, L. v. d. Self-supervised learning of pretext-invariant representations. In CVPR, 2020.
- Nichol et al. (2021) Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- Nichol and Dhariwal (2021) Nichol, A. Q. and Dhariwal, P. Improved denoising diffusion probabilistic models. In International conference on machine learning, pp. 8162–8171. PMLR, 2021.
- Oord et al. (2022) Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. In NeurIPS, 2018.
- Pandey et al. (2022) Pandey, K., Mukherjee, A., Rai, P., and Kumar, A. DiffuseVAE: Efficient, controllable and high-fidelity generation from low-dimensional latents. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=ygoNPRiLxw.
- Peebles and Xie (2023) Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.
- Pernias et al. (2024) Pernias, P., Rampas, D., Richter, M. L., Pal, C., and Aubreville, M. Würstchen: An efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=gU58d5QeGv.
- Podell et al. (2024) Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=di52zR8xgf.
- Polyak et al. (2022) Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- Preechakul et al. (2022) Preechakul, K., Chatthee, N., Wizadwongsa, S., and Suwajanakorn, S. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10619–10629, 2022.
- Ramesh et al. (2022) Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- Ranzato et al. (2022) Ranzato, M., Huang, F.-J., Boureau, Y.-L., and LeCun, Y. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In CVPR, 2007.
- Rombach et al. (2021) Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Salakhutdinov, R. and Hinton, G. Deep Boltzmann machines. In AI-STATS, 2009.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=TIdIXIpzhoI.

Salimans, T., Mensink, T., Heek, J., and Hoogeboom, E. Multistep distillation of diffusion models via moment matching. arXiv preprint arXiv:2406.04103, 2024.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=St1giarCHLP.

Song, Y. and Dhariwal, P. Improved techniques for training consistency models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=WNzy9bRDvG.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum?id=PxTIG12RRHS.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., and Change Loy, C. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pp. 0–0, 2018.

Wang, X., Xie, L., Dong, C., and Shan, Y. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1905–1914, 2021.

Xie, S., Xiao, Z., Kingma, D. P., Hou, T., Wu, Y. N., Murphy, K. P., Salimans, T., Poole, B., and Gao, R. Em distillation for one-step diffusion models. arXiv preprint arXiv:2405.16852, 2024.

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, W. T. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024a.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In CVPR, 2024b.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Zhao, L., Woo, S., Wan, Z., Li, Y., Zhang, H., Gong, B., Adam, H., Jia, X., and Liu, T. $\epsilon$-vae: Denoising as visual decoding, 2024. URL https://arxiv.org/abs/2410.04081.

|  Model | #Params | c1 | c2 | c3 | temb  |
| --- | --- | --- | --- | --- | --- |
|  DiTo-B | 162.8M | 128 | 256 | 512 | 1280  |
|  DiTo-L | 338.5M | 192 | 384 | 768 | 1280  |
|  DiTo-XL | 620.9M | 320 | 640 | 1024 | 1280  |

Table 3: Configuration details of the UNet diffusion decoder in DiTo at different scales.

![img-7.jpeg](img-7.jpeg)
Figure 7: Training loss curves of DiTo at different scales. We observe the loss keeps improving as scaling up the model and the improvement is not saturated yet. The objective is Flow Matching and the loss is averaged over the latest 10K iterations.

# A. Experiment Details

# A.1. Architecture

We follow the encoder in LDM (Rombach et al., 2022) and the decoder in Consistency Decoder (Song et al., 2023). Both the encoder and decoder are fully convolutional. The UNet diffusion network contains 4 stages, each stage contains 3 residual blocks. In the downsampling phase of the UNet, stages 1 to 3 are followed by an additional residual block with downsampling rate 2. The number of channels in 4 stages are  $c_{1}, c_{2}, c_{3}, c_{3}$  correspondingly. The upsampling phase of the UNet is in reverse order accordingly. The time in the diffusion process is projected to a vector with  $t_{\mathrm{emb}}$  dimension and modulates the convolutional residual blocks. The configurations used for our diffusion tokenizers are summarized in Table 3.

# A.2. Tokenizer training

In the tokenizer training stage, the model is trained with batch size 64 for 300K iterations, which takes about 432, 864, 1728 NVIDIA A100 hours for DiTo-B, DiTo-L, and DiTo-XL models correspondingly. The training loss curves are shown in Figure 7. When scaling up the model, the loss of flow matching objective keeps improving and we did not observe it to be saturated yet. The corresponding baselines GLPTo-B, GLPTo-L, and GLPTo-XL take longer time per training iteration than their DiTo counterparts due to their additional LPIPS and discriminator networks. For all GLPTo, we follow the standard training setting in LDM (Rombach et al., 2022) for models with downsampling factor 8, where the loss weights  $\lambda_{\mathrm{L1}} = 1.0$ ,  $\lambda_{\mathrm{LPIPS}} = 1.0$ ,  $\lambda_{\mathrm{GAN}} = 0.5$ , the gradient norm of regression loss (L1 + LPIPS) and GAN loss are adaptively balanced during training, and the GAN loss is enabled after 50K iterations. To speed up training, we use mix-precision training with bfloat16.

# A.3. Latent diffusion model training

We train DiT-XL/2 (Peebles &amp; Xie, 2023) as the latent diffusion model for the learned latent space. We follow the standard setting that uses batch size 256, Adam optimizer with learning rate  $1 \cdot 10^{-4}$ , no weight decay, and horizontal flips as data augmentation. Flow Matching is used as the training objective. We use classifier-free guidance 2 to generate the samples. To efficiently compare the models, the latent diffusion model is trained for 400K iterations for all tokenizers.

# A.4. Human evaluation

We use Amazon Mechanical Turk (MTurk) to collect human preferences for reconstruction and compare the methods. In the evaluation interface, we first present a few examples of better reconstructions and equally good reconstructions, where for better reconstructions, the number of examples is equal for different methods. The worker is presented with three images

|  Model | Preference vs. GLPTo (%)  |   |   |   |   |
| --- | --- | --- | --- | --- | --- |
|   |  = | > | < | ≥ | ≤  |
|  DiTo-B (+LPIPS) | 26.22 | 34.33 | 39.44 | 47.44 | 52.56  |
|  DiTo-XL (+LPIPS) | 22.11 | 42.56 | 35.33 | 53.61 | 46.39  |
|  DiTo-B | 27.22 | 20.44 | 52.33 | 34.06 | 65.94  |
|  DiTo-L | 22.56 | 23.89 | 53.56 | 35.17 | 64.83  |
|  DiTo-XL | 19.33 | 42.78 | 37.89 | 52.44 | 47.56  |

Table 4: Human evaluation results in detail. Models are compared to the GLPTo at the corresponding size. “&gt;” means DiTo is preferred than GLPTo, “=” means equal preference. “≥” is counted as the value of “&gt;” plus half of the value of “=”.

|  Model | rFID@5K | gFID@5K  |
| --- | --- | --- |
|  DiTo-B (KL loss) | 13.50 | 17.96  |
|  DiTo-B (LayerNorm) | 8.91 | 17.00  |

Table 5: Ablation on DiTo's latent space regularization. rFID is evaluated for autoencoder reconstruction. gFID is evaluated for image generation.

in a row, with tags "input image", "reconstruction 1", and "reconstruction 2". Two reconstructions are from two different methods and are randomly shuffled with 0.5 probability. The worker is asked to select the better reconstruction result based on: (i) the faithfulness of contents to the input image; and (ii) the visual quality of the reconstructed image. There are three available options on the interface: (i) reconstruction 1; (ii) reconstruction 2; and (iii) very hard to tell which is better.

For the comparison of each model pair, we collect 900 preference results. The results in detail are shown in Table 4, where “&gt;” means DiTo is preferred than GLPTo and “=” means equal preference (option (iii)). We count “≥” as the number of “&gt;” plus half of the number of “=”. From the results, we observe that DiTo with LPIPS loss (which is used in GLPTo) is competitive to GLPTo at B size and outperforms GLPTo at larger XL size. DiTo significantly improves as scaling up and outperforms GLPTo at XL size.

# B. Ablation on LayerNorm

Unlike GLPTo (Rombach et al., 2022) which uses a KL regularization loss on the latent  $z$ , in DiTo we only apply LayerNorm on the latent representation  $z$  for both tokenizer and latent diffusion model training. The ablation on this design choice is shown in Table 5. We observe that using LayerNorm has a better reconstruction performance than KL loss for DiTo, and has a competitive performance for image generation. While the weight of KL loss is originally optimized for GLPTo and further sweeping the weight for DiTo may potentially improve the result, we choose LayerNorm for simplicity. There are several main reasons for using LayerNorm in DiTo: (i) The KL loss introduces an additional loss weight to tune, which is not convenient in practice; (ii) Noise synchronization supervises  $z_{t} = \alpha_{t}z_{0} + \sigma_{t}\epsilon$ , LayerNorm ensures  $z_{0}$  to have 0 mean and 1 std so that it does not collapse to the trivial solution; (iii) LayerNorm shows a better reconstruction performance. Moreover, with LayerNorm, the latent representation  $z$  no longer needs to be normalized for training the latent diffusion model.

# C. Comparison to rFID with 50K Samples

For computation efficiency, we evaluate the reconstruction FID on a fixed set of 5K samples. In Table 7, we compare the metric evaluated with 5K samples and 50K samples. The FID evaluated with 50K samples typically has a smaller value than the one evaluated with 5K samples, while it preserves the order in comparison between different methods. We observe FID with 5K samples to be stable enough to compare different checkpoints of the same method or different methods, therefore we mainly compare with FID@5K for more efficient evaluation.

# D. Evaluation on other metrics

We evaluate the autoencoder models on other common metrics for reconstruction, the results are shown in Table 6. Note that GLPTo-XL and DiTo-XL (+LPIPS) are trained with the LPIPS loss. We observe that DiTo-XL has the best PSNR and SSIM. For the metrics associated with the deep network features, LPIPS and Inception Score (IS), GLPTo-XL and DiTo-XL (+LPIPS) achieve better results as they explicitly match the deep features in training (LPIPS loss), while DiTo-XL (+LPIPS) achieves the best results on LPIPS and IS.

|  Model | PSNR (↑) | SSIM (↑) | LPIPS (↓) | IS (↑)  |
| --- | --- | --- | --- | --- |
|  GLPTo-XL | 24.82 | 0.7434 | 0.1528 | 127.06  |
|  DiTo-XL (+LPIPS) | 24.10 | 0.7061 | 0.1017 | 128.05  |
|  DiTo-XL | 25.92 | 0.7646 | 0.2304 | 109.13  |

Table 6: Evaluation on other metrics for reconstruction. Note that GLPTo-XL and DiTo-XL (+LPIPS) are trained with the LPIPS loss.

|  Model | rFID@5K | rFID@50K  |
| --- | --- | --- |
|  GLPTo-XL | 4.14 | 1.24  |
|  DiTo-XL (+LPIPS) | 3.53 | 0.78  |
|  DiTo-XL | 7.95 | 3.26  |

Table 7: Comparison to reconstruction FID (rFID) evaluated with 50K samples. rFID@50K typically has a lower value than rFID@5K, while it is consistent with rFID@5K (for a fixed set) and preserves the order for comparison.

# E. DiTo with LPIPS Loss

In DiTo, the diffusion decoder is trained with Flow Matching objective and does not directly output an image. To apply the LPIPS loss, we need to first convert it to the diffusion model's sample-prediction  $\bar{\pmb{x}} = \mathbb{E}_{q(\pmb{x}_0,\pmb{\epsilon},\pmb{x}_t)}[\pmb{x}_0|\pmb{x}_t]$ , then supervise the sample-prediction with LPIPS loss, so that the gradient can be backpropagated. In general scenarios of diffusion decoders, assume the diffusion network prediction  $f_{\theta}(\pmb{x}_t)$  is minimizing the L2 loss to  $A_{t}\pmb{x}_{0} + B_{t}\pmb{\epsilon}$ , we have

$$
\left[ \begin{array}{l} \boldsymbol {x} _ {t} \\ \boldsymbol {f} _ {\theta^ {*}} (\boldsymbol {x} _ {t}) \end{array} \right] = \left[ \begin{array}{l l} \alpha_ {t} &amp; \sigma_ {t} \\ A _ {t} &amp; B _ {t} \end{array} \right] \left[ \begin{array}{l} \bar {\boldsymbol {x}} \\ \bar {\boldsymbol {\epsilon}} \end{array} \right], \tag {10}
$$

where  $\bar{\epsilon} = \mathbb{E}_{q(\pmb{x}_0,\pmb{\epsilon},\pmb{x}_t)}[\pmb{\epsilon}|\pmb{x}_t]$ ,  $f_{\theta^*}$  is the network prediction at optimal network point  $\theta^*$ . This is because

$$
\begin{array}{l} \boldsymbol {x} _ {t} = \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \boldsymbol {x} _ {t} | \boldsymbol {x} _ {t} ] \\ = \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \alpha_ {t} \boldsymbol {x} _ {0} + \sigma_ {t} \boldsymbol {\epsilon} | \boldsymbol {x} _ {t} ] \\ = \alpha_ {t} \cdot \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \boldsymbol {x} _ {0} | \boldsymbol {x} _ {t} ] + \sigma_ {t} \cdot \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \boldsymbol {\epsilon} | \boldsymbol {x} _ {t} ] \\ = \alpha_ {t} \bar {\boldsymbol {x}} + \sigma_ {t} \bar {\boldsymbol {\epsilon}}, \\ \end{array}
$$

and the optimal prediction under L2 loss is

$$
\begin{array}{l} \boldsymbol {f} _ {\theta^ {*}} (\boldsymbol {x} _ {t}) = \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ A _ {t} \boldsymbol {x} _ {0} + B _ {t} \boldsymbol {\epsilon} | \boldsymbol {x} _ {t} ] \\ = A _ {t} \cdot \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \boldsymbol {x} _ {0} | \boldsymbol {x} _ {t} ] + B _ {t} \cdot \mathbb {E} _ {q (\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t})} [ \boldsymbol {\epsilon} | \boldsymbol {x} _ {t} ] \\ = A _ {t} \bar {\boldsymbol {x}} + B _ {t} \bar {\boldsymbol {\epsilon}}. \\ \end{array}
$$

According to Equation (10), we have

$$
\left[ \begin{array}{l} \bar {\boldsymbol {x}} \\ \bar {\boldsymbol {\epsilon}} \end{array} \right] = \left[ \begin{array}{l l} \alpha_ {t} &amp; \sigma_ {t} \\ A _ {t} &amp; B _ {t} \end{array} \right] ^ {- 1} \left[ \begin{array}{l} \boldsymbol {x} _ {t} \\ \boldsymbol {f} _ {\theta^ {*}} (\boldsymbol {x} _ {t}) \end{array} \right], \tag {11}
$$

In the Flow Matching we used,

$$
\left[ \begin{array}{l l} \alpha_ {t} &amp; \sigma_ {t} \\ A _ {t} &amp; B _ {t} \end{array} \right] ^ {- 1} = \left[ \begin{array}{l l} 1 - t &amp; t \\ - 1 &amp; 1 \end{array} \right] ^ {- 1} = \left[ \begin{array}{l l} 1 &amp; - t \\ 1 &amp; 1 - t \end{array} \right]. \tag {12}
$$

Therefore, the sample prediction is

$$
\bar {\boldsymbol {x}} _ {\theta} \left(\boldsymbol {x} _ {t}\right) = \boldsymbol {x} _ {t} - t \cdot f _ {\theta} \left(\boldsymbol {x} _ {t}\right), \tag {13}
$$

on which we apply the LPIPS loss. Intuitively, it can be also viewed as a "one-step generation" under  $\pmb{v}$ -prediction. Our weight for the LPIPS loss is 0.5.

![img-8.jpeg](img-8.jpeg)
Figure 8: Zero-shot generalization to tokenizing images at higher resolution. Our diffusion tokenizer is fully convolutional and thus can generalize to autoencoding images at resolutions higher than the training setting (256 pixels) in zero-shot. The resolution is  $512 \times 512$  in the shown examples. A quantitative evaluation is shown in Table 8.

![img-9.jpeg](img-9.jpeg)

![img-10.jpeg](img-10.jpeg)

![img-11.jpeg](img-11.jpeg)

|  Model | rFID@5K  |   |
| --- | --- | --- |
|   |  256 × 256 | 512 × 512  |
|  (Rombach et al., 2022) | 4.37 | 2.54  |
|  GLPTo-XL | 4.14 | 2.13  |
|  DiTo-XL | 7.95 | 2.32  |

Table 8: Quantitative comparison of zero-shot generalization to tokenizing images at higher resolution. Similar to GLPTo, DiTo can also generalize to resolutions higher than training. We observe the rFID gap is significantly closed when evaluating at  $512 \times 512$  resolution.

# F. Zero-Shot Generalization of Tokenization for Higher-Resolution Images

We observe that our diffusion tokenizer, when trained for images at fixed  $256 \times 256$  pixels resolution, can generalize to a higher resolution at inference time (without any further training). We show examples for generating images at  $512 \times 512$  resolution in Figure 8, and evaluate the corresponding reconstruction FID in Table 8. From the quantitative results at  $512 \times 512$  pixels resolution, we observe that the reconstruction FID gap between DiTo and GLPTo is significantly closed when generalizing to the higher resolution (from 7.95 vs. 4.14 to 2.32 vs. 2.13).

# G. Motivation of Connecting to ELBO Theory

We explain the motivation for connecting diffusion tokenizers to the recent ELBO theory (Kingma &amp; Gao, 2024) in this section. In theory, given a fixed target distribution, general diffusion models with arbitrary weighting for log signal-to-noise ratios (SNR) can learn the correct score functions, which allow the model to sample from the target distribution. However, in the joint training of diffusion tokenizers, it is not clear what information is encouraged to be in  $z$  when the diffusion decoder is learning the score function of  $p(\boldsymbol{x}_t|\boldsymbol{z})$ . We note that in the view of learning score functions, it is not directly maximizing the probability  $p(\boldsymbol{x}|\boldsymbol{z})$ . Take  $\epsilon$ -prediction as an example, it optimizes

$$
\mathcal {L} = \mathbb {E} _ {q \left(\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t}\right)} \left[ \left| \left| \boldsymbol {\epsilon} _ {\theta} \left(\boldsymbol {x} _ {t}, t\right) - \boldsymbol {\epsilon} \right| \right| _ {2} ^ {2} \right], \tag {14}
$$

and ensures that

$$
\boldsymbol {\epsilon} _ {\theta^ {*}} \left(\boldsymbol {x} _ {t}, t\right) = \mathbb {E} _ {q \left(\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, \boldsymbol {x} _ {t}\right)} \left[ \boldsymbol {\epsilon} \mid \boldsymbol {x} _ {t} \right] \tag {15}
$$

at the optimal point  $\theta^{*}$ . The loss cannot and does not have to be zero, and a smaller loss does not necessarily mean more accurate score functions. Under specific conditions for effective log SNRs weighting (which depends on prediction type, noise schedule, and explicit time weighting), the loss becomes ELBO maximization objective, and minimizing the loss gets a theoretical guarantee.

Intuitively, while learning some representation  $z$  that is helpful to denoising  $x_{t}$  seems to be related to reconstructing  $x$ , the weights for different times are crucial for learning  $z$ . For example, if the weights at small times are too high,  $z$  may not need to store the global information of  $x$  (e.g., the overall color), as such information is always available at small noise

![img-12.jpeg](img-12.jpeg)
Figure 9: Number of decoding steps vs. image reconstruction quality. We vary the number of steps in DiTo's diffusion decoder used for image reconstruction. We use the simple Euler ODE solver and observe that 20 to 50 steps are generally sufficient for good reconstruction quality. The rFID metric mostly converges after 50 steps, while the visual differences are not obvious after 10 to 20 steps.

![img-13.jpeg](img-13.jpeg)

levels while  $z$  only has limited capacity to be allocated. As a result, the reconstruction may have color shifts since  $z$  does not contain sufficient such information. Therefore, we propose to use the diffusion objectives that: (i) are shown to be stable and scalable in prior works; and (ii) are ELBO maximization objectives.

# H. Number of decoding steps vs. quality of reconstruction.

The decoder in DiTo is a diffusion model that reconstruct the image from the latent  $z$  using an iterative denoising process. We study the effect of the number of iterations, i.e., decoding steps on the image reconstruction quality in Figure 9. As expected, the image reconstruction quality improves with more steps (indicated by a lower rFID). However, the visual quality mostly saturates after about 20 steps. Since the iterative process can be computationally expensive, one-step diffusion distillation methods (Song &amp; Dhariwal, 2024; Yin et al., 2024b; Xie et al., 2024; Salimans et al., 2024; Yin et al., 2024a; Lu &amp; Song, 2024) may be applied to speed up decoding in the future.

# I. Additional Qualitative Results

We provide additional qualitative comparisons between the GAN-LPIPS tokenizer (GLPTo) and the diffusion tokenizer (DiTo) at XL size. The input images and corresponding reconstruction results are shown in Figure 10 and Figure 11. We observe that GLPTo and DiTo are competitive in general, and DiTo usually has a better reconstruction quality for regular visual structures, symbols, and text.

![img-14.jpeg](img-14.jpeg)
Figure 10: Additional qualitative comparison of tokenizers (at 256 pixel resolution).

![img-15.jpeg](img-15.jpeg)
Figure 11: Additional qualitative comparison of tokenizers (at 256 pixel resolution).