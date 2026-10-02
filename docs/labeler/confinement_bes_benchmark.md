# Confinement benchmark: the BES classifier

Status: **run 2026-10-01, retrained from the paper's recipe.** The owner named the BES-based
confinement classifier as the benchmark for the confinement labels, the way the CO2 LSTM and RCN are
for the AE labels (2026-10-01). Its weights and code were not found, so the recipe below was
retrained on the corpus BES and scored on shots the network never saw. Script:
`scripts/labeler/confinement_bes_benchmark.py` (stages `features`, `train`, `evaluate`); record:
`outputs/labeler/confinement/bes/evaluation.json` (git `2b4da1d`); features, fold models and
out-of-fold predictions: `$LABELER_ROOT/benchmarks/confinement/bes/`. The paper's title, authors and
citation were not in the paste: fill them in below when known.

Citation: _not given_

## What the paper's model is

A classifier from a block of BES (beam emission spectroscopy) data to a
probability over K = 4 confinement regimes: **L, H, QH, WP QH**.

| Item | Paper |
|---|---|
| Input block | 1024 time samples of the BES array in a 6 (radial rows) x 8 (poloidal columns) configuration, 48 channels |
| Band-pass | 2.5-150 kHz |
| Standardisation | mean and standard deviation of the training set |
| Features | the 1024 samples are split into 2 sub-windows of 512; each is split in 2 segments of 256; FFT of each segment; magnitudes; log10 of the squared magnitudes; the 2 segments averaged. Result: 128 unique spectral features per 512-sample sub-window, per channel (2 x 128 per channel) |
| Network | dropout, 3D convolution (10 kernels of (3, 3, 5) over radial, poloidal, frequency; stride 1; zero padding; groups = 2, one per sub-window), batch norm, LeakyReLU, 3D max-pool (1, 2, 4), flatten, MLP with 2 hidden layers of 60 (LeakyReLU), 4 logits, softmax |
| Loss | cross-entropy over the 4 classes, averaged over the mini-batch |
| Optimiser | Adam with weight decay; one learning rate for the convolution and a different one for the MLP; 60,000 optimiser steps; early stopping after 30 evaluations without improvement (not triggered); the checkpoint with the best validation F1 is kept |
| Software and compute | PyTorch Lightning 1.6.5; 48 A100 GPUs on Perlmutter, one run about 5-10 h |
| Metrics | per-class recall, precision and F1; confusion matrix; one-vs-rest ROC AUC |

**Not in the paste** (equations and symbols were lost, so these need the paper
or the original code): the dropout probability, both learning rates, the weight
decay and Adam betas, the batch size, the parameter counts, the sampling rate and
the window duration in time, the channel selection for the 6 x 8 block, the class
counts (the paper's table 1) and the train/validation/test split.

## Published results (as pasted)

| Confinement regime | F1 | Precision | Recall |
|---|---|---|---|
| L-mode | 0.94 | 0.98 | 0.90 |
| H-mode | 0.97 | 0.95 | 0.98 |
| QH-mode | 0.94 | 0.92 | 0.96 |
| WP QH-mode | 0.90 | 0.95 | 0.86 |
| Average | 0.94 | 0.95 | 0.93 |

These come from the paper's own test split (not described in the excerpt), not from
our labels, so they are the reference for what the model can do, not a number to
compare with ours directly.

## The original is not on disk

No checkpoint or training code for this network was found in the group's project directories. The
nearest relative, `/projects/EKOLEMEN/jzimmerman/bes-ml`, has a four-class path (L 0, H 1, QH 2, WP 3;
channel = row * 8 + column) in plain PyTorch, which is not this network. So the recipe was
reimplemented from the excerpt and retrained, as for the AE detectors that could not be run, except
that here the retraining is ours.

## What was run

| Item | Here |
|---|---|
| Data | the corpus BES (`<shot>_processed.h5`, group `bes`, 64 channels at **500 kHz**) of the 119 shots (185871-196493) of the merged intervals that the corpus holds BES for, out of 448 labelled shots |
| Windows | 1024 samples (2.05 ms) wholly inside one interval of a single regime, one every 2048 samples: 66,532 windows (L 8,209 on 38 shots; H 47,373 on 75; QH 5,608 on 17; WP 5,342 on 20); the one H/L conflict interval is left out |
| Block | channels 8-55, rows 1-6 of the 8 x 8 grid: the paper's 6 x 8 block (which rows is not in the excerpt) |
| Pre-processing | as the paper: band-pass 2.5-150 kHz (4th-order Butterworth, causal), per-channel standardisation by the training windows' standard deviation, 2 sub-windows of 512, 2 segments of 256, FFT, log10 of the squared magnitude of bins 0-127, the two segments averaged: 2 x 128 features per channel |
| Network | dropout, Conv3d (10 kernels (3, 3, 5), zero padding, groups 2), batch norm, LeakyReLU, MaxPool3d (1, 2, 4), MLP 2 x 60, 4 logits: 465,244 parameters |
| Training | cross-entropy, AdamW, 60,000 steps allowed and early stopping after 30 evaluations (every 500 steps) without a better validation loss, as the paper; it stopped at 15,500-16,500 steps. The checkpoint with the best validation macro-F1 is kept (best steps 7,500, 1,500, 12,500, 6,000, 1,500). About 2.3 minutes per fold on one V100S |
| Assumed, not in the excerpt | dropout 0.2, learning rates 1e-3 (convolution) and 1e-4 (MLP), weight decay 0.01, batch 256, natural class frequencies, 500 kHz, rows 1-6, stride 2048 samples. None was tuned |
| Protocol | shot-grouped 5-fold cross-validation, folds dealt within class-presence strata; a fold's test shots are scored by a network trained on three other folds and validated on a sixth of the rest, so every prediction is out of sample. One seed. Intervals are 95 % shot bootstraps (1000 replicates) |

## Results

**All 119 shots, out of sample** (window level, as the paper scores):

| Regime | Windows (shots) | Precision | Recall | F1 [95 % CI] | AUROC | AUPRC | Paper F1 (P / R) |
|---|---|---|---|---|---|---|---|
| L-mode | 8,209 (38) | 0.765 | 0.703 | 0.733 [0.60, 0.84] | 0.945 | 0.783 | 0.94 (0.98 / 0.90) |
| H-mode | 47,373 (75) | 0.946 | 0.939 | 0.943 [0.91, 0.97] | 0.948 | 0.979 | 0.97 (0.95 / 0.98) |
| QH-mode | 5,608 (17) | 0.505 | 0.639 | 0.564 [0.39, 0.71] | 0.942 | 0.496 | 0.94 (0.92 / 0.96) |
| WP QH-mode | 5,342 (20) | 0.495 | 0.452 | 0.473 [0.25, 0.65] | 0.870 | 0.546 | 0.90 (0.95 / 0.86) |
| **Macro** | 66,532 (119) | 0.678 | 0.683 | 0.678 [0.60, 0.75] | 0.926 [0.89, 0.96] | 0.701 | 0.94 (0.95 / 0.93) |

Confusion matrix (windows, rows true):

| true \ predicted | L | H | QH | WP |
|---|---|---|---|---|
| L | 5,773 | 1,527 | 411 | 498 |
| H | 1,462 | 44,493 | 992 | 426 |
| QH | 268 | 222 | 3,583 | 1,535 |
| WP | 43 | 768 | 2,116 | 2,415 |

**On the 27 test shots of the diagnostic-summary models** ([confinement_model_card.md](confinement_model_card.md); the
shots among its 28 test shots that have corpus BES), same predictions: macro F1 0.724 [0.46, 0.86], AUROC
0.955. Only 2 QH and 4 WP shots, so the per-class intervals reach 0.

**Beside the four-class diagnostic-summary baseline**, on the 1,354 50 ms bins of those shots that hold both
(the BES network's window probabilities averaged in the bin; that model's four-class histogram-gradient-boosting
probabilities; truth the bin's merged regime):

| Model | F1 L | F1 H | F1 QH | F1 WP | Macro F1 |
|---|---|---|---|---|---|
| BES network (retrained) | 0.743 | 0.967 | 0.592 | 0.657 | 0.740 |
| Diagnostic-summary baseline | 0.851 | 0.981 | 0.742 | 0.806 | 0.845 |

The paired macro-F1 difference (BES minus baseline) is -0.105, 95 % shot bootstrap
[-0.34, +0.13].

**Agreement-only windows** (intervals where Gill's and Butt's tables agree; 110 shots, 54,817 windows):
macro F1 0.589 [0.50, 0.67]; per class
L 0.354, H 0.943, QH 0.575, WP 0.485. The single-source intervals are all Gill's BES-time files
(5,423 of the 8,209 L-mode windows, 6,252 of the 47,373 H-mode, 40 WP QH, no QH): the network scores them better than the intervals
the two tables share, which is why L-mode drops from 0.73 to 0.35 here.

**Within-shot split (a leaky diagnostic, not a benchmark number).** The same windows with 0.2 s blocks of each
shot assigned at random to train, validation and test (60/20/20), so a test window has training windows of
its own shot beside it: macro F1 0.927 on 13,397 test windows (L 0.95, H 0.99,
QH 0.85, WP 0.91). `train --protocol blocks` reproduces it.

## Reading them

- Across shots the recipe reaches macro F1 0.68 [0.60, 0.75] against the paper's 0.94. H-mode
  is close to the paper (F1 0.94 against 0.97); L-mode is 0.73; QH and WP QH are weak (0.56
  and 0.47) and are mostly taken for each other (1,535 of 5,608 QH windows called WP, 2,116 of 5,342 WP windows
  called QH), though their one-vs-rest AUROC is 0.94 and 0.87.
- The same pipeline scores 0.93 when test windows share shots with training windows, near the paper's 0.94, so the
  pipeline is not what holds the shot-grouped score down: it is generalising to a shot never seen, with 119 shots to learn from
  (QH on 17, WP QH on 20). The excerpt does not say how the paper split its data; if its test windows came from
  shots it also trained on, its figure is of the within-shot kind.
- Beside the four-class diagnostic-summary baseline the BES network is lower on these bins (0.74 against 0.85), but the interval of
  the paired difference includes zero, and only 2 QH and 4 WP shots are in it.
- Gill and Butt are not independent annotators (their tables agree exactly where both cover a shot), and two thirds of
  the L-mode windows come from Gill's BES-time files alone, which no second table confirms.

## Not done

- The corpus holds BES for 119 of the 448 labelled shots. The other 329 (older shots) would need the 64-channel
  record fetched at 1 MS/s (about 1.5 GB a shot), which would roughly triple the training shots and is the likeliest way
  to close the gap.
- Anything the excerpt leaves open was set once and not tuned: dropout, learning rates, batch size, channel rows, window
  stride, class weights; a second seed.
- The paper's own split, class counts and sampling rate (its table 1) are unknown; so is whether it used 1 MHz.

## Source excerpt, as pasted

Web-page controls ("Zoom In", "Download figure", ...) are removed. Equations,
Greek symbols and some numbers were lost in the paste; they are left as gaps.

> **3. Neural network architecture and training**
>
> **3.1. Setup**
> Our goal is to identify the global confinement state (e.g. L, H, QH, or WP QH) from the evolution of local density fluctuations that exist within the real-time BES data stream. Formally, this amounts to a multivariate time-series classification task, which can be thought of as finding a mapping from , where X represents a 3D block of BES data and Y is its corresponding probability distribution over the different confinement regimes. Our approach is to find a classifier (neural network) parameterized by weights and biases θ, such that  is a good approximation of Y.
>
> In order to do this, we train our classifier on data X and corresponding labels Y via an optimization procedure. Specifically, we seek to minimize the following objective function with respect to θ:
>
> [equation lost]
>
> where the inner sum iterates over the K = 4 classes to compute the cross-entropy loss [33], which measures the dissimilarity between the true distribution of labels Ynj and the predicted probability distribution output by the model. The outer sum explicates averaging the loss over a mini-batch of size N (batch-size) of 3D BES data blocks from the training set. The parameters of the network are updated after the completion of a forward pass of the entire mini-batch via gradient descent:
>
> [equation lost]
>
> where the learning rate α controls the step size used by the optimizer. We save the model whose performance on the validation set is most optimal. We provide further practical details of our implementation in the following section 3.3.
>
> **3.2. Preprocessing**
> Figure 4 illustrates the general workflow for data-driven real-time confinement-regime classification in DIII-D, from data collection (figure 4(a)) to model classification (figure 4(e)). The input to the workflow is a  block of BES data (see figure 4(b)), where 1024 corresponds to a  signal window size, and  refers to the spatial configuration of BES consisting of 6 (radially-spanning) rows and 8 (poloidally-spanning) columns. We explore differing input dimensions in section 4.1. We perform the following preprocessing procedure (figure 4(c)). First, we apply a bandpass filter of  to the signal to filter out unwanted and/or irrelevant information such as the low-frequency component due to the neutral beam and the high-end excess photon and electronic noise [44]. As mentioned earlier, broadband turbulence characteristic of L-mode plasmas can be found in this frequency range. The EHO, signature of QH plasmas, is a low toroidal (n) mode number MHD oscillation also found in this frequency range (), and leaves a trace on the density fluctuations made visible by BES. Similarly, in transition to WP QH, this EHO is often lost to edge broadband MHD, of which associated density fluctuations are visible too with edge channels of BES [23]. It is worthy to note that we experimented with broadening and narrowing this frequency range and saw no major improvements to the overall performance and therefore justify our preprocessing selection empirically as well.
>
> Figure 4. Workflow for data-driven real-time confinement regime classification using high-bandwidth fluctuation measurements in DIII-D. (a) Cross-section of DIII-D tokamak, showing nested magnetic flux surfaces and highlighting 2D BES array location. The BES array is shown in  configuration, which we utilize in this work; i.e. in this example, we utilize signals only from turquoise channels, and ignore red channels. (b) Extract signal window of 2D BES array, containing information about density fluctuations and pedestal turbulence dynamics. (c) Raw signals undergo preprocessing by applying a 2.5–150 kHz bandpass filter, standardizing, and further FFT operations (more detail in figure 5). (d) Neural network consisting of shallow convolutional network and multi-layer perceptron (MLP). (e) Model outputs confinement regime with highest probability for given input BES data.
>
> We then standardize the frequency-filtered signals using the mean and standard deviation of the training set to remove shot-to-shot variation of the signals, and to avoid outliers caused by extraneous events such as beam modulations of the NBI. In order to allow for an easier extraction of this meaningful information, we then utilize the Fourier-transformed signals via the fast Fourier transform [58] (FFT). The signal window is split into 2 sub-windows of length 512  to promote the model learning temporal evolution in the data stream. Then, we further split this sub-window in 2 and perform the FFT on both segments of 256 . We retain only the magnitudes (i.e. absolute value of complex-valued FFT), and take the logarithm of the squared magnitudes to avoid exploding and vanishing gradients [59] during neural network training. Finally, we average these 2 segments to reduce noise, resulting in 128 (i.e. due to symmetry about Nyquist frequency) unique spectral-like features for each 512  window. In conclusion, for each channel in the BES array, we recast our original 1024  time-series data into , the latter of which is the only input to the neural network. This way, the network can extract frequency information and still learn temporal evolution across the 2 distinct sub-windows. Figure 5 visualizes this preprocessing technique for one of the 48 signals in the BES array. As these choices are partly empirically motivated, we explore different values for this preprocessing procedure in appendix.
>
> Figure 5. Preprocessing procedure illustrated for 1 channel of BES array. (a)  time window of raw signal from BES channel. (b) Signal after applying 2.5–150 kHz bandpass filter and dividing into first and second half. (c) Subsequent FFT power spectra for different time windows in signal. (d) Result after squaring, then taking the base-10 logarithm of magnitudes to keep values close to unity. (e) Final result is 2 sets of spectral-like features for 2 halves corresponding to first and second  of original signal. For each channel, (e) is fed to neural network.
>
> **3.3. Neural network architecture and training**
> The neural network consists of a convolutional layer followed by a shallow multi-layer perceptron (MLP). In the convolutional layer, we follow the standard procedure of dropout [60], convolution [61], batch normalization [62], LeakyReLU activation [63], and max-pooling, where we note that the convolution and pooling operations are three-dimensional (3D). For the 3D convolution, we used 10 kernels of shape (3, 3, 5) with stride (1, 1, 1) acting in the radial, poloidal, and frequency dimension, respectively, with zero padding, and groups = 2 (number of sub-windows). For the 3D max-pooling operation, we used a kernel of shape (1, 2, 4) acting in the radial, poloidal, and frequency dimension, respectively, with zero padding. The features output by the convolutional layer are then flattened and serve as input to an MLP with 2 hidden layers each of size 60, utilizing LeakyReLU activation. The output layer is of size 4, where a final softmax [64] operation converts the raw logits from the network into a probability distribution over the different confinement regimes.
>
> The model is optimized using the Adam optimizer [65], with weight decay [66] , , and . The learning rate is set to  for the parameters in the convolutional layer and  for the parameters in the MLP. This is to promote stability in the MLP, with far more parameters than in the convolutional layer (i.e.  parameters versus parameters). We set the training process to terminate after 60 000 optimizer steps, with an early stopping mechanism in place that would halt training if the validation loss did not improve over 30 consecutive evaluations. However, the training concluded at the 60 000 step mark, as the early stopping criterion was not met. We saved the model whose F1 score is highest on the validation set, and used that for testing the model. Our framework was implemented using PyTorch Lightning 1.6.5.
>
> Despite the relatively small architecture used (i.e.  K parameters), the sheer volume of data (i.e. ) necessitates parallel processing to manage the high data throughput efficiently, distribute the workload, and maintain a reasonable training time. Therefore, we utilized distributed training using 48 Nvidia A100 Tensor Core Graphics Processing Units (GPUs) on the Perlmutter supercomputer at the National Energy Research Scientific Computing Center (NERSC), resulting in a single training run taking ∼5–10 h.
>
> **3.4. Metrics for evaluation**
> In order to accurately gauge the performance of the network, we report recall, precision, and F1 score:
>
> [equations lost]
>
> where TP, FP, and TN refer to true positives, false positives, and false negatives, respectively. Recall can be intuited as the true positive rate (TPR) (e.g. out of all true L-modes, how many cases does the model classify correctly), whereas precision can be intuited as the positive predictive power of a model (e.g. out of all the cases where the model predicts L-mode, how many of these predictions are correct). In classification tasks, it is common to use the F1 score, i.e. the geometric mean of the precision and recall, to accurately assess performance in instances where there exists a large class imbalance, such as in our case with BES data belonging to L-mode or WP QH-mode being much more scarce than BES data belonging to H-mode or QH-mode (see table 1).
>
> Also common to binary and multi-class classification tasks is the confusion matrix, a square matrix whose shape is determined by the number of classes in the task. The confusion matrix illustrates the classifications made by the model, in a way such that correct classifications fall along the diagonal, while the off-diagonal elements reflect signs of confusion for the model. A perfect classifier, therefore, would be represented by a purely diagonal confusion matrix. Looking along rows one can infer recall values while looking along columns one can infer precision values.
>
> Finally, we use the receiver operating characteristic (ROC) curve to evaluate the performance of our model. The ROC curve is generated by varying the probability threshold at which we classify an input as belonging to a particular class, which in turn affects the TPR and false positive rate (FPR). An optimal classifier is found by its proximity to the top-left corner of the ROC space, representing a perfect classifier (i.e. FPR = 0, TPR = 1). The area under the ROC curve (AUC) is a measure of the model's overall performance, with AUC = 0.5 indicating performance no better than random chance and AUC = 1.0 indicating perfect classification. To extend this binary classification evaluation to our multi-class scenario, we use the same multi-class model and evaluate its performance on binary classification tasks by considering one class against the rest (e.g. L-mode vs. rest, H-mode vs. rest). This approach allows us to assess the separability and performance of each individual class within the multi-class framework.
