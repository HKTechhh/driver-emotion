<!--
DRAFT STATUS (delete this block before submission)
- Every number in Sections 3-5 and the Abstract comes from results/*.csv|json and docs/experiment_log.md;
  robustness (Section 4.5), significance (Section 4.2), abstract and conclusion are now written from the final results.
- A single live webcam frame was captured and added as evidence that the pipeline runs end to end on a real camera
  (Figure 10, right; Section 3.9). This is a spot check, not a systematic live evaluation with drivers - if you run a
  full live session, add the observed accuracy, FPS and behaviour over time.
- TODO(references): all references were written from memory. Verify each one (authors, year, venue, pages/DOI)
  before submitting, and add the literature that your department expects in Section 2.
- TODO(format): author/supervisor/institution are placeholders. Citations and the reference list are now APA 7th
  edition; the template and page limit are still not applied (currently ~16 pages with 14 figures).
- This is a first draft of the writing, not a finished paper. Read it, correct it, and put it in your own words.
-->

# Recognising Driver Emotions from Facial Images under Urban Traffic Conditions: A Comparison of a Custom VGG-Style CNN and Fine-Tuned VGG16

**Author:** *[your name]* &nbsp;|&nbsp; **Supervisor:** *[name]* &nbsp;|&nbsp; **Institution / Department:** *[…]* &nbsp;|&nbsp; **Date:** *[…]*

## Abstract

Recognising a driver's emotional state from a camera could support safer driving, but models that score well on clean benchmarks may fail on the road. This study compares two convolutional models for seven-class facial expression recognition on FER2013 (28,709 training and 7,178 test images): a compact custom CNN of VGG-style blocks (4.8 M parameters) trained from scratch, and an ImageNet-pretrained VGG16 (15.0 M parameters) fine-tuned in two stages. On the test set the custom CNN reached 67.5% accuracy and 0.660 macro-F1 against 65.8% and 0.640 for VGG16. The 1.7-point accuracy gap is small but statistically distinguishable (paired bootstrap 95% CI 0.7 to 2.8 points; McNemar *p* = 0.0014), and the CNN ran 5.4 times faster on a laptop CPU (25 versus 4.6 frames per second). Under simulated driving conditions applied to the whole test set, the CNN degraded less than VGG16 in all five cases, by a large margin only for motion blur; low light reduced both models to roughly the accuracy of always predicting the most frequent class, while in-plane head rotation was nearly harmless. Grad-CAM showed both models attending to facial features when correct, whereas VGG16's errors more often rested on off-face regions. Limitations: FER2013 consists of web images, not in-cabin footage; the degradations are synthetic; each model was trained once; and the real-time pipeline was verified only on simulated frames.

**Keywords:** facial expression recognition, driver monitoring, convolutional neural networks, transfer learning, robustness, Grad-CAM, FER2013

---

## 1. Introduction

Road traffic injuries remain a leading cause of death worldwide (World Health Organization, 2023), and a substantial share of crashes involve driver error rather than mechanical failure. A driver's emotional state is one factor that can degrade driving: anger and other high-arousal states are associated with aggressive behaviour, and low-arousal states such as sadness with reduced attention (Jeon, 2016; Mesken et al., 2007). If a vehicle could notice, from the driver's face, that the driver is becoming angry or afraid, it could respond with a calm prompt, a change in assistance level, or a record for the driver's own review. This is the motivation behind *driver monitoring systems* that use an in-cabin camera.

Recognising emotion from a face is a hard problem even in benign conditions. Expressions are subtle and ambiguous, human annotators disagree with one another, and the standard benchmark used here (FER2013) is known to be noisy (Goodfellow et al., 2013; Barsoum et al., 2016). Driving makes it harder still. Illumination changes quickly, from tunnels to low sun; motion and vibration blur the image; hands, sunglasses and steering-wheel spokes occlude the face; and the driver's head is rarely frontal. A model that scores well on a clean benchmark may therefore fail on the road, and an accuracy figure alone does not say how.

This project studies that gap with a deliberately simple design: two convolutional models, one benchmark, and a controlled set of "urban traffic" corruptions.

**Research questions.**

- **RQ1.** How does a compact custom CNN built from VGG-style blocks, trained from scratch, compare with an ImageNet-pretrained VGG16 fine-tuned by two-stage transfer learning, in overall accuracy, per-class behaviour and computational cost?
- **RQ2.** How does each model degrade under conditions typical of driving: low light, glare, motion blur, partial occlusion and head rotation?
- **RQ3.** Where in the face does each model look when it is right, and when it is wrong?
- **RQ4.** Is inference cheap enough for a commodity CPU to run a live pipeline?

**Contributions.** (i) A like-for-like comparison of the two models under one preprocessing pipeline, one data split and one evaluation protocol, with all results logged. (ii) A reproducible robustness protocol for simulated driving conditions applied to the whole test set. (iii) A Grad-CAM analysis of correct and incorrect predictions. (iv) A real-time pipeline and its measured inference cost. (v) An account of the pitfalls met along the way (Section 5.4), several of which silently produced plausible-looking but wrong numbers, because they are the kind of error that also affects published comparisons.

**Scope.** All accuracy claims are on FER2013 and on *simulated* degradations of it. No in-cabin footage was used, so the results do not establish performance on real drivers (Section 5.5).

---

## 2. Literature review

*This section is a scaffold. It contains only claims I am confident are well established; it needs your department's expected depth, and every citation must be verified.*

### 2.1 Emotion and driving

Discrete-emotion models, in which a small set of "basic" emotions is recognisable across cultures from facial expression, go back to Ekman and Friesen (1971). FER2013 and most subsequent benchmarks use the seven-class version (anger, disgust, fear, happiness, sadness, surprise, neutral). Studies of emotion in driving report that specific emotional states, notably anger, are associated with riskier driving behaviour, and that sadness can impair performance as well (Jeon, 2016; Mesken et al., 2007). That is the rationale for treating anger and fear as the *alert* emotions in our real-time application (Section 3.9), while acknowledging that a facial expression is not the same thing as an inner state (Section 6).

### 2.2 Facial expression recognition and FER2013

Facial expression recognition (FER) with deep networks is surveyed by Li and Deng (2022). The FER2013 dataset (Goodfellow et al., 2013) contains 48 × 48 grayscale faces collected from the web and labelled with seven emotions; it became a standard benchmark because it is large, free and difficult. Its labels are noisy: the original report notes human accuracy of roughly 65% (Goodfellow et al., 2013), and relabelling efforts such as FER+ (Barsoum et al., 2016) were produced precisely because of this. Later datasets such as AffectNet (Mollahosseini et al., 2019) are larger and collected in the wild. Reported FER2013 accuracies for single, heavily tuned VGG-style networks are in the low 70s percent (Khaireddin & Chen, 2021); results in that range typically involve extensive hyper-parameter search, specific optimisers and schedules, and test-time augmentation, none of which are attempted here. A ceiling well below 100% should therefore be expected, and our results should be read against that ceiling and not against 100%.

### 2.3 Convolutional architectures and transfer learning

Deep convolutional networks (LeCun et al., 2015; Krizhevsky et al., 2012) are the dominant approach to image classification. VGG (Simonyan & Zisserman, 2015) showed that stacking small 3 × 3 convolutions between pooling layers gives a simple, effective and easily described design; VGG16 trained on ImageNet (Russakovsky et al., 2015) is a common starting point for transfer learning. Batch normalisation (Ioffe & Szegedy, 2015) and dropout (Srivastava et al., 2014) are standard regularisers, and Adam (Kingma & Ba, 2015) a standard optimiser. Transfer learning is usually most valuable when the target data are scarce and resemble the source domain. FER2013 is small by modern standards but differs from ImageNet in resolution, colour and content, so it is not obvious in advance that transfer will beat a compact network trained from scratch. RQ1 tests exactly this.

### 2.4 Driver monitoring

In-cabin driver monitoring typically combines a camera (often near-infrared, so that it works at night) with models for attention, drowsiness and, more recently, affect. Publicly available data for *driver* expressions are scarce. KMU-FED (Jeong & Ko, 2018) is a database of facial expressions captured in a vehicle; we did not obtain it, which is a limitation of this study (Section 5.5).

### 2.5 Explaining what a network looks at

Grad-CAM (Selvaraju et al., 2017) uses the gradient of a class score with respect to the activations of a late convolutional layer to produce a coarse heat map of the regions that most influenced that score. It is widely used to check whether a classifier attends to plausible evidence or to an artefact such as the background. It is a diagnostic, not a proof of causal reliance.

### 2.6 Gap addressed here

Comparisons of FER models usually report clean-test accuracy and stop. We add a controlled robustness sweep motivated by driving, compare a compact from-scratch model against a fine-tuned pretrained model on cost as well as accuracy, and inspect both correct and incorrect predictions.

---

## 3. Methodology

Figure 1 summarises the whole study: the offline pipeline that trains and evaluates the two models, and the real-time application that reuses the same preprocessing.

![Figure 1. Overview: (A) training and evaluation, (B) the real-time application. The shared preprocessing module keeps live frames identical to training images.](figures/pipeline.png)

### 3.1 Data

We used the FER2013 images as distributed in the widely used Kaggle folder version, arranged as one directory per class. The **train** partition has 28,709 images and the **test** partition 7,178 (the latter equals the combined size of the original PublicTest and PrivateTest partitions). Images are 48 × 48, 8-bit grayscale. Class counts are strongly imbalanced (Table 1; Figure 2): *disgust* is about 16.5 times rarer than *happy* in training.

**Table 1.** FER2013 class counts.

| Class | Train | Test |
|---|---|---|
| angry | 3,995 | 958 |
| disgust | 436 | 111 |
| fear | 4,097 | 1,024 |
| happy | 7,215 | 1,774 |
| neutral | 4,965 | 1,233 |
| sad | 4,830 | 1,247 |
| surprise | 3,171 | 831 |
| **Total** | **28,709** | **7,178** |

![Figure 2. FER2013 class distribution (left) and sample faces (right; the first five files of each class in the training partition).](figures/class_distribution.png;figures/sample_grid.png)

A validation set was carved from the training partition (15%, seed 42), giving 24,403 training and 4,306 validation images. The split is by file and is not stratified by class. **The test partition was used only for the final evaluation, and for choosing nothing.** Model selection used the validation set alone.

### 3.2 Preprocessing and augmentation

Both models share one preprocessing module (`src/preprocess.py`) that is used identically in training, evaluation, robustness testing and the live application, so that a live frame is transformed exactly like a training image.

- **Custom CNN:** grayscale, 48 × 48, pixel values scaled to [0, 1].
- **VGG16:** grayscale converted to three channels, resized to 224 × 224 (bilinear), then Keras' VGG16 `preprocess_input` (per-channel mean subtraction).

Training images are augmented with a random horizontal flip, rotation (factor 0.05, i.e. up to ±18°), zoom (10%), brightness (±20%) and contrast (±20%). Augmentation is applied to the raw [0, 255] pixels *before* the model-specific rescaling; the reason is a bug described in Section 5.4.

### 3.3 Model A: custom VGG-style CNN

The custom CNN (4,826,055 parameters) has four VGG-style blocks. Each block is two 3 × 3 convolutions (padding "same", L2 weight decay 10⁻⁴), each followed by batch normalisation and ReLU, then 2 × 2 max-pooling and dropout 0.25. The blocks use 64, 128, 256 and 512 filters. The head is global average pooling, a 256-unit dense layer with batch normalisation, ReLU and dropout 0.4, and a 7-way softmax. Layers are named (`block1_conv1`, …, `block4_conv2`) so that the network reads clearly in a viewer and so that Grad-CAM can target `block4_conv2`. The architecture is drawn in Figure 3.

![Figure 3. The custom CNN as rendered by the Netron model viewer from the trained checkpoint `cnn_v1.keras`. Read the four columns left to right; each node shows the layer type and its weight shapes.](figures/netron_cnn_v1.png)

### 3.4 Model B: VGG16 with two-stage transfer learning

Model B (14,980,935 parameters) is Keras' VGG16 with ImageNet weights and no top, followed by global average pooling, a 512-unit ReLU dense layer, dropout 0.5 and a 7-way softmax (Figure 4). Training has two stages. **Stage 1** (15 epochs, learning rate 10⁻³) trains only the new head with the convolutional base frozen (266,247 trainable parameters). **Stage 2** (25 epochs, learning rate 10⁻⁵) unfreezes the last convolutional block (`block5_conv1` onward) and fine-tunes it together with the head. Layers up to `block4_pool` remain frozen throughout.

![Figure 4. VGG16 with the new head as rendered by Netron from `vgg16_v1.keras` (13 convolutions, 5 poolings, global average pooling, a 512-unit dense layer, dropout, and the 7-way output). Read left to right.](figures/netron_vgg16_v1.png){5.6}

### 3.5 Training protocol

Table 2 lists the setup. Both models use Adam (Kingma & Ba, 2015) and sparse categorical cross-entropy, with class weights (Section 3.6). Callbacks are shared: the checkpoint with the best **validation** accuracy is kept; early stopping (patience 10, restore best weights); `ReduceLROnPlateau` on validation accuracy (factor 0.5, patience 4). The custom CNN used batch size 64 and a maximum of 60 epochs from a learning rate of 10⁻³; VGG16 used batch size 32 and the two stages above. Random seeds were fixed (seed 42) for Python, NumPy and TensorFlow, but the GPU kernels are not bit-wise deterministic, and **each model was trained once**, so run-to-run variation is not measured. Training was done on a Kaggle NVIDIA T4 GPU (custom CNN: 21.5 min; VGG16: 123.4 min). Software: Python 3.11.15, TensorFlow 2.20 / Keras 3.15 for the local evaluation.

**Table 2.** Training setup for the two models.

| Setting | Custom CNN | VGG16 (fine-tuned) |
|---|---|---|
| Input | 48 × 48 × 1, scaled to [0, 1] | 224 × 224 × 3, VGG16 `preprocess_input` |
| Parameters | 4,826,055 | 14,980,935 |
| Optimiser, learning rate | Adam, 10⁻³ | Adam, 10⁻³ (stage 1), 10⁻⁵ (stage 2) |
| Batch size | 64 | 32 |
| Epochs run | 60 | 15 + 25 = 40 |
| Regularisation | L2 10⁻⁴ (conv), dropout 0.25 per block and 0.4 in the head, batch norm | dropout 0.5 in the head |
| Schedule / stopping | LR halved after 4 epochs without val-accuracy gain; early-stop patience 10 | same |
| Class weights | square root of balanced, capped at 3.0 | same |
| Augmentation | flip, rotation ±18°, zoom 10%, brightness ±20%, contrast ±20% | same |
| Hardware, time | Kaggle T4 GPU, 21.5 min | Kaggle T4 GPU, 123.4 min |

### 3.6 Handling class imbalance

sklearn's "balanced" class weights give *disgust* a weight of about 9.4. In our runs this destabilised training: in a controlled comparison (same seed and data, 15% subset), training with no class weights progressed normally, whereas balanced weights left training accuracy at chance level, and lowering the learning rate to 3 × 10⁻⁴ did not resolve it over the four epochs tested. We therefore use **square-root-damped weights capped at 3.0** (disgust 9.4 → 3.0), which keeps an imbalance correction without the instability, and report macro-F1 alongside accuracy so that majority-class behaviour is visible.

### 3.7 Evaluation metrics and protocol

On the full test partition we report accuracy, macro-F1 (the unweighted mean of per-class F1, which weights the rare classes equally), weighted-F1, per-class precision, recall and F1, and row-normalised confusion matrices. We also record parameter count, saved file size and single-image inference latency: the mean and 95th percentile of 200 timed calls after 20 warm-up calls, using direct `model(x)` calls on one image, from which frames per second (FPS) is derived. To ask whether the difference between the two models could be test-sample noise, we compared their predictions image by image: 95% bootstrap intervals (2,000 resamples of the test images, seed 42) for accuracy and macro-F1, a paired bootstrap of the differences, and an exact McNemar test on the images for which exactly one model is correct. These intervals reflect sampling of the test set only; they do not capture variation between training runs. Latency was measured on the development laptop (Intel Core i5-8265U, 8 threads, no GPU), so it says something about a modest CPU and nothing about an in-vehicle processor.

### 3.8 Robustness protocol

To probe RQ2, six conditions are applied to *every* test image (7,178), deterministically (seeded generator), at native 48 × 48 resolution before the model-specific resize (Table 3; example images in Figure 11):

**Table 3.** Simulated driving conditions (each applied to every test image, seeded).

| Condition | Simulation |
|---|---|
| clean | none (baseline) |
| low_light | brightness × 0.4, plus Gaussian noise (σ = 10), for night driving or tunnels |
| glare | brightness × 1.6, plus a random bright, blurred elliptical blob, for sun on the windscreen |
| motion_blur | horizontal 7-pixel box blur, for vehicle motion and vibration |
| occlusion | a random black rectangle covering about 20% of the image, for a hand or sunglasses |
| head_pose | in-plane rotation drawn uniformly from ±20°, for a driver glancing away |

Both models are re-evaluated on each corrupted copy of the test set and accuracy and macro-F1 are recorded. These are *simulations*: they are cheap and reproducible but they are not real recordings, and "head_pose" is only an in-plane rotation, not a change of viewpoint.

### 3.9 Explainability and the real-time pipeline

**Grad-CAM.** Heat maps are computed from `block4_conv2` (custom CNN, a 6 × 6 map at 48-pixel input) and `block5_conv3` (VGG16, a 14 × 14 map), for the class the model predicted, and overlaid on the face. Because of the compute involved, the figures use a class-balanced 25% subset of the test set (about 1,800 images). We show one example per emotion that both models classify correctly, and six misclassified examples per model, one per true class for the six classes with most errors.

**Real-time pipeline.** The application captures frames with OpenCV, detects faces with MediaPipe's BlazeFace detector (Bazarevsky et al., 2019) (falling back to an OpenCV Haar cascade if MediaPipe is unavailable), crops the largest face with 15% padding, applies the shared preprocessing, predicts, and smooths the class probabilities with a 10-frame moving average. If *angry* or *fear* remains the top class for more than 3 seconds it shows an on-screen prompt. Per-frame output (timestamp, emotion, confidence, FPS) is logged to a CSV; **no image is stored**. *Verification status:* the full detect-crop-preprocess-predict path was run with the trained checkpoints on *simulated* frames (a test face enlarged six times and placed on a grey 640 × 480 canvas). A face was found in 118 of 120 frames for the custom CNN and 38 of 40 for VGG16. End-to-end accuracy on those frames was 0.669 (CNN, n = 118) and 0.789 (VGG16, n = 38; a small sample), against 0.708 and 0.725 for the same models applied directly to the original 48-pixel images, differences that are within sampling noise at these sample sizes. However, the label produced by the real-time path agreed with the direct label on only 78% (CNN) and 76% (VGG16) of frames: the detector's box plus 15% padding frames the face differently from the tight FER2013 crops the models were trained on, and individual predictions are sensitive to that framing. A live webcam session was additionally run to confirm the pipeline operates end to end on a real camera feed (Figure 10, right); this single-frame spot check is not a systematic live evaluation, so no live accuracy claim beyond it is made, and the loop has **not** been evaluated with actual drivers.

---

## 4. Results

### 4.1 Training behaviour

Figures 5 and 6 show the training curves. The custom CNN was trained for the full 60 epochs; its best validation accuracy of **0.6665** occurred at epoch 59. The learning rate was halved six times by `ReduceLROnPlateau` (at epochs 11, 24, 30, 38, 48 and 55), from 10⁻³ to 1.6 × 10⁻⁵. Validation loss reached its minimum (1.092) at epoch 32 and stayed near 1.12 afterwards, and validation accuracy gained only about one point over the last twenty epochs, while training accuracy kept rising to 77%: a moderate generalisation gap, not runaway overfitting (Figure 5).

VGG16's frozen-base stage plateaued at about 48–50% validation accuracy (50.1% at the end of stage 1). Unfreezing the last block produced an immediate jump (55.4% in the first fine-tuning epoch) and a best validation accuracy of **0.6600** at epoch 38 of 40. From about epoch 20, its validation loss flattens near 1.0 while training loss keeps falling (73.6% training vs. 65.7% validation at the end), so it overfits somewhat more than the custom CNN (Figure 5).

![Figure 5. Training curves. Left: custom CNN. Right: VGG16 (dashed line: start of fine-tuning).](figures/cnn_v1_curves.png;figures/vgg16_v1_curves.png)

![Figure 6. TensorBoard view of the same per-epoch values (smoothing off): accuracy (left) and loss (right), training and validation, both models. The original TensorBoard event files of the Kaggle runs were not exported, so this view was regenerated from the per-epoch history files with `src/export_tensorboard.py`; the plotted values are identical to the CSVs.](figures/tensorboard_curves.png)

### 4.2 Test-set performance

Table 4 gives the headline results, Table 5 their uncertainty and the paired comparison, and Figure 7 summarises accuracy, macro-F1 and speed.

**Table 4.** Test-set results (7,178 images; the test set was not used for any selection).

| | Custom CNN | VGG16 (fine-tuned) |
|---|---|---|
| Accuracy | **0.6753** | 0.6581 |
| Macro-F1 | **0.6595** | 0.6397 |
| Weighted-F1 | **0.6743** | 0.6506 |
| Best validation accuracy | 0.6665 | 0.6600 |
| Parameters | 4.83 M | 14.98 M |
| File size | 55.4 MiB | 113.3 MiB |
| Latency, mean / p95 (CPU, 1 image) | 39.9 / 48.4 ms | 216.2 / 224.1 ms |
| Throughput | **25.1 FPS** | 4.6 FPS |

Test accuracy is close to validation accuracy for both models (0.6753 vs 0.6665; 0.6581 vs 0.6600), which indicates that selecting the checkpoint on validation data did not overfit to it. The custom CNN is better on every metric: about 1.7 percentage points more accurate, 5.4 times faster, and half the size. Is a 1.7-point gap more than test-sample noise? Bootstrap 95% intervals (Section 3.7) for accuracy are 0.663–0.686 for the CNN and 0.647–0.670 for VGG16, and for macro-F1 0.644–0.673 and 0.625–0.654. These intervals overlap, but both models are scored on the same images and make correlated errors, so the appropriate comparison is a paired one. The paired accuracy difference is 1.71 points (95% CI 0.70 to 2.76) and the paired macro-F1 difference is 1.98 points (95% CI 0.67 to 3.28); both intervals exclude zero. Of the images on which the two models disagree, the CNN is right on 795 and VGG16 on 672 (both are right on 4,052 and both wrong on 1,659), and an exact McNemar test gives *p* = 0.0014. The custom CNN's advantage is therefore small but unlikely to be test-sample noise. Two limits apply. Each model was trained once, and the bootstrap resamples test images only, so it does not capture variation between training runs; another seed could move either model by an amount we have not measured. And a difference of under two points carries little practical weight next to the fivefold difference in speed.

**Table 5.** Bootstrap 95% confidence intervals and the paired comparison (2,000 resamples of the 7,178 test images; exact McNemar *p* = 0.0014; correct for the CNN only: 795 images, for VGG16 only: 672).

| Measure | Custom CNN [95% CI] | VGG16 [95% CI] | Paired difference, CNN − VGG16 [95% CI] |
|---|---|---|---|
| Accuracy | 0.675 [0.663, 0.686] | 0.658 [0.647, 0.670] | +1.71 points [+0.70, +2.76] |
| Macro-F1 | 0.659 [0.644, 0.673] | 0.640 [0.625, 0.654] | +1.98 points [+0.67, +3.28] |

![Figure 7. Accuracy and macro-F1 (left axis) and inference speed (right axis).](figures/comparison.png){4.4}

### 4.3 Per-class behaviour

**Table 6.** Per-class precision (P), recall (R) and F1 on the test set.

| Class | Support | CNN P | CNN R | CNN F1 | VGG16 P | VGG16 R | VGG16 F1 |
|---|---|---|---|---|---|---|---|
| angry | 958 | 0.625 | 0.601 | 0.613 | 0.589 | 0.599 | 0.594 |
| disgust | 111 | 0.624 | 0.703 | 0.661 | 0.742 | 0.622 | 0.676 |
| fear | 1024 | 0.531 | 0.487 | 0.508 | 0.574 | 0.331 | 0.420 |
| happy | 1774 | 0.889 | 0.865 | 0.877 | 0.828 | 0.885 | 0.856 |
| neutral | 1233 | 0.579 | 0.701 | 0.634 | 0.578 | 0.672 | 0.621 |
| sad | 1247 | 0.561 | 0.509 | 0.534 | 0.517 | 0.559 | 0.537 |
| surprise | 831 | 0.782 | 0.795 | 0.789 | 0.768 | 0.779 | 0.773 |

![Figure 8. Per-class F1 (left) and recall (right) on the test set.](figures/per_class_f1_recall.png)

Both models find *happy* (F1 0.86–0.88) and *surprise* (0.77–0.79) easiest and *fear* and *sad* hardest. Apart from *fear*, the two models are within about two points of each other on every class. *Fear* is the exception: VGG16's recall is 0.331 against the CNN's 0.487, so VGG16 misses about two thirds of the fearful faces. Since *fear* is one of the two alert emotions in the application, this matters for the intended use. *Disgust*, the rarest class, is recognised with recall 0.70 (CNN) and 0.62 (VGG16), which suggests that the capped class weights did their job. The main confusions of the custom CNN (row-normalised, Figure 9) are *sad* predicted as *neutral* (22%), *fear* as *sad* (17%), *disgust* as *angry* (14%) and *angry* as *sad* (13%), which are confusions between visually similar, low-intensity negative expressions.

![Figure 9. Row-normalised confusion matrices on the test set. Left: custom CNN. Right: VGG16.](figures/cnn_v1_confusion_matrix_normalized.png;figures/vgg16_v1_confusion_matrix_normalized.png)

### 4.4 Computational cost and the real-time application

On the development CPU the custom CNN runs at about 25 frames per second for the model alone, VGG16 at about 4.6. Live video also needs face detection, cropping and drawing, so end-to-end frame rates will be lower than these model-only figures. The 25 FPS figure is an upper bound for this pipeline on this laptop, not a measurement of the live application. Figure 10 shows what the application displays: (left) simulated webcam frames, produced by the application's own frame-processing code with the trained custom CNN on real test faces placed on a 640 × 480 canvas, used to illustrate the bounding box, probability bars, alert banner and no-face message in a controlled way; (right) one live capture from an actual webcam session with the custom CNN, confirming the pipeline runs end to end on a real camera feed at 4.5 FPS on the development laptop, correctly reading a neutral expression at 92% confidence.

![Figure 10. Left: output on simulated frames (bounding box, top emotion, probability bars, frame rate; the alert banner with its delay set to 0 s purely to illustrate it, default 3 s; and the no-face message). Right: one live webcam frame from an actual session, not simulated.](figures/realtime_demo.png;figures/realtime_live_screenshot.png){6.4}

### 4.5 Robustness under simulated driving conditions

Both models were evaluated on all 7,178 test images under each of the six conditions (Table 7, Figures 11 to 13). As a check on the harness, the `clean` row reproduces the separately measured test accuracies to within 0.3 points (0.676 vs 0.675 for the CNN; 0.655 vs 0.658 for VGG16).

**Table 7.** Accuracy and macro-F1 under simulated driving conditions, and the accuracy lost relative to `clean` (percentage points).

| Condition | CNN acc | CNN macro-F1 | CNN acc lost | VGG16 acc | VGG16 macro-F1 | VGG16 acc lost |
|---|---|---|---|---|---|---|
| clean | 0.676 | 0.657 | — | 0.655 | 0.635 | — |
| low_light | 0.286 | 0.137 | 39.0 | 0.218 | 0.115 | 43.7 |
| glare | 0.515 | 0.466 | 16.1 | 0.478 | 0.424 | 17.7 |
| motion_blur | 0.441 | 0.332 | 23.5 | 0.286 | 0.157 | 37.0 |
| occlusion | 0.444 | 0.368 | 23.2 | 0.396 | 0.273 | 25.9 |
| head_pose | 0.668 | 0.645 | 0.8 | 0.634 | 0.616 | 2.2 |

![Figure 11. One example face under each condition.](figures/conditions_grid.png)

![Figure 12. Accuracy and macro-F1 under each condition.](figures/robustness.png)

![Figure 13. Share of its own clean accuracy that each model keeps under each condition.](figures/robustness_retained.png){4.6}

Three findings stand out.

1. **The custom CNN degrades less than VGG16 under every corruption**, but by very different amounts: 13.5 points for motion blur, 4.8 for low light, 2.7 for occlusion, 1.6 for glare and 1.4 for head rotation. Only the motion-blur gap is large; the glare and head-rotation gaps are small and should not be over-read (no confidence intervals were computed for the corrupted results). Under motion blur VGG16 almost collapses (macro-F1 0.157) while the CNN keeps 65% of its clean accuracy.
2. **Low light is by far the most damaging condition for both models.** The CNN keeps 42% of its clean accuracy and VGG16 33%. To put the absolute numbers in context, always predicting the most frequent class (*happy*, 24.7% of the test set) would score 24.7%: the CNN's 28.6% is only about four points above that, and VGG16's 21.8% is below it. Their macro-F1 of 0.14 and 0.12 indicate that predictions concentrate on a few classes. We did not analyse which classes they collapse onto. Neither model, as trained, is usable in this condition.
3. **In-plane head rotation of ±20° is almost harmless** (0.8 and 2.2 points). This is unsurprising, because training augmentation already includes rotations of up to about 18°. By contrast, the augmentation brightness range (±20%) is far weaker than the ×0.4 dimming used here, and blur, glare and occlusion were not in the training augmentation at all.

These results are for one severity level per condition on simulated versions of clean 48-pixel web images, and each model was trained once.

### 4.6 What the networks look at (Grad-CAM)

**Correct predictions (Figure 14, left).** Both networks concentrate on facial features: the mouth for *happy*, the brows and eyes for *sad* and *fear*, the nose and mouth for *disgust*, and the eyes and mouth for *angry*. In none of the seven examples is the main evidence in the background. Two weaknesses are visible: the CNN's map for *surprise* is a vague horizontal band across the whole width, and both models look at the lower face rather than the eyes for *neutral*. The CNN's maps are blobbier because its last convolutional layer is only 6 × 6 at this input size; VGG16's 14 × 14 maps are sharper.

**Errors (Figure 14, right).** The custom CNN's hot spots in its errors mostly stay on the face (nose bridge, mouth, brows, cheeks), with two of six uncertain (hair and a region at the image edge). For VGG16, four of the six sampled errors have their main hot spot away from the expressive regions: the hair-line and a hand, a frame edge, a bottom corner, and a baseball cap. In one *surprise* error, VGG16 looks only at the mouth and ignores the wide-open eyes. This is consistent with VGG16's weaker *fear* recall, but it is a qualitative impression from a handful of hand-inspected images, and Grad-CAM shows where the gradient signal lies and not what the model "relies on".

![Figure 14. Grad-CAM. Left: one correctly classified example per emotion (original, custom CNN, VGG16). Right: misclassified examples, one per true class, for the custom CNN (top) and VGG16 (bottom).](figures/gradcam_combined.png)

---

## 5. Discussion

### 5.1 Accuracy versus cost (RQ1)

The main finding is negative for transfer learning: a 4.8 M-parameter network trained from scratch slightly but significantly exceeded fine-tuned VGG16 on accuracy and macro-F1 (Section 4.2) while being 5.4 times faster and half the size. We can offer hypotheses but have not tested them. (a) FER2013 images are 48-pixel grayscale faces, far from the colour, high-resolution ImageNet images on which VGG16's features were learned; the frozen-base stage reaching only about 50% is consistent with that. (b) Only the last convolutional block was fine-tuned, at a small learning rate; unfreezing more, or using a different schedule, might help. (c) VGG16 has three times the parameters of the CNN on about 24,000 training images, and its validation loss diverges from its training loss earlier. We did no hyper-parameter search for either model, and each was trained once, so the comparison is between two reasonable configurations and not between two tuned models. Both are also below the low-70s accuracies reported for heavily tuned FER2013 systems (Khaireddin & Chen, 2021), which is expected given the absence of tuning, test-time augmentation and ensembling.

### 5.2 Which model suits a vehicle

On the evidence here, the custom CNN is the better fit for an in-vehicle system: equal or better accuracy, a much lower computational cost, and a smaller file. VGG16's cost is a real disadvantage on modest hardware (4.6 FPS on our CPU), and it is the worse model on *fear*, an alert emotion. It is also the more robust of the two under every simulated driving condition (Section 4.5), although neither model is usable in low light as trained; a real system would need a near-infrared camera, low-light augmentation, or both. This conclusion is specific to FER2013-like data and to CPU inference; a GPU or accelerator would narrow the speed gap, though not the model-size gap.

### 5.3 Failure cases

The dominant errors are between visually similar, low-intensity negative expressions (*sad*↔*neutral*, *fear*→*sad*, *angry*→*sad*), which is also where human annotators disagree on FER2013. Some of the "errors" are probably label noise and not model failure. Under the simulated driving conditions the failures are of a different kind: the models lose most of their accuracy when the image loses information (darkness with noise, heavy blur, a covered fifth of the face), and lose very little when the face is merely rotated. We can only offer hypotheses for why VGG16 suffers more from blur (Section 4.5): its fine-tuned filters may depend more on fine detail, or the mismatch between its training inputs (enlarged 48-pixel images) and blurred ones may matter more for a network whose first layers were tuned on sharp ImageNet photographs. Neither was tested. What the results do show is that robustness gains are more likely to come from *training* on the relevant corruptions (brightness, blur, glare and occlusion augmentation) than from a larger network.

### 5.4 Pitfalls that produced plausible but wrong numbers

We report these because each one silently changed the results and each is easy to make.

1. **Augmentation applied after rescaling.** Keras' `RandomBrightness` and `RandomContrast` assume pixel values in [0, 255] by default. Applied after the images had been scaled to [0, 1], they added a brightness shift sized for the wrong range and wiped out the image (pixel values of 13–51 in a [0, 1] batch, several images entirely black). The first training run converged to 24% accuracy, roughly the share of the largest class, and was discarded. Augmenting the raw pixels first fixed it; the same ordering would have corrupted VGG16's mean-subtracted inputs.
2. **Aggressive class weights.** Balanced weights (disgust ≈ 9.4) destabilised training even after fix 1 (Section 3.6).
3. **Resizing method mismatch.** In the robustness code, enlarging 48-pixel images to 224 with OpenCV's area interpolation (nearest-neighbour-like when enlarging) instead of the bilinear resize used in training dropped VGG16 from 0.66 to 0.18 accuracy on *clean* faces. The bug was caught because the clean-condition accuracy did not match the separately measured test accuracy; the clean baseline is therefore a useful correctness check for any robustness harness.
4. **Stacked training histories.** A re-used run name made the logger append a third 40-epoch run to one history file; only the last block matched the saved model. Older blocks must not be quoted.

None of these was found by inspecting accuracy alone. Sanity checks that did find them were: inspecting an actual training batch, overfitting a single fixed batch (the custom CNN reaches 100% accuracy on one batch of 64 images both with and without dropout and L2, so its architecture and the labels are sound), and comparing a "clean" baseline against an independent measurement.

### 5.5 Limitations

- **Domain.** FER2013 consists of web images of mostly frontal or near-frontal faces, many of them posed, at 48 × 48 grayscale. An in-vehicle camera differs in resolution, viewpoint, spectrum (often near-infrared) and expression naturalness. We did not obtain an in-vehicle dataset such as KMU-FED (Jeong & Ko, 2018), so **this study does not establish performance on real drivers.**
- **Simulated degradations.** The robustness conditions are simple synthetic transformations of clean images.
- **Single runs.** Each model was trained once; seed-to-seed variance is unknown. The confidence intervals and the McNemar test in Section 4.2 reflect test-set sampling only, and the robustness results in Section 4.5 carry no intervals.
- **No tuning.** Neither model was tuned; the ranking might change with tuning.
- **Labels.** FER2013 labels are noisy, which caps attainable accuracy.
- **Grad-CAM** is qualitative, computed on a subset, with a handful of examples inspected by eye.
- **Real-time system.** Checked on simulated frames only, where individual predictions differ from the direct model output about one time in four because of framing (Section 3.9); no live evaluation with drivers or a real camera.
- **Latency** is for one laptop CPU and model inference only.

---

## 6. Ethics and privacy

*Consent.* Any live testing with volunteers requires their informed consent, and this study collected no new data about people. *Data minimisation.* The application processes frames on the local device, keeps no images, and writes only a timestamp, the predicted emotion, its confidence and the frame rate to a local file. *Validity.* Inferring a person's inner state from their face is contested: expressions vary across individuals and cultures and are not reliable indicators of emotion, so any deployed system should treat output as a weak, probabilistic cue, never as a fact about the driver, and should not be used to penalise drivers. *Bias.* FER2013 was collected from the web without controlled demographics; we did not measure performance across groups, so we cannot rule out uneven accuracy across age, gender or skin tone, and we make no fairness claim. *Misuse and regulation.* Emotion recognition can be repurposed for surveillance (for example by insurers or employers), and the regulatory treatment of emotion recognition is evolving; any real deployment would need legal review. *Alert design.* False alarms from a calming prompt are an annoyance; missed alerts (our *fear* recall is 0.49 for the better model) mean it cannot be relied on for safety.

---

## 7. Conclusion

On FER2013, a compact custom CNN built from VGG-style blocks (4.8 M parameters, trained from scratch) reached 67.5% test accuracy and 0.66 macro-F1, slightly but significantly exceeding a fine-tuned VGG16 (65.8%, 0.64; paired accuracy difference 1.7 points, 95% CI 0.7 to 2.8, McNemar *p* = 0.0014) while running 5.4 times faster and occupying half the space. Both models find *happy* and *surprise* easy and *fear* and *sad* hard, and VGG16 is notably worse at *fear*. Grad-CAM shows both attending to facial features when correct; VGG16's errors more often rest on off-face regions. Under simulated driving conditions the CNN degraded less than VGG16 in all five cases, by a large margin only for motion blur (13.5 points); low light collapsed both models to about the accuracy of always predicting the most frequent class, whereas in-plane head rotation was nearly harmless. The results support a small from-scratch model as the more practical choice for a CPU-bound in-vehicle system, subject to the important limitation that everything here was measured on web images and simulated degradations, not on drivers.

**Future work.** Evaluate on in-vehicle data (for example, Jeong and Ko's [2018] KMU-FED); train with the robustness corruptions as augmentation; repeat runs over several seeds with paired significance tests; tune both models; try near-infrared imagery; and run and report a live end-to-end evaluation with consenting volunteers.

---

## References

*Written from memory, so verify each entry (authors, year, venue, pages or DOI) before submission.*

Barsoum, E., Zhang, C., Canton Ferrer, C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowd-sourced label distribution. In *Proceedings of the ACM International Conference on Multimodal Interaction*.

Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond neural face detection on mobile GPUs. *arXiv*. https://arxiv.org/abs/1907.05047

Ekman, P., & Friesen, W. V. (1971). Constants across cultures in the face and emotion. *Journal of Personality and Social Psychology, 17*(2), 124–129.

Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., … Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. In *Neural Information Processing (ICONIP 2013)* (LNCS 8228, pp. 117–124). Springer.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the International Conference on Machine Learning* (pp. 448–456).

Jeon, M. (2016). Don't cry while you're driving: Sad driving is as bad as angry driving. *International Journal of Human–Computer Interaction, 32*(10), 777–790.

Jeong, M., & Ko, B. C. (2018). Driver's facial expression recognition in real-time for safe driving. *Sensors, 18*(12), Article 4270.

Khaireddin, Y., & Chen, Z. (2021). Facial emotion recognition: State of the art performance on FER2013. *arXiv*. https://arxiv.org/abs/2105.03588

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the International Conference on Learning Representations*.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems* (Vol. 25).

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*, 436–444.

Li, S., & Deng, W. (2022). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing, 13*(3), 1195–1215.

Mesken, J., Hagenzieker, M. P., Rothengatter, T., & de Waard, D. (2007). Frequency, determinants, and consequences of different drivers' emotions: An on-the-road study using self-reports, (observed) behaviour, and physiology. *Transportation Research Part F: Traffic Psychology and Behaviour, 10*(6), 458–475.

Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). AffectNet: A database for facial expression, valence, and arousal computing in the wild. *IEEE Transactions on Affective Computing, 10*(1), 18–31.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision, 115*(3), 211–252.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626).

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. In *Proceedings of the International Conference on Learning Representations*. https://arxiv.org/abs/1409.1556

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*, 1929–1958.

World Health Organization. (2023). *Global status report on road safety 2023*. https://www.who.int/publications/i/item/9789240086517

*Software:* TensorFlow/Keras, OpenCV, scikit-learn, NumPy, pandas, matplotlib, MediaPipe. All code, configuration, logs and figures are in the project repository (`docs/experiment_log.md` records every run and decision).
