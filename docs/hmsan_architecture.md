# Hybrid Multi-Scale Attention Network (HMSAN)

This document formalizes the proposed **Hybrid Multi-Scale Attention Network (HMSAN)** for plant leaf disease detection. The design combines CNN-based local feature extraction and Transformer-based global context modeling, with explicit multi-scale processing, attention refinement, and optional domain adaptation.

## 1) Input Layer

- **Input shape:** `224 × 224 × 3` (RGB leaf image)
- **Preprocessing:**
  - Pixel normalization
  - Data augmentation (e.g., rotation, flipping, brightness variation)

## 2) CNN Backbone (Local Feature Extraction)

- **Candidate backbones:** `ResNet50` or `EfficientNet-B0`
- **Role:** extract local spatial cues such as edges, texture patterns, and color variations
- **Output tensor:**
  - `F1 ∈ R^(H×W×C)`

## 3) Multi-Scale Feature Extraction Module

Apply parallel convolution branches over the backbone feature map:

- `1×1` convolution branch
- `3×3` convolution branch
- `5×5` convolution branch

Concatenate branch outputs to capture fine-to-coarse context.

- **Output tensor:** `F2`

## 4) Vision Transformer (Global Context Learning)

- Convert feature maps into patch tokens
- Apply linear patch embedding + positional encoding
- Pass through Transformer encoder blocks:
  - Multi-head self-attention
  - Feed-forward network (FFN)

- **Output tensor:** `F3` (global representation)

## 5) Attention Refinement (CBAM + SE)

Use attention blocks to emphasize disease-relevant information:

- **Channel attention:** prioritize informative channels
- **Spatial attention:** highlight lesion/infection regions

- **Output tensor:** `F4` (refined feature map)

## 6) Feature Fusion Layer (Core Contribution)

Fuse heterogeneous features:

- CNN local features: `F1`
- Multi-scale features: `F2`
- Transformer global features: `F3`

Suggested strategy:

1. Align tensor sizes/channels
2. Concatenate feature vectors/maps
3. Project via fully connected (or `1×1` projection + FC)

- **Output tensor:** `F_fused`

## 7) Domain Adaptation Layer (Advanced Contribution)

To improve cross-dataset and real-world generalization:

- Add a **Gradient Reversal Layer (GRL)** before a domain classifier head
- Train adversarially so shared features become domain-invariant

This layer is optional for single-domain settings.

## 8) Classification Head

Recommended head:

- `Dense(512) → ReLU → Dropout`
- `Dense(N_classes) → Softmax`

- **Output:** class probabilities for plant disease labels

## 9) Explainability Module

Integrate post-hoc explainability:

- Grad-CAM and/or attention rollout maps

- **Output:** heatmaps over infected leaf regions for interpretability

## 10) Objective Function

Primary loss:

- Cross-entropy classification loss

Optional multi-objective extension:

- `L_total = L_cls + λ * L_domain`

where `L_domain` is adversarial domain loss and `λ` controls adaptation strength.

## 11) Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## End-to-End Flow

`Input → CNN Backbone → Multi-Scale Module → ViT Encoder → Attention Module → Feature Fusion → (Optional) Domain Adaptation → Classification Head → Prediction`

## Claimed Novel Contributions

1. Hybrid CNN + Transformer modeling for local-global complementarity
2. Multi-scale feature extraction with parallel kernels
3. Attention-guided refinement (CBAM + SE)
4. Domain adaptation for cross-domain robustness
5. Explainable AI integration via heatmaps
