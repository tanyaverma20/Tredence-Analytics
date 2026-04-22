# Tredence-Analytics

# Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study Submission

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

---

## Problem Statement

Design and implement a neural network that **learns to prune itself during training** — not as a post-training step, but dynamically, by associating each weight with a learnable gate parameter that can shut off unnecessary connections on its own.

---

## Core Idea

```
Normal Layer:     output = weight × input
Prunable Layer:   output = (weight × gate) × input
```

Each weight has a companion **gate** — a learnable scalar in (0, 1) produced by a sigmoid function. An L1 penalty on all gates during training pushes unimportant ones toward zero, effectively removing those weights from the network.

---

## Architecture

```
Input (32×32×3)
        ↓
Conv2d(3→32) + BatchNorm + ReLU + MaxPool
        ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool
        ↓  [4096 features]
PrunableLinear(4096 → 512) + ReLU + Dropout(0.3)
        ↓
PrunableLinear(512 → 256)  + ReLU + Dropout(0.3)
        ↓
PrunableLinear(256 → 128)  + ReLU
        ↓
PrunableLinear(128 → 10)
        ↓
Output (10 classes)
```

Each `PrunableLinear` layer contains:
- `weight` — standard learnable weight matrix
- `gate_scores` — learnable parameters, same shape as weight
- `gates = sigmoid(gate_scores)` — soft on/off switch per weight

---

## Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

where SparsityLoss = Σ sigmoid(gate_scores)  [across all layers]
```

**Why L1 encourages sparsity:**

Unlike L2 which applies a gradient proportional to magnitude (letting small values persist), L1 applies a **constant gradient** regardless of gate size. Even a gate at 0.001 keeps getting pushed toward 0. The optimizer only keeps a gate open if its contribution to accuracy outweighs its L1 penalty cost — forcing a sparse solution. Combined with sigmoid's asymptote at 0, gates are pushed toward binary on/off behaviour, achieving true structural sparsity.

---

## Training Configuration

| Component | Detail |
|-----------|--------|
| Optimizer | Adam (lr = 1e-3) |
| Scheduler | CosineAnnealingLR |
| Early Stopping | Patience = 10 |
| Max Epochs | 50 |
| Dropout | 0.3 between FC layers |

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) | Layer-wise Sparsities |
|:----------:|:-----------------:|:------------:|:---------------------:|
| 1e-5 (Low) | 81.45 | 51.88 | [52.0%, 57.9%, 21.9%, 6.5%] |
| 1e-4 (Med) | 79.75 | 74.12 | [74.7%, 72.2%, 49.4%, 13.0%] |
| 1e-3 (High)| 74.57 | 99.11 | [99.7%, 93.6%, 85.4%, 32.0%] |

**Key finding:** At λ=1e-4, the network pruned **74% of its weights** while losing only **~1.7% accuracy** — confirming those connections were genuinely redundant.

---

## Gate Value Distributions

The gate histograms for each λ reveal the pruning mechanism in action:

- **λ=1e-5:** Gates spread across 0–1, minimal pruning pressure — model stays dense
- **λ=1e-4:** Bimodal distribution — spike near 0 (pruned weights) + cluster near 1 (task-critical weights). This is the ideal signature of self-pruning working correctly.
- **λ=1e-3:** Near-total collapse to 0, only a fraction of weights survive

---

## Analysis

| λ | Behaviour | Best Use Case |
|---|-----------|---------------|
| 1e-5 | ~52% sparsity, highest accuracy | Accuracy-critical tasks |
| 1e-4 | ~74% compression, ~1.7% accuracy drop | **Best practical tradeoff** |
| 1e-3 | ~99% compression, ~7% accuracy drop | Memory-constrained deployment |

The monotonic increase in sparsity with λ confirms the self-pruning mechanism works exactly as designed. Early layers (closer to input) prune more aggressively than the final classification layer — consistent with the finding that high-level task-specific features are harder to discard.

---

## How to Run

**Option 1 — Google Colab (Recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v9lFtv6j4zOtDnDcg2E7NiUXMc6JAEUz?usp=sharing)

**Option 2 — Local**

```bash
git clone https://github.com/tanyaverma20/self-pruning-neural-network
cd self-pruning-neural-network
pip install torch torchvision matplotlib numpy
jupyter notebook Self_Pruning_NN_Model.ipynb
```

---

## 📁 Repository Structure

```
self-pruning-neural-network/
├── Self_Pruning_NN_Model.ipynb    # Full implementation + outputs
├── gate_distributions.png         # Gate value distribution plots
└── README.md                      # This file
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model implementation, autograd |
| Torchvision | CIFAR-10 dataset loading |
| Matplotlib | Gate distribution visualization |
| Google Colab | T4 GPU training environment |

---

## 👩‍💻 Author

**Tanya Verma**  
B.E. Computer Engineering — Thapar Institute of Engineering and Technology  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-tanyaverna-blue)](https://www.linkedin.com/in/tanyaverma/)
[![GitHub](https://img.shields.io/badge/GitHub-tanyaverma20-black)](https://github.com/tanyaverma20)
