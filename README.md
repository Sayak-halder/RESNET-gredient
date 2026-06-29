# 🧠 ResNet Anatomy Lab: Dissecting Skip Connections & Gradient Flow

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Everyone imports `torchvision.models.resnet50`. But how many of us actually understand the internal math of the Bottleneck block, or can visually prove *why* skip connections solve the vanishing gradient problem?

This project is a **ResNet Anatomy Lab**. I built ResNet from scratch in PyTorch to dissect its internal mechanics, modify its DNA with Squeeze-and-Excitation (SE) attention, and empirically prove the core thesis of the original 2015 paper using custom gradient hooks.

---

##  The Proof: Visualizing Gradient Flow

The entire premise of the ResNet paper is that identity mappings (skip connections) allow gradients to flow directly backward through the network. To prove this, I wrote custom PyTorch hooks to capture the **L2 norm of the gradients** at each major layer during the backward pass.

![Gradient Flow Comparison](outputs/charts/gradient_flow_comparison.png)

### How to read this graph:
*   **The Baseline (ResNet Thesis):** Both lines show a stable, gradual decline rather than a catastrophic drop to zero. This empirically proves that the skip connections ($F(x) + x$) are successfully routing gradients backward, preventing the vanishing gradient problem that plagues plain networks.
*   **The Modification (SE-ResNet vs. Standard):** Notice that the **SE-ResNet (green)** consistently maintains a higher gradient norm than the **Standard ResNet (blue)**. 
*   **Why does this happen?** The Squeeze-and-Excitation mechanism dynamically recalibrates channel-wise feature responses. By amplifying the most informative feature channels, it creates stronger, more robust gradient signals flowing backward through the network, resulting in higher overall gradient norms.

---

## 🧬 Internal Modifications: SE-ResNet

Standard ResNet treats all feature channels equally. I modified the internal DNA of the Bottleneck block to create an **SE-Bottleneck (Squeeze-and-Excitation)**.

1.  **Squeeze:** Global Average Pooling compresses the spatial dimensions of the feature map.
2.  **Excitation:** A small Multi-Layer Perceptron (MLP) learns a set of channel-wise weights.
3.  **Scale:** These weights are multiplied back onto the original feature map, amplifying important features and suppressing useless ones *before* the residual addition.

---

## ️ The Internal Math: Handling Dimension Mismatches

The hardest part of building ResNet from scratch is the skip connection. When spatial dimensions shrink (stride=2) or channel depth increases, $F(x) + x$ breaks because the tensors have different shapes. 

I manually implemented the $1 \times 1$ convolution projection in the `downsample` block to match the dimensions of the identity mapping, ensuring the addition is mathematically valid at every stage.

---

## 📂 Project Structure

```text
resnet-anatomy-lab/
├── README.md
├── requirements.txt
├── main.py                  # Entry point
├── config.py                # Hyperparameters
├── data.py                  # Data loading and transforms
├── models.py                # The "DNA" (Blocks and ResNet Engine)
├── analysis.py              # Gradient hooks and plotting
├── trainer.py               # Training loop
└── outputs/
    └── charts/              # Where the gradient graph is saved