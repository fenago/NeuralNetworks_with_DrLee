# Chapter 5 – Generalised Gradient Descent

## 🧠 Core Concept: From Solo to Symphony

Neural networks gain their power by using multiple weights working together, just like ecosystems function through interconnected species. Real-world problems (like predicting forest fire risk) need multiple inputs because single measurements rarely tell the whole story.

## 📊 Three Network Architectures

### 1. Multiple Inputs → Single Output

- **Example**: Temperature, humidity, wind speed → Fire risk score
- Each input has its own weight
- Prediction = Sum of (input × weight) for all inputs

### 2. Single Input → Multiple Outputs

- **Example**: Atmospheric pressure → Precipitation, temperature change, wind speed
- One weight per output prediction
- Each output calculated independently

### 3. Multiple Inputs → Multiple Outputs

- **Example**: Temperature, humidity, soil → Plant growth, water needs, pest risk
- Creates a weight matrix (inputs × outputs)
- Most realistic for environmental monitoring

## 🎯 How Multi-Weight Learning Works

### The Three-Step Dance:

1. **Predict**: Calculate weighted sum of inputs
2. **Compare**: Find error between prediction and actual
3. **Learn**: Update each weight based on:
   - The shared error signal
   - Its specific input's contribution
   - Learning rate (prevents overshooting)

**Key Insight**: Weights connected to larger inputs receive larger updates. This is why data normalization is crucial - it prevents variables with naturally larger values from dominating the learning process.

## 🔍 What Weights Really Mean

Weights encode discovered patterns and relationships:

- **Positive weight** = Variables increase together
- **Negative weight** = Inverse relationship
- **Near-zero weight** = Little/no relationship

## 💡 The Dot Product Revelation

Neural networks work by computing dot products - measuring similarity between input patterns and learned weight patterns:

- **High dot product** = Input matches the pattern weights are looking for
- **Low dot product** = Poor match

Think of weights as "detectors":

- **Forest detector**: High weights for tree density, humidity, ground cover
- **Desert detector**: High weights for temperature, low for humidity

## 🎨 Visualizing Learning

Weight visualization reveals what patterns the network discovered:

- **In image processing**: Weights show which pixel patterns activate each detector
- **In environmental monitoring**: Weights reveal which sensors matter most for each prediction

## 📦 Batch Learning

Instead of updating after each example:

1. Process multiple examples
2. Average the errors
3. Update weights once

**Benefits**: More stable learning, better for seasonal patterns, reduces noise

## 🌍 Real-World Impact

Multi-weight networks excel at:

- **Pattern recognition**: Finding similar environmental conditions
- **Anomaly detection**: Identifying unusual measurements
- **Relationship discovery**: Revealing hidden correlations in ecological data
- **Feature importance**: Showing which sensors are crucial vs redundant

## 🔑 Key Takeaway

Neural networks achieve their power through the coordinated adjustment of many simple weights. Each weight alone is basic, but together they form a system capable of modeling complex environmental relationships - transforming raw sensor data into meaningful ecological insights.

---

## 🧠 Theoretical Overview

### Core Concepts:

- **Many Inputs & Outputs**: Real tasks need lots of features/predictions
- **Weight Updates**: Each weight moves in proportion to its input and the error
- **Matrix Ops**: Predictions = X · W
- **Dot-Product Matching**: High dot products flag strong input–pattern matches
- **Interpreting Weights**: Reveals hidden ecological relationships
- **Batch Gradient Descent**: Uses many samples at once to stabilise learning
- **Ecological Insight**: Models don't just predict—they uncover what matters

### Key Principles:

- Handles many inputs and outputs (matrix view)
- Weight update: ΔW = xᵀ·δ generalises
- Batch learning stabilises updates

## 📑 Assumptions & Variable Definitions

| Symbol | Shape | Meaning |
|--------|-------|---------|
| X | (b, n) | Batch of b input vectors (rows) each of length n |
| Y | (b, m) | Corresponding batch of targets |
| W | (n, m) | Weight matrix to learn |
| P | (b, m) | Predictions X @ W |
| Δ (Delta) | (b, m) | Prediction errors P – Y |
| G | (n, m) | Gradient Xᵀ @ Δ / b |
| α (lr) | float | Learning rate (same concept as Ch 4) |
| b | int | Batch size (rows in X) |
| n | int | # inputs; m – # outputs |
| epochs | int | Training loops |

## 🔑 Algorithm: Batch Gradient Descent with Matrices

### Inputs:

- Feature matrix **X** shape (b, n) (batch of b examples).
- Target matrix **Y** shape (b, m).
- Weight matrix **W** shape (n, m).

### Steps:

1. **Forward**: P = X·W.
2. **Error / deltas**: Δ = P − Y   (shape (b, m)).
3. **Gradient**: G = Xᵀ·Δ / b   (shape (n, m)).
4. **Update**: W ← W − α × G.
5. Loop over epochs.

## 💻 Code (NumPy, full batch)

```python
import numpy as np

def train_matrix(X, Y, lr=1e-3, epochs=500):
    """
    Batch gradient descent for multi-input, multi-output linear net.
    
    Args:
        X (np.ndarray): (b, n) input batch.
        Y (np.ndarray): (b, m) target batch.
        lr (float): Learning rate (α).
        epochs (int): Training iterations.
    
    Returns:
        np.ndarray: Trained weight matrix W (n, m).
    """
    b, n = X.shape
    m    = Y.shape[1]
    rng  = np.random.default_rng(0)
    W    = rng.uniform(-0.1, 0.1, size=(n, m))   # initialise
    
    for _ in range(epochs):
        P      = X @ W            # forward
        Delta  = P - Y            # error matrix
        G      = (X.T @ Delta) / b
        W     -= lr * G           # gradient step
    return W

# Demo dataset: 3 inputs ➜ 2 outputs
X_data = np.array([[28, 65, 40],
                   [22, 80, 35],
                   [30, 55, 50]], dtype=float)
Y_data = np.array([[6.5, 4.2],
                   [7.0, 5.1],
                   [6.2, 4.8]], dtype=float)

W_trained = train_matrix(X_data, Y_data, lr=1e-4, epochs=1000)
print("Final weight matrix:\n", W_trained)
