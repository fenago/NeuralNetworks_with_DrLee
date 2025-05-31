# Chapter 7 – Neural Network Visualization

## 📑 Assumptions & Variable Definitions (for the demo visualizer)

| Symbol / Name | Type / Shape | Plain-language meaning |
|---------------|-------------|----------------------|
| model | PyTorch nn.Module | A trained feed-forward neural network (three dense layers in the example) |
| layer_outputs | dict[str, torch.Tensor] | Captured activations for any layer that registers a forward hook |
| sample_inputs | torch.Tensor [b, n] | A small batch (b rows) of environmental measurements fed through the net |
| weights | torch.Tensor | Parameter matrix for a given layer (e.g. model.fc1.weight) |
| gradients | torch.Tensor.grad | of the same parameter after a backward pass |
| pca_2d | numpy.ndarray [b, 2] | 2-D projection (via PCA) of hidden-layer activations, used for scatter plotting |
| plt | Matplotlib module | Standard plotting library for heat-maps & scatters |

**Context reminder** – nothing about training changes here; we only inspect a network that already works.
The logic focuses on **collect → condense → show**.

## 🔑 Algorithm — "Collect-Condense-Show" Visualization Pipeline

1. **Instrument the model**
   - A. Define a Python dict to hold named layer activations.
   - B. For each layer of interest, register a forward hook that stores output.detach().

2. **Pass data through the net**
   - Run a representative batch (or a single record) through model(sample_inputs).

3. **Plot weight heat-maps**
   - For every weight matrix, call plt.imshow(weights.cpu(), cmap='bwr', vmin=-|w|, vmax=|w|); add color-bar.

4. **Plot activation statistics**
   - Histogram or violin-plot each captured activation tensor to see sparsity / saturation.

5. **Visualize gradient flow (optional)**
   - After a dummy backward pass, extract .grad from each weight, then heat-map magnitudes or print layer-wise norms.

6. **Dimensionality reduction**
   - Apply PCA (or t-SNE if you prefer) to the hidden-layer captures → get 2-D array pca_2d.

7. **Scatter the reduced points**
   - Color-code by the true label (e.g., CROSS = red, WAIT = blue) to inspect class separation.

8. **Interpret & iterate**
   - Spot dead ReLU units, exploding gradients, or weight patterns → feed the findings back into model design.

## 💻 Reference Code — Minimal, Fully-Commented PyTorch Visualizer

```python
# ----------------------------------------------
# Chapter 7 – "see_inside.py"
# Visualize weights, activations, and PCA layout
# ----------------------------------------------
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# ---------- 1. Dummy trained network & data ----------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)   # (input ➜ hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)   # (hidden ➜ output)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

model = SimpleMLP()
model.load_state_dict(torch.load("animal_crossing_trained.pt"))  # load your own weights
model.eval()

sample_inputs = torch.tensor([[1., 0., 1.],
                              [0., 1., 1.],
                              [0., 0., 1.],
                              [1., 1., 1.],
                              [0., 1., 1.],
                              [1., 0., 1.]])

# ---------- 2. Hook machinery to capture activations ----------
layer_outputs = {}

def make_hook(name):
    def hook(_, __, output):
        layer_outputs[name] = output.detach().cpu()
    return hook

# Attach to desired layers
model.fc1.register_forward_hook(make_hook("layer_1"))
model.fc2.register_forward_hook(make_hook("layer_2"))  # not strictly needed but illustrative

# ---------- 3. Forward pass (captures populate layer_outputs) ----------
with torch.no_grad():
    preds = model(sample_inputs)

# ---------- 4. Weight heat-maps ----------
def plot_weight_heat(weight_tensor, title):
    w = weight_tensor.cpu().numpy()
    vmax = np.abs(w).max()
    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.imshow(w, cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="weight value")
    plt.xlabel("Output neuron index")
    plt.ylabel("Input neuron index")
    plt.tight_layout()

plot_weight_heat(model.fc1.weight, "Input ➜ Hidden weights (fc1)")
plot_weight_heat(model.fc2.weight, "Hidden ➜ Output weights (fc2)")

# ---------- 5. Activation histograms ----------
def plot_activation_hist(act, title):
    a = act.flatten().numpy()
    plt.figure()
    plt.title(title)
    plt.hist(a, bins=20, edgecolor='k')
    plt.xlabel("Activation value")
    plt.ylabel("Frequency")
    plt.tight_layout()

for name, act in layer_outputs.items():
    plot_activation_hist(act, f"Histogram of {name}")

# ---------- 6. PCA of hidden activations ----------
hidden = layer_outputs["layer_1"].numpy()  # shape: (b, 4)
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(hidden)

plt.figure(figsize=(4, 4))
colors = ["red" if p > 0.5 else "blue" for p in preds.numpy().flatten()]
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, edgecolors='k')
plt.title("PCA of hidden layer (red=CROSS, blue=WAIT)")
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.tight_layout()

# ---------- 7. Show everything ----------
plt.show()
```

## How the Code Reflects Chapter Concepts

| Code Section | Chapter 7 Idea → Implementation |
|-------------|--------------------------------|
| Hooks (make_hook) | Activation visualization – captures neuron outputs with one line of instrumentation |
| plot_weight_heat | Weight visualization – red/blue heat-map instantly shows excitatory vs inhibitory links |
| plot_activation_hist | Activation inspection – reveals dead or saturated neurons |
| PCA scatter | Dimensionality reduction – distills 4-D hidden vectors into a 2-D map that exposes class clustering |
| Color-coding by preds | Trust & explanation – lets a user see which regions of hidden space correspond to CROSS/WAIT decisions |
| Tight, 70-line script | Demonstrates the "Collect-Condense-Show" algorithm in practice |
