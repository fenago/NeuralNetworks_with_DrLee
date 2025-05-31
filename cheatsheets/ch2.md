# Chapter 2 – Prediction Basics

## 🧠 Theoretical Overview

### Core Concepts:

- **Making Predictions**: Neural networks transform input data (e.g., temperature, rainfall) into predictions (e.g., plant growth) with simple math
- **Weights and Inputs**: Each prediction equals the sum of (input × weight) pairs
- **Error and Correction**: The gap between prediction and reality points the way to better weights

### Key Principles:

- Predictions = inputs × weights (simple multiplication and sum)
- Observed error guides weight corrections
- Observe → Model → Refine is the universal learning loop

### The Learning Paradigm:

1. **Observe** – collect data
2. **Model** – predict from data
3. **Refine** – tweak weights for accuracy

**Why It Works**: Repeated error-driven refinement mimics natural learning and steadily improves accuracy

## 📑 Assumptions & Variable Definitions

| Symbol | Type | Meaning (spoken-out) |
|--------|------|---------------------|
| x | float | One environmental measurement (e.g., temperature) |
| w | float | Weight showing how strongly x influences the prediction |
| ŷ | float | Model's predicted value (reads "y-hat") |
| y | float | True/observed value, used only if we calculate an error |

## 🔑 Algorithm: Single-Input Forward Prediction

1. Collect one input value **x**.
2. Store a weight **w** that represents how strongly **x** should influence the output.
3. Predict using **ŷ = x × w**.
4. (Optional) Compare to the actual target **y** to compute error **(ŷ − y)**.

## 💻 Code (fully commented)

```python
# ------------------------------------------
# Simple 1-input, 1-weight forward predictor
# ------------------------------------------
def single_feature_predict(x: float, w: float) -> float:
    """
    Predicts an output given one feature and one weight.
    
    Args:
        x (float): The input measurement (e.g., temperature).
        w (float): The connection strength / learned weight.
    
    Returns:
        float: The predicted value ŷ.
    """
    # Core of Chapter 2: one multiplication → one prediction
    return x * w

# Example usage
if __name__ == "__main__":
    temperature = 30.0       # °C  (input)
    weight_temp = 0.2        # learned weight
    prediction = single_feature_predict(temperature, weight_temp)
    print(f"Predicted value: {prediction:.2f}")
