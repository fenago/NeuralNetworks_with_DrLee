# Chapter 4 – Gradient Descent

## 🎯 Core Concept: The Learning Cycle

Neural networks learn through a simple three-step cycle called **"Observe-Model-Refine"**:

1. **Observe**: Feed data through the network and make predictions
2. **Model**: Compare predictions to reality and measure the error
3. **Refine**: Adjust the network's weights to reduce future errors

## 📊 Measuring Mistakes: Mean Squared Error

Networks measure how wrong they are using **Mean Squared Error (MSE)**

### Why square the error? Three reasons:

- Always positive (being wrong is wrong, regardless of direction)
- Prioritizes big mistakes over small ones
- Mathematically convenient for calculations

## 🔥 "Hot and Cold" Learning

A simple but inefficient approach:

- Try adjusting weights up and down
- See which direction reduces error
- Move in that direction
- Like playing "hot and cold" to find something

## ⛰️ Gradient Descent: The Smart Way to Learn

The key algorithm that makes neural networks work:

- **Gradient** = The slope of the error surface (tells you which way is "downhill")
- **Descent** = Move in the direction that reduces error most efficiently
- Like a hiker in fog finding their way down a mountain by feeling the slope

### The Magic Formula:
```
gradient = 2 × (prediction - actual) × input
new_weight = old_weight - (learning_rate × gradient)
```

## 🎚️ Learning Rate: The Speed Control

Controls how big each step is when adjusting weights

- **Too small** = Learning is unnecessarily slow
- **Too large** = Overshoot and diverge (error gets worse!)
- Finding the right balance is crucial

## 💡 Key Insights

### The Rod and Box Analogy:

- Derivatives tell us how changing one thing (weight) affects another (error)
- Like connected rods - push one, the other moves proportionally

### Natural Learning Pattern:

- Gradient descent mirrors how nature learns (rivers carving stone, evolution, children learning to walk)
- It's a universal principle: adjust based on feedback

### Distributed Memory:

- Weights encode learned patterns across the entire network
- Similar to how our brains store memories in connection strengths

## 🌍 Environmental Applications

The chapter uses real-world examples:

- Predicting evaporation rates from temperature
- Forecasting soil moisture based on weather conditions
- Estimating CO2 absorption in different forest types

## 📝 Main Takeaway

Neural networks learn by:

1. Making predictions
2. Measuring how wrong they are
3. Using calculus (gradients) to figure out how to adjust weights
4. Taking small steps toward better predictions
5. Repeating thousands of times until accurate

**Think of it as**: A self-correcting system that gets better through practice, just like learning to ride a bike - you fall, adjust your balance, and gradually improve until it becomes second nature.

---

## 🧠 Theoretical Overview

### Core Concepts:

- **Learning through Errors**: Start wrong, measure error, correct
- **Gradient Descent**: Calculates how each weight should shift to cut error
- **MSE**: Squared error magnifies big mistakes
- **Learning-Rate α**: Step size—too big ⇒ diverge; too small ⇒ crawl
- **Analogy**: Like walking down a foggy mountain, following the steepest slope
- **Applications**: Powers climate, agriculture, and energy forecasting models

### Key Principles:

- Learns by minimising mean-squared error (MSE)
- Gradient gives direction & size
- Learning-rate α critical

## 📑 Assumptions & Variable Definitions

| Symbol | Type | Meaning |
|--------|------|---------|
| x | float | Input value |
| y | float | Target (ground-truth) value |
| w | float | Current weight |
| ŷ | float | Prediction (x × w) |
| error | float | Squared error (ŷ – y)² |
| gradient | float | Slope of error wrt w = 2 × (ŷ – y) × x |
| α (lr) | float | Learning rate – step size (first seen here) |
| epochs | int | How many training iterations |
| history | list[float] | Stored weights for visualising learning |

## 🔑 Algorithm: 1-Weight Gradient Descent Loop

1. Initialise weight **w** (often random).
2. Repeat for **N** iterations:
   - a. ŷ = x × w     # forward pass
   - b. error = (ŷ − y)²
   - c. gradient = 2 × (ŷ − y) × x     # derivative of MSE
   - d. w ← w − α × gradient     # step downhill
3. Return final **w**.

## 💻 Code (annotated loop)

```python
def train_single_weight(x, y, w_init=0.5, lr=1e-3, epochs=100):
    """
    Trains one weight using basic gradient descent.
    
    Args:
        x (float): Input feature.
        y (float): True target value.
        w_init (float): Starting weight.
        lr (float): Learning rate α.
        epochs (int): Number of training iterations.
    
    Returns:
        list[float]: History of weight values.
    """
    w = w_init
    history = [w]
    for _ in range(epochs):
        y_hat   = x * w
        gradient = 2 * (y_hat - y) * x  # ∂MSE/∂w
        w -= lr * gradient
        history.append(w)
    return history

# Demo
hist = train_single_weight(x=30, y=4.2, lr=1e-3, epochs=50)
print(f"Trained weight: {hist[-1]:.4f}")
