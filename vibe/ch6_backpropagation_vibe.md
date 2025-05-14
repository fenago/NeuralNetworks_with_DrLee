# Vibe Coding Assignment: Backpropagation with AI-Powered Development

> *"Just as a pebble's ripple travels backward across a pond when it strikes the shore, so too does error travel backward through a neural network, refining every connection along its path."* u2014Dr. Ernesto Lee

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Your Assignment: Neural Network Error Attribution Visualizer

In this Vibe Coding assignment, you'll use AI-powered development tools to build an interactive visualization that demonstrates backpropagationu2014the fundamental algorithm that makes deep learning possible. You'll focus on how neural networks learn complex patterns by attributing error and adjusting weights across multiple layers.

### Project Overview

You'll create an interactive web application that:

1. Implements a multi-layer neural network for an environmental prediction problem
2. Visualizes both the forward and backward passes of backpropagation
3. Shows how error is attributed through the network using the Chain Rule
4. Demonstrates the role of nonlinear activation functions in creating representational power
5. Includes ecological examples that demonstrate pattern recognition where direct correlations don't exist

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select a Problem Domain

Choose one of these ecological pattern recognition challenges for your neural network visualizer:

1. **The Nature Crossing Problem**: Implement the animal crossing predictor from Chapter 6 that learns patterns from light combinations
2. **Ecosystem Health Prediction**: Create a network that predicts ecosystem health from multiple environmental factors
3. **Species Identification**: Build a network that identifies species based on multiple characteristics
4. **Climate Pattern Recognition**: Develop a network that identifies climate patterns from weather data where no single variable provides a clear signal

## Vibe Coding Prompts Guide

Below are example natural language prompts you can use with your chosen AI coding tool. These are starting pointsu2014refine and expand them as you develop your project.

### Project Setup Prompts

```
Create a new web application that visualizes backpropagation in neural networks. The application should demonstrate both forward and backward passes and show how error is propagated backward through the network using the chain rule.
```

```
Set up the project structure with these components: 1) a neural network visualization panel, 2) an error propagation visualization, 3) a training control panel, and 4) a data playground where users can experiment with different input patterns.
```

### Neural Network Architecture Prompts

```
Create a visualization of a neural network with 3 input nodes, 4 hidden nodes, and 1 output node. Use SVG or Canvas to draw the network with circles for nodes and lines for connections. The visualization should update during training to show activations and error propagation.
```

```
Implement a mermaid.js diagram generator that creates an architecture diagram of the current neural network. The diagram should show layer sizes and include proper styling.
```

Example mermaid.js diagram to include:

```
flowchart LR
    subgraph "Input Layer"
        I1["Input 1"]
        I2["Input 2"]
        I3["Input 3"]
    end
    
    subgraph "Hidden Layer"
        H1["Hidden 1"]
        H2["Hidden 2"]
        H3["Hidden 3"]
        H4["Hidden 4"]
    end
    
    subgraph "Output Layer"
        O1["Output"]
    end
    
    I1 --> H1 & H2 & H3 & H4
    I2 --> H1 & H2 & H3 & H4
    I3 --> H1 & H2 & H3 & H4
    
    H1 --> O1
    H2 --> O1
    H3 --> O1
    H4 --> O1
    
    style I1 fill:#bbdefb,stroke:#333,stroke-width:1px
    style I2 fill:#bbdefb,stroke:#333,stroke-width:1px
    style I3 fill:#bbdefb,stroke:#333,stroke-width:1px
    style H1 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style H2 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style H3 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style H4 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style O1 fill:#f8bbd0,stroke:#333,stroke-width:1px
```

### Forward Propagation Prompts

```
Implement the forward propagation function for a neural network with one hidden layer using ReLU activation for the hidden layer. Visualize the data flow through the network with animations showing how inputs transform through each layer.
```

```
Create a step-by-step visualization of the forward pass that shows:
1. The initial input values
2. The weighted sum calculation for each hidden node
3. The activation function application
4. The weighted sum calculation for the output node
5. The final prediction
```

Example forward pass visualization to include:

```
flowchart TB
    subgraph "Forward Pass"
        A1["Inputs: [1, 0, 1]"] --> A2["Hidden Layer Weighted Sum:\n[0.5, -0.3, 0.8, 0.1]"]  
        A2 --> A3["After ReLU:\n[0.5, 0, 0.8, 0.1]"]  
        A3 --> A4["Output Weighted Sum:\n0.7"]  
        A4 --> A5["Final Prediction:\n0.7"]  
    end
    
    style A1 fill:#bbdefb,stroke:#333,stroke-width:1px
    style A2 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style A3 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style A4 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style A5 fill:#f8bbd0,stroke:#333,stroke-width:1px
```

### Backpropagation Prompts

```
Implement backpropagation for the neural network. Visualize how the error at the output propagates backward through the network, showing the calculation of deltas and weight updates at each layer.
```

```
Create an animated visualization of the chain rule in action during backpropagation. Show how each component of the chain (dError/dOutput, dOutput/dHidden, dHidden/dWeight) contributes to the final weight updates.
```

Example backpropagation visualization to include:

```
flowchart TB
    subgraph "Backward Pass"
        B1["Output Error:\n0.3"] --> B2["Output Delta:\n0.3"]  
        B2 --> B3["Hidden Layer Deltas:\n[0.09, 0, 0.15, 0.03]"]  
        B3 --> B4["Input-Hidden Weight Updates"]  
        B2 --> B5["Hidden-Output Weight Updates"]  
    end
    
    style B1 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style B2 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style B3 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style B4 fill:#bbdefb,stroke:#333,stroke-width:1px
    style B5 fill:#e1bee7,stroke:#333,stroke-width:1px
```

### Chain Rule Visualization Prompts

```
Create a visualization that demonstrates the chain rule in neural networks. Show how the derivatives at each layer combine to determine the overall gradient for each weight, similar to tracing environmental impacts through an ecosystem.
```

```
Visualize the chain rule calculation for a specific weight in the network. Show the individual components (dError/dOutput, dOutput/dHidden, dHidden/dWeight) and how they multiply together to determine the weight update.
```

Example chain rule visualization to include:

```
flowchart LR
    subgraph "Chain Rule Components"
        C1["dError/dOutput:\n0.3"] --> C4["×"]  
        C2["dOutput/dHidden:\n0.6"] --> C4  
        C4 --> C5["="]  
        C5 --> C6["dError/dHidden:\n0.18"]  
        C6 --> C7["×"]  
        C3["dHidden/dWeight:\n1.0"] --> C7  
        C7 --> C8["="]  
        C8 --> C9["dError/dWeight:\n0.18"]  
    end
    
    style C1 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style C2 fill:#e1bee7,stroke:#333,stroke-width:1px
    style C3 fill:#bbdefb,stroke:#333,stroke-width:1px
    style C6 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C9 fill:#ffcc80,stroke:#333,stroke-width:1px
```

### Nonlinearity Demonstration Prompts

```
Create an interactive demonstration that shows why nonlinearity is essential in neural networks. Allow users to toggle between linear and nonlinear (ReLU) activation functions and see how it affects the network's ability to learn non-linearly separable patterns.
```

```
Implement a visualization that shows how ReLU activation introduces nonlinearity to the network. Include a comparison of what happens when trying to solve the XOR or animal crossing problem with and without nonlinear activation functions.
```

Example nonlinearity visualization to include:

```
flowchart TB
    subgraph "Linear Network Prediction"
        L1["Input Pattern: [1, 0, 1]"] --> L2["Prediction: 0.5"]  
        L3["Input Pattern: [0, 1, 1]"] --> L4["Prediction: 0.5"]  
        L5["Linear Network Cannot Distinguish Patterns"]  
    end
    
    subgraph "Nonlinear Network Prediction"
        N1["Input Pattern: [1, 0, 1]"] --> N2["Prediction: 0.1"]  
        N3["Input Pattern: [0, 1, 1]"] --> N4["Prediction: 0.9"]  
        N5["Nonlinear Network Can Distinguish Patterns"]  
    end
    
    style L5 fill:#ffcdd2,stroke:#333,stroke-width:2px
    style N5 fill:#c8e6c9,stroke:#333,stroke-width:2px
```

### Training Visualization Prompts

```
Implement an interactive training visualization that shows the network learning over time. Display the error curve, accuracy metrics, and how the network's predictions improve with each training epoch.
```

```
Create a visualization that demonstrates overfitting and the train/validation split. Show how the network's performance on training and validation data diverges when overfitting occurs.
```

## Sample Project: Nature Crossing Visualizer

Here's an example workflow for building the Nature Crossing Visualizer from Chapter 6:

### 1. Project Setup

```
Create a new web application for visualizing neural network learning on the Nature Crossing Problem from Chapter 6. The app should demonstrate how a neural network with a hidden layer can learn patterns from light combinations (represented as [1, 0, 1], [0, 1, 1], etc.) to predict animal crossing behavior (0 for WAIT, 1 for CROSS).
```

### 2. Data Setup

```
Create a dataset for the Nature Crossing Problem with these training examples:
- Light Pattern [1, 0, 1] → Animal Behavior: 0 (WAIT)
- Light Pattern [0, 1, 1] → Animal Behavior: 1 (CROSS)
- Light Pattern [0, 0, 1] → Animal Behavior: 0 (WAIT)
- Light Pattern [1, 1, 1] → Animal Behavior: 1 (CROSS)
- Light Pattern [0, 1, 1] → Animal Behavior: 1 (CROSS)
- Light Pattern [1, 0, 1] → Animal Behavior: 0 (WAIT)

Add a visualization that shows these patterns as traffic lights with red, yellow, and blue lights that can be ON or OFF.
```

### 3. Network Architecture

```
Create a visualization of a neural network with 3 input nodes (one for each light), 2 hidden nodes, and 1 output node (animal behavior). Use SVG to render the network with circles for nodes and lines for connections. Also generate a mermaid.js diagram showing the same architecture.
```

### 4. Forward Propagation Implementation

```
Implement forward propagation for the Nature Crossing neural network. Use matrix multiplication to calculate the weighted sum at each layer and apply ReLU activation to the hidden layer. Create a step-by-step visualization that shows how a light pattern flows through the network to produce a prediction.
```

### 5. Backpropagation Implementation

```
Implement backpropagation for the Nature Crossing neural network. Create a visualization that shows:
1. How the error is calculated at the output
2. How this error is propagated back to the hidden layer using the chain rule
3. How weight updates are calculated for both layers

Include animation that shows the error flowing backward through the network.
```

### 6. Training Process

```
Create an interactive training interface where users can:
1. Start/pause training
2. Adjust the learning rate
3. See the error graph updating in real-time
4. Watch the network connections update as weights change
5. Test the network with new light patterns after training

Include a "step-by-step" mode where users can advance through individual forward and backward passes to understand the learning process.
```

### 7. Pattern Recognition Demonstration

```
Add a feature that demonstrates why a single-layer network cannot solve this problem but a network with a hidden layer can. Show how the hidden nodes learn to detect specific light combinations, creating useful intermediate representations that make the problem linearly separable.
```

## Advanced Features to Try

After completing the basic implementation, try using Vibe Coding to add these advanced features:

1. **Multiple Activation Functions**: Allow users to switch between different activation functions (ReLU, Sigmoid, Tanh) and see how they affect learning

2. **Batch vs. Stochastic Gradient Descent**: Add options for different training approaches and visualize the differences

3. **Learning Rate Scheduler**: Implement and visualize a learning rate scheduler that adjusts the learning rate during training

4. **Regularization**: Add L1 or L2 regularization and visualize its effect on preventing overfitting

5. **Hidden Layer Analysis**: Create visualizations that help interpret what patterns each hidden node has learned to detect

## Create Your Own Backpropagation Visualizer

Now it's your turn to create an application that demonstrates backpropagation on an environmental problem you find interesting!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an interactive application that helps others understand how neural networks learn complex patterns through backpropagation.

Some ideas to inspire you:

- A coral reef health predictor that uses backpropagation to learn from complex patterns of temperature, acidity, and pollutant levels
- A wildlife migration predictor that learns to identify complex patterns in environmental cues
- A climate anomaly detector that identifies unusual weather patterns that no single measurement can detect
- A sustainable agriculture advisor that learns complex relationships between soil, weather, and crop performance

Remember to incorporate visualizations of both the forward and backward passes, and to use mermaid.js diagrams to help explain network architecture and the chain rule!

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A document containing the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Network Architecture Diagram**: A mermaid.js diagram showing your neural network architecture

4. **Chain Rule Visualization**: A visualization showing how the chain rule works in your backpropagation implementation

5. **Demo**: A link to a live demo or a video showcasing your application in action

6. **Reflection**: A short reflection (2-3 paragraphs) on your experience using Vibe Coding with AI tools to build this application

## Tips for Effective Vibe Coding

- **Break It Down**: Backpropagation is complex, so break your prompts into smaller, focused requests rather than trying to implement everything at once

- **Visualize Each Component**: Create separate visualizations for forward propagation, error calculation, and backward propagation

- **Use Metaphors**: When describing backpropagation to the AI, use environmental metaphors like those in Chapter 6 to guide the implementation

- **Start Small**: Begin with a simple network (3 inputs, 2 hidden nodes, 1 output) before attempting more complex architectures

- **Focus on Interactivity**: Make sure users can interact with the visualization to truly understand how backpropagation works

- **Educational Focus**: Remember that the primary goal is to help others understand backpropagation, so prioritize clarity and educational value over complexity

## Final Note

Backpropagation is the heart of how deep neural networks learn. Just as environmental scientists trace the impacts of changes through complex ecosystems, backpropagation traces the influence of each weight on the final prediction error. By building this visualization, you're creating a powerful tool for understanding one of the most important algorithms in machine learning.

Happy Vibe Coding!
