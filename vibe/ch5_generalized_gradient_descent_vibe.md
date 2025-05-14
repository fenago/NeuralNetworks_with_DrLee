# Vibe Coding Assignment: Multi-Weight Neural Networks with AI-Powered Development

> *"The true power of a neural network emerges not from a single connection, but from the harmonious dance of many weights working together."* —Dr. Ernesto Lee

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Your Assignment: The Ecological Symphony of Weights

In this Vibe Coding assignment, you'll build an interactive application that demonstrates how neural networks with multiple weights process environmental data. Using natural language instructions with an AI coding tool, you'll create a system that showcases the "multi-weight waltz" that enables neural networks to model complex ecological relationships.

## The Multi-Weight Network Explorer

Your task is to create an interactive web application that:

1. Demonstrates neural networks with multiple inputs, multiple outputs, or both
2. Visualizes weight matrices and their connection to real-world environmental patterns
3. Shows the learning process as weights adjust to reduce error
4. Allows users to experiment with different environmental scenarios

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select Your Network Architecture Focus

Choose one of these multi-weight network structures to focus on:

1. **Multiple Inputs → One Output**: Creating prediction models that consider multiple environmental factors (like the fire risk example in Chapter 5)

2. **One Input → Multiple Outputs**: Building models that extract multiple insights from a single measurement (like the weather station example)

3. **Multiple Inputs → Multiple Outputs**: Developing comprehensive environmental monitoring systems (like the ecosystem health example)

## Vibe Coding Prompts Guide

Below are example natural language prompts you can use with your chosen AI coding tool. Remember to adjust these based on your specific focus and project needs.

### Project Setup Prompts

```
Create a web application that demonstrates multi-weight neural networks for environmental science. The app should include interactive visualizations, a neural network simulator, and educational explanations about how weights work together.
```

```
Set up the basic application structure with these components: a neural network visualization panel, input controls for environmental variables, a weight matrix display, prediction outputs, and an error graph.
```

### Neural Network Visualization Prompts

```
Create a visualization of a neural network with [X] inputs and [Y] outputs using SVG or Canvas. Each connection between nodes should represent a weight, with line thickness proportional to weight magnitude and color indicating positive (green) or negative (red) values.
```

```
Add a mermaid.js diagram generator that can create network architecture diagrams based on the current network structure. The diagram should update when the user changes the network configuration.
```

Example mermaid.js diagram to include:

```
flowchart LR
    subgraph "Inputs"
        A1["Temperature"] 
        A2["Humidity"]
        A3["Wind Speed"]
    end
    
    subgraph "Outputs"
        B1["Fire Risk"]
        B2["Plant Stress"]
    end
    
    A1 -->|"w11"| B1
    A1 -->|"w12"| B2
    A2 -->|"w21"| B1
    A2 -->|"w22"| B2
    A3 -->|"w31"| B1
    A3 -->|"w32"| B2
    
    style A1 fill:#bbdefb,stroke:#333,stroke-width:1px
    style A2 fill:#bbdefb,stroke:#333,stroke-width:1px
    style A3 fill:#bbdefb,stroke:#333,stroke-width:1px
    style B1 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style B2 fill:#f8bbd0,stroke:#333,stroke-width:1px
```

### Network Simulation Prompts

```
Implement a multi-weight neural network simulator that can perform forward propagation with multiple inputs and outputs. Include a matrix multiplication visualization that shows how inputs and weights combine to produce predictions.
```

```
Create an interactive weight matrix editor where users can manually adjust weights and immediately see how changes affect predictions for environmental variables.
```

### Learning Visualization Prompts

```
Implement gradient descent for the multi-weight network. Show how the gradients are calculated for each weight and how they're used to update the weights. Display this process step-by-step with visual aids.
```

```
Add an animated learning visualization that shows how weights change over time as the network learns. Include a 3D error surface for two selected weights, with a path showing how these weights evolve during training.
```

Example learning process visualization to include:

```
flowchart TB
    subgraph "PREDICT"
        A1["Inputs"]
        A2["Weights"]
        A3["Matrix Multiplication"]
        A4["Predictions"]
        A1 --> A3
        A2 --> A3
        A3 --> A4
    end
    
    subgraph "COMPARE"
        B1["Actual Values"]
        B2["Calculate Errors"]
        B3["Calculate Deltas"]
        B1 --> B2
        A4 --> B2
        B2 --> B3
    end
    
    subgraph "LEARN"
        C1["Calculate Weight Deltas"]
        C2["Update Weights"]
        C3["New Weights"]
        B3 --> C1
        A1 --> C1
        C1 --> C2
        A2 --> C2
        C2 --> C3
    end
    
    style A1 fill:#bbdefb,stroke:#333,stroke-width:1px
    style A2 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style A3 fill:#ffcc80,stroke:#333,stroke-width:1px
    style A4 fill:#ffcc80,stroke:#333,stroke-width:1px
    style B1 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style B2 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style B3 fill:#f8bbd0,stroke:#333,stroke-width:1px
    style C1 fill:#d1c4e9,stroke:#333,stroke-width:1px
    style C2 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C3 fill:#c8e6c9,stroke:#333,stroke-width:1px
```

### Environmental Data Integration Prompts

```
Create several environmental science scenarios that demonstrate the use of multi-weight neural networks, such as a forest fire risk predictor, a crop yield estimator, or a water quality monitor.
```

```
Implement a data visualization component that shows the relationship between input features and outputs in environmental datasets. Include correlation heatmaps and feature importance charts.
```

### Weight Interpretation Prompts

```
Add a weight interpretation visualization that helps users understand what patterns the network has learned. For instance, show how the weights for a forest detector might prioritize certain environmental features.
```

```
Create a dot product visualization as described in Chapter 5 that shows how environmental inputs combine with weights to produce meaningful predictions. Include examples for different ecosystems (forest, desert, wetland).
```

Example weight pattern visualization to include:

```
flowchart LR
    subgraph "Forest Detector Weights"
        W1["Tree Density: 0.8"]
        W2["Humidity: 0.6"]
        W3["Ground Cover: 0.5"]
    end
    
    subgraph "Forest Input Values"
        I1["Tree Density: 90%"]
        I2["Humidity: 75%"]
        I3["Ground Cover: 80%"]
    end
    
    subgraph "Dot Product"
        D1["0.8 × 90% = 0.72"]
        D2["0.6 × 75% = 0.45"]
        D3["0.5 × 80% = 0.40"]
        D4["Sum = 1.57"]
        D1 --> D4
        D2 --> D4
        D3 --> D4
    end
    
    style W1 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style W2 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style W3 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style I1 fill:#bbdefb,stroke:#333,stroke-width:1px
    style I2 fill:#bbdefb,stroke:#333,stroke-width:1px
    style I3 fill:#bbdefb,stroke:#333,stroke-width:1px
    style D1 fill:#ffcc80,stroke:#333,stroke-width:1px
    style D2 fill:#ffcc80,stroke:#333,stroke-width:1px
    style D3 fill:#ffcc80,stroke:#333,stroke-width:1px
    style D4 fill:#ffcc80,stroke:#333,stroke-width:2px
```

## Sample Project: Ecosystem Monitoring System

Here's an example workflow using Vibe Coding to build an Ecosystem Monitoring System with multiple inputs and outputs:

### 1. Set Up Project Structure

```
Create a new React application for an ecosystem monitoring system that uses a neural network with 3 inputs (temperature, humidity, soil moisture) and 3 outputs (plant growth prediction, water needs, pest risk). Include Bootstrap for styling and D3.js for data visualization.
```

### 2. Create Network Visualization

```
Create an interactive visualization of a 3×3 neural network (3 inputs, 3 outputs) using SVG. Each node should be a circle with a label, and connections should be lines with thickness proportional to weight magnitude. Use green for positive weights and red for negative weights. Also generate a mermaid.js diagram showing the same network structure.
```

### 3. Implement Forward Propagation

```
Implement the forward propagation function for the ecosystem monitoring neural network. Use matrix multiplication to calculate outputs from inputs and weights. Create a step-by-step visualization that shows how the weighted sum is calculated for each output.
```

### 4. Add Weight Matrix Editor

```
Create an interactive weight matrix editor with a 3×3 grid of input fields. Each field should represent a weight from one input to one output. When users change a weight, update the network visualization and recalculate predictions instantly. Include preset weight configurations that demonstrate different ecological relationships.
```

### 5. Implement Learning Process

```
Implement gradient descent for the ecosystem monitoring network. Create a training dataset with 20 samples of environmental data and corresponding ecosystem health metrics. Add a "Train Network" button that runs gradient descent for 100 epochs. Show a live visualization of how weights change during training and how error decreases.
```

### 6. Add Environmental Scenarios

```
Add a feature that lets users select from different ecosystem scenarios:
1. Temperate Forest (moderate temperature, high humidity, high soil moisture)
2. Desert (high temperature, low humidity, low soil moisture)
3. Wetland (moderate temperature, high humidity, very high soil moisture)
4. Agricultural Land (variable temperature, moderate humidity, managed soil moisture)

For each scenario, show the network's predictions and explain what they mean for ecosystem management.
```

### 7. Create Weight Interpretation Visualizations

```
Create visualizations that help users interpret what the trained weights mean. For each output (plant growth, water needs, pest risk), show a bar chart of the weights from each input. Add explanations of what these weights reveal about environmental relationships, such as "High temperatures increase pest risk but decrease plant growth."
```

## Advanced Features to Try

After building your base application, try using Vibe Coding to add these advanced features:

1. **Normalization Visualization**: Add a feature that shows why input normalization is important for multi-weight networks, as explained in Chapter 5

2. **Batch Learning**: Implement and visualize batch gradient descent with multiple samples

3. **Hidden Layers**: Extend your network to include hidden layers, moving toward a deep learning architecture

4. **Weight Initialization Strategies**: Compare different weight initialization approaches and their effect on learning

5. **Feature Importance Analysis**: Add tools to analyze which input features have the most impact on predictions

## Create Your Own Multi-Weight Application

Now it's your turn to use Vibe Coding to create an application that applies multi-weight neural networks to an environmental challenge you care about!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an application that demonstrates the power of multiple weights working together to model a complex ecological system.

Some ideas to inspire you:

- A biodiversity prediction system that estimates species richness from multiple environmental factors
- A climate change impact analyzer that predicts multiple ecological effects from changing climate variables
- A sustainable agriculture planner that optimizes multiple crop outputs based on various growing conditions
- A water quality monitoring system that estimates multiple pollutant levels from sensor readings

Remember to incorporate mermaid.js diagrams to help visualize your network architecture and learning process!

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A document containing the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Network Diagram**: At least one mermaid.js diagram showing your neural network architecture

4. **Learning Visualization**: At least one visualization showing how your multi-weight network learns

5. **Demo**: A link to a live demo or a video showcasing your application in action

6. **Reflection**: A short reflection (2-3 paragraphs) on your experience using Vibe Coding with AI tools to build this application

## Tips for Effective Vibe Coding with Multiple Weights

- **Think in Matrices**: When describing neural networks with multiple inputs and outputs, use matrix terminology to guide the AI

- **Visualize Each Component**: Request visualizations for network structure, weight matrices, learning process, and predictions separately

- **Start Simple, Then Expand**: Begin with a small network (2×2) before scaling up to more complex architectures

- **Include Ecological Context**: Always connect your network to real environmental applications to maintain the theme

- **Break Down Complex Mathematics**: When implementing gradient descent for multiple weights, ask the AI to visualize each step

- **Use Mermaid.js**: Specifically request mermaid.js diagrams for network architecture and learning workflow

## Final Note

Just as ecosystems are interconnected webs of relationships, neural networks with multiple weights capture the complex interactions between environmental variables. By building this application, you're not just learning about neural networks—you're gaining insight into how we can model and understand the intricate balance of our natural world.

Happy Vibe Coding!
