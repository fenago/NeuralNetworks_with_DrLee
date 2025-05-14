# Vibe Coding Assignment: The Activation Function Playground

> *"Just as different species respond differently to environmental changes, neural networks behave differently depending on the activation functions they use."* —Dr. Ernesto Lee

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Your Assignment: Build an Activation Function Visualizer

In this Vibe Coding assignment, you'll create an interactive web application that demonstrates how different activation functions affect neural network behavior and learning. Using natural language prompts with your chosen AI coding tool, you'll build a visualization that helps others understand why nonlinearity matters and how different activation functions shape network capabilities.

## Project Overview

You'll create an interactive web application that:

1. Visualizes different activation functions and their derivatives
2. Demonstrates how activation functions affect gradient flow in networks
3. Shows how networks with different activation functions learn different patterns
4. Connects activation functions to environmental data modeling

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select an Environmental Focus

Choose one of these environmental contexts to frame your visualization:

1. **Wildlife Population Dynamics**: How population growth follows sigmoid-like patterns
2. **Ecosystem Thresholds**: How environmental systems have ReLU-like response thresholds
3. **Climate Patterns**: How seasonal variations resemble tanh-like cycles

## Vibe Coding Prompts Guide

Here are example natural language prompts to help you build your application:

### Project Setup Prompts

```
Create a web application that visualizes different activation functions and demonstrates their effects on neural network learning. The application should have a clean interface with interactive visualizations and clear explanations.
```

```
Set up a basic layout with three main sections: 1) An activation function visualization panel showing functions and their derivatives, 2) A neural network learning panel demonstrating how networks with different activations learn, and 3) Simple controls to select different activation functions and datasets.
```

### Activation Function Visualization Prompts

```
Create an interactive visualization that plots the following activation functions side by side: ReLU, Sigmoid, Tanh, Leaky ReLU, and ELU. For each function, show its curve over the range [-5, 5] on the x-axis. Include a slider to adjust the input value and highlight the current output value on each curve.
```

```
Add a second panel that shows the derivatives of each activation function. When the user moves the input slider, highlight the derivative value at that point for each function, illustrating the gradient flow during backpropagation.
```

Example mermaid.js diagram to include:

```
flowchart LR
    subgraph "Activation Functions Comparison"
        A["Input x"] --> B["ReLU"] & C["Sigmoid"] & D["Tanh"] & E["Leaky ReLU"] & F["ELU"]
        B --> G["Derivative of ReLU"]
        C --> H["Derivative of Sigmoid"]
        D --> I["Derivative of Tanh"]
        E --> J["Derivative of Leaky ReLU"]
        F --> K["Derivative of ELU"]
    end
    
    style A fill:#bbdefb,stroke:#333,stroke-width:1px
    style B fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C fill:#ffcc80,stroke:#333,stroke-width:1px
    style D fill:#e1bee7,stroke:#333,stroke-width:1px
    style E fill:#b2dfdb,stroke:#333,stroke-width:1px
    style F fill:#f8bbd0,stroke:#333,stroke-width:1px
    style G fill:#c8e6c9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style H fill:#ffcc80,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style I fill:#e1bee7,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style J fill:#b2dfdb,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style K fill:#f8bbd0,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
```

### Neural Network Learning Visualization Prompts

```
Implement a simple neural network visualization that shows how networks with different activation functions learn patterns. Create a 2D scatter plot with two classes of points (e.g., healthy vs. polluted water samples based on two features). Show how the decision boundary evolves during training for networks using different activation functions.
```

```
Add a 'Play' button that animates the training process, showing how the decision boundary changes over epochs. Display the training and test accuracy for each activation function as the animation progresses.
```

Example learning visualization diagram to include:

```
flowchart TD
    subgraph "Learning With Different Activations"
        A["Training Data"] --> B["ReLU Network"] & C["Sigmoid Network"] & D["Tanh Network"]
        B --> E["ReLU Boundary"]
        C --> F["Sigmoid Boundary"]
        D --> G["Tanh Boundary"]
        E & F & G --> H["Compare Performance"]
    end
    
    style A fill:#bbdefb,stroke:#333,stroke-width:1px
    style B fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C fill:#ffcc80,stroke:#333,stroke-width:1px
    style D fill:#e1bee7,stroke:#333,stroke-width:1px
    style E fill:#c8e6c9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style F fill:#ffcc80,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style G fill:#e1bee7,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style H fill:#f8bbd0,stroke:#333,stroke-width:1px
```

### Vanishing/Exploding Gradient Demonstration Prompts

```
Create a visualization that demonstrates the vanishing and exploding gradient problems. Show a deep neural network (at least 10 layers) and visualize how the gradient magnitude changes as it flows backward through the network with different activation functions.
```

```
Implement a slider to adjust the network depth and show how deeper networks amplify the vanishing gradient problem with sigmoid and tanh, but ReLU and its variants maintain better gradient flow.
```

Example gradient flow diagram to include:

```
flowchart LR
    subgraph "Gradient Flow Visualization"
        X["Error Signal"] --> L10["Layer 10"] --> L9["Layer 9"] --> L8["Layer 8"] --> L7["..."]
        --> L3["Layer 3"] --> L2["Layer 2"] --> L1["Layer 1"] --> I["Input Layer"]
        
        subgraph "Gradient Magnitude"
            G10["High"] --- G9 --- G8 --- G7 --- G6 --- G5 --- G4 --- G3 --- G2 --- G1["Low"]
        end
    end
    
    style X fill:#ffcdd2,stroke:#333,stroke-width:1px
    style L10,L9,L8,L7,L3,L2,L1,I fill:#bbdefb,stroke:#333,stroke-width:1px
    style G10 fill:#c8e6c9,stroke:#333,stroke-width:1px
    style G9,G8,G7,G6,G5,G4,G3,G2 fill:#ffcc80,stroke:#333,stroke-width:1px
    style G1 fill:#ffcdd2,stroke:#333,stroke-width:1px
```

### Environmental Application Prompts

```
Implement a module that demonstrates how different activation functions are suitable for different environmental data patterns. Create three example datasets:
1. A sigmoid-like wildlife population growth curve
2. A ReLU-like pollution threshold effect
3. A tanh-like seasonal temperature variation

Allow users to train a simple network with different activation functions on each dataset and visualize how well the network learns the pattern.
```

```
Add a recommendation system that suggests which activation function might be most appropriate for different types of environmental data, based on the patterns in the data. Include brief explanations of why certain activations work better for certain patterns.
```

Example environmental application diagram to include:

```
flowchart TD
    subgraph "Environmental Data Patterns"
        D1["Population Growth\
(Sigmoid-like)"] --- A1["Best: Sigmoid/Tanh"]
        D2["Threshold Effect\
(ReLU-like)"] --- A2["Best: ReLU/Leaky ReLU"]
        D3["Seasonal Variation\
(Tanh-like)"] --- A3["Best: Tanh/GELU"]
        D4["Multiple Thresholds\
(Step-like)"] --- A4["Best: Custom Activation"]
    end
    
    style D1,D2,D3,D4 fill:#bbdefb,stroke:#333,stroke-width:1px
    style A1,A2,A3,A4 fill:#c8e6c9,stroke:#333,stroke-width:1px
```

### Interactive Controls Prompts

```
Add a control panel that allows users to:
1. Select from common activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, GELU)
2. Adjust hyperparameters specific to each activation (e.g., alpha for Leaky ReLU) 
3. Choose from different environmental datasets
4. Change neural network parameters (depth, width)

Update all visualizations when these controls change.
```

```
Implement a 'comparison mode' that trains networks with different activation functions side by side on the same data and shows their learning curves (training and validation loss over epochs). Allow users to see which activation functions learn faster or achieve better final accuracy.
```

### Educational Elements Prompts

```
Add tooltips or information cards that explain each activation function in simple terms, similar to Dr. Lee's explanations in Chapter 9. Include environmental analogies where possible.
```

```
Create a 'Learning Journey' section that guides users through key concepts:
1. Why nonlinearity matters (show a linear network failing on nonlinear data)
2. How different activation functions shape the network's behavior
3. How gradient flow affects learning in deep networks
4. How to choose the right activation for different environmental data patterns
```

## Sample Project: EcoActivations Explorer

Here's an example workflow for building an Activation Function Visualizer with an environmental focus:

### 1. Create the Application Structure

```
Create a web application with a clean, modern interface to demonstrate how activation functions affect neural network learning on environmental data. The app should include tabs for 'Function Visualizer', 'Network Learning', 'Gradient Flow', and 'Environmental Applications'.
```

### 2. Implement the Activation Function Visualizer

```
Implement an interactive module that plots ReLU, Sigmoid, Tanh, Leaky ReLU, and ELU functions. Show the function curve in the top panel and its derivative in the bottom panel. Add a vertical line that users can drag to see the function value and derivative at different input points.
```

### 3. Create a Pattern Learning Demonstration

```
Create a demonstration showing how networks with different activation functions learn environmental patterns. Generate synthetic data for three scenarios:

1. A wildlife population curve that follows a sigmoid-like growth pattern
2. A pollution impact curve that shows a ReLU-like threshold effect
3. A seasonal temperature variation that follows a tanh-like pattern

Show how networks with different activation functions perform on each pattern.
```

### 4. Visualize Gradient Flow in Deep Networks

```
Implement a visualization of a 10-layer neural network. Show how the gradient magnitude changes during backpropagation for networks using Sigmoid, Tanh, ReLU, and Leaky ReLU. Use color coding to indicate gradient strength at each layer, and update the visualization as users change the network depth using a slider.
```

### 5. Implement a Decision Guide

```
Create an interactive decision tree that helps users select the right activation function for different environmental modeling tasks. Ask questions about the data pattern, the depth of the network, and the desired properties, then recommend appropriate activation functions with brief explanations.
```

## Create Your Own Activation Function Explorer

Now it's your turn to use Vibe Coding to create an intuitive, interactive activation function visualizer!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an application that demonstrates how activation functions work and their effects on neural networks learning environmental patterns.

Focus on making the visualizations clear and intuitive - the goal is to help people develop an intuition for how activation functions shape neural network behavior.

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A list of the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Demo**: A link to a live demo or a video showing your application in action

4. **Brief Explanation**: A short explanation (1-2 paragraphs) of how your visualization demonstrates activation function concepts

## Tips for Successful Vibe Coding

- **Focus on Visualization**: Create clear, intuitive visualizations that help build intuition

- **Use Environmental Analogies**: Connect activation functions to patterns in natural systems

- **Start with Comparison**: A side-by-side comparison of different activation functions is a great starting point

- **Keep It Interactive**: Let users adjust parameters and see immediate effects

- **Connect Theory to Practice**: Show both the mathematical function and its practical effect on network learning

## Final Note

Activation functions are the nonlinear heart of neural networks - they're what give these models their remarkable ability to learn complex patterns. By creating this visualization, you're building an important tool for understanding one of the fundamental building blocks of deep learning, while gaining valuable experience with AI-powered development.

Just as ecosystems thrive on diversity and nonlinear relationships, neural networks achieve their power through the subtle curves and bends introduced by activation functions. Your visualization will help others understand this critical connection.

Happy Vibe Coding!
