# Vibe Coding Assignment: Regularization Explorer

> *"Just as a forest thrives through diversity rather than monotony, a neural network performs best when it learns patterns, not specifics."* —Dr. Ernesto Lee

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Your Assignment: Build a Regularization Visualizer

In this simpler Vibe Coding assignment, you'll create an interactive web application that demonstrates how regularization techniques prevent overfitting in neural networks. Using natural language prompts with your chosen AI coding tool, you'll build a visualization that helps others understand the "Goldilocks principle" of finding the right balance in model complexity.

## Project Overview

You'll create an interactive web application that:

1. Demonstrates overfitting with a simple dataset
2. Visualizes the effects of different regularization techniques
3. Allows users to adjust regularization parameters and see the results
4. Includes intuitive explanations of how each technique works

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select a Dataset Focus

Choose one of these simple datasets to visualize regularization effects:

1. **Species Distribution**: Predict presence/absence of a species based on environmental variables
2. **Climate Patterns**: Predict temperature based on historical patterns
3. **Pollution Levels**: Predict air quality based on various factors

## Vibe Coding Prompts Guide

Here are example natural language prompts to help you build your application:

### Project Setup Prompts

```
Create a simple web application that demonstrates how regularization techniques (L1, L2, dropout, and early stopping) help prevent overfitting in neural networks. The application should have a clean, minimalist interface focused on visualizations rather than complex controls.
```

```
Set up a basic layout with three main sections: 1) A data visualization panel showing the training and test data, 2) A model fitting visualization showing how different regularization techniques affect the decision boundary, and 3) Simple controls to adjust regularization parameters.
```

### Dataset Generation Prompts

```
Create a simple 2D dataset for visualizing overfitting. Generate two classes of points with some noise and overlap to represent [YOUR_CHOSEN_DOMAIN]. For example, if using species distribution, generate points representing presence and absence based on temperature and rainfall.
```

```
Split the generated dataset into training and test sets. Visualize both sets on a 2D scatter plot with different colors for each class and different markers for training versus test data.
```

Example mermaid.js diagram to include:

```
flowchart LR
    subgraph "Dataset Structure"
        A["Generate Data"] --> B["Split Into Train/Test"]  
        B --> C["Training Data"]  
        B --> D["Test Data"]  
    end
    
    style A fill:#bbdefb,stroke:#333,stroke-width:1px
    style B fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C fill:#ffcc80,stroke:#333,stroke-width:1px
    style D fill:#e1bee7,stroke:#333,stroke-width:1px
```

### Model Implementation Prompts

```
Implement a simple neural network model using TensorFlow.js or another JavaScript machine learning library. The model should have an input layer for two features, a hidden layer with adjustable size, and an output layer for binary classification.
```

```
Add implementations for L1 and L2 regularization to the model. Create sliders that allow users to adjust the regularization strength (lambda) from 0 (no regularization) to 1 (strong regularization).
```

### Overfitting Visualization Prompts

```
Create a visualization that shows how the decision boundary changes with different levels of model complexity. Plot the training data points, test data points, and the model's decision boundary. As the model becomes more complex (more neurons), show how it starts to create a jagged boundary that perfectly fits the training data but fails on test data.
```

```
Add a graph showing the training and test accuracy over increasing model complexity. Highlight the point where the test accuracy starts to decrease while training accuracy continues to improve—the classic sign of overfitting.
```

Example overfitting visualization to include:

```
flowchart TB
    subgraph "Overfitting Visualization"
        A["Simple Model\nSmooth Boundary\nGood Generalization"] --> B["Medium Complexity\nModerate Boundary\nOptimal Generalization"]  
        B --> C["Complex Model\nJagged Boundary\nPoor Generalization"]  
    end
    
    style A fill:#bbdefb,stroke:#333,stroke-width:1px
    style B fill:#c8e6c9,stroke:#333,stroke-width:2px
    style C fill:#ffcdd2,stroke:#333,stroke-width:1px
```

### Regularization Effects Prompts

```
Create a side-by-side comparison showing how different regularization techniques affect the decision boundary of an overfit model. Show one panel with no regularization, one with L1, one with L2, and one with dropout. Use the same complex model architecture for all four but apply the different regularization techniques.
```

```
Add a visualization showing how weights are distributed in the neural network with different regularization techniques. For L1, show how many weights are pushed to exactly zero. For L2, show how all weights are made smaller but non-zero. For dropout, show how the network becomes more robust.
```

Example regularization comparison to include:

```
flowchart LR
    subgraph "Regularization Effects"
        A["No Regularization\nOverfit to Data"] --- B["L1 Regularization\nSparse Weights\nSimpler Boundary"]  
        A --- C["L2 Regularization\nSmaller Weights\nSmoother Boundary"]  
        A --- D["Dropout\nRobust Features\nBetter Generalization"]  
    end
    
    style A fill:#ffcdd2,stroke:#333,stroke-width:1px
    style B fill:#c8e6c9,stroke:#333,stroke-width:1px
    style C fill:#bbdefb,stroke:#333,stroke-width:1px
    style D fill:#e1bee7,stroke:#333,stroke-width:1px
```

### Interactive Controls Prompts

```
Add simple, intuitive controls that let users adjust:
1. Model complexity (number of neurons in the hidden layer)
2. L1 regularization strength (lambda)
3. L2 regularization strength (lambda)
4. Dropout rate (0% to 70%)
5. Early stopping patience (number of epochs without improvement)

Make sure each control has a clear label and a simple explanation of what it does.
```

```
Implement a 'Train Model' button that, when clicked, trains the model with the current settings and updates all visualizations. Show a loading indicator during training and display both training and test accuracy when complete.
```

### Educational Elements Prompts

```
Add simple tooltips or information boxes that explain each regularization technique in plain language, similar to Dr. Lee's explanations in Chapter 8. Include analogies to environmental science where appropriate.
```

```
Create a 'Learning Journey' section that guides users through a series of pre-configured scenarios showing:
1. Underfitting (too simple model)
2. Optimal fitting (balanced model)
3. Overfitting (too complex model)
4. How each regularization technique helps correct overfitting
```

## Sample Project: Species Distribution Regularizer

Here's a simpler example workflow for building a Species Distribution visualization:

### 1. Create the Application Structure

```
Create a new web application with a clean, minimalist interface for visualizing how regularization affects species distribution modeling. The main view should be a 2D plot where x-axis is temperature and y-axis is precipitation, with points representing species presence (green) and absence (red).
```

### 2. Generate Sample Data

```
Generate synthetic data for a fictional butterfly species with these characteristics:
- Present in areas with moderate temperature (0.3-0.7 on a 0-1 scale) and moderate to high precipitation (0.4-0.9)
- Some outliers and noise to make the pattern less obvious
- About 200 data points total, split 70% for training and 30% for testing
```

### 3. Implement the Basic Model

```
Create a simple neural network using TensorFlow.js with:
- 2 input nodes (temperature and precipitation)
- 1 hidden layer with adjustable size (default 16 neurons)
- 1 output node (species presence probability)
- ReLU activation for hidden layer and sigmoid for output

Add a slider to control the hidden layer size from 2 to 64 neurons.
```

### 4. Add Regularization Controls

```
Add controls for regularization:
- A slider for L1 regularization (0 to 0.1)
- A slider for L2 regularization (0 to 0.1)
- A slider for dropout rate (0% to 70%)
- A checkbox for early stopping

Each control should have a simple tooltip explaining its purpose.
```

### 5. Visualize the Decision Boundary

```
Create a visualization that shows the model's decision boundary as a contour on the 2D plot. Use a gradient coloring to show the predicted probability of species presence across the entire temperature/precipitation space. Update this visualization whenever the model is retrained with new settings.
```

### 6. Compare Regularization Effects

```
Add a button that runs four training scenarios and displays them side by side:
1. No regularization
2. L1 regularization (lambda = 0.01)
3. L2 regularization (lambda = 0.01)
4. Dropout (rate = 30%)

For each scenario, show the decision boundary and the test accuracy.
```

## Create Your Own Regularization Visualizer

Now it's your turn to use Vibe Coding to create a simple but effective visualization of regularization techniques!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an interactive application that helps others understand how regularization prevents overfitting.

The key is to focus on clarity and simplicity—choose a straightforward environmental dataset and create intuitive visualizations that clearly show the effects of regularization.

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A list of the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Demo**: A link to a live demo or a video showing your application in action

4. **Brief Explanation**: A short explanation (1-2 paragraphs) of how your visualization demonstrates regularization techniques

## Tips for Successful Vibe Coding

- **Keep It Simple**: Focus on clear visualizations of one or two key concepts rather than trying to include everything

- **Use Plain Language**: When describing regularization to the AI, use simple analogies like those in Chapter 8

- **Prioritize Visuals**: A good visualization is worth a thousand words—focus your prompts on creating effective visual explanations

- **Start Small**: Begin with a minimal viable product that shows one regularization technique, then expand

- **Learn From Examples**: Study the code the AI generates to better understand both regularization and how to guide AI effectively

## Final Note

Regularization is all about finding the sweet spot—the Goldilocks zone where your model learns the true patterns in the data without memorizing the noise. By creating this visualization, you're building an important tool for understanding one of the most critical concepts in machine learning, while also gaining valuable experience with AI-powered development.

Happy Vibe Coding!
