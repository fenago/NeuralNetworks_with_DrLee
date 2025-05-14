# Vibe Coding Assignment: Gradient Descent Visualization with AI-Powered Development

> *"The essence of intelligence is not in knowing all the answers, but in learning to find them."* —Dr. Ernesto Lee

## Gradient Descent Explorer: Learning Through Errors

In this Vibe Coding assignment, you'll use AI-powered development tools to build an interactive visualization of gradient descent—the fundamental learning algorithm that allows neural networks to learn from their mistakes. Rather than writing every line of code manually, you'll use natural language prompts to direct the AI to create your application.

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Your Assignment: Build an Interactive Gradient Descent Visualizer

You will create an interactive application that:

1. Visualizes the error landscape for an environmental prediction problem
2. Demonstrates gradient descent descending toward the minimum error
3. Allows users to experiment with different learning rates and initial weights
4. Incorporates real-world environmental data examples

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select an Environmental Problem Domain

Choose one of these environmental prediction problems for your gradient descent visualizer:

1. **Forest Carbon Sequestration**: Predict carbon absorption based on forest density, tree age, and rainfall
2. **Solar Energy Production**: Predict solar panel output based on sunlight hours, panel angle, and temperature
3. **Water Evaporation Rates**: Predict evaporation based on temperature, humidity, and wind speed
4. **Crop Yield Estimation**: Predict agricultural yields based on rainfall, soil quality, and temperature

## Vibe Coding Prompts Guide

Below are example natural language prompts you can use with your chosen AI coding tool. These are starting points—refine and expand them as you develop your project.

### Setup and Basic Framework Prompts

```
Create a web application that visualizes gradient descent for environmental predictions. The app should have a 3D visualization of an error surface and show the path of gradient descent as it finds the minimum error.
```

```
Set up a basic layout with four main sections: 1) A 3D visualization of the error landscape, 2) Controls for adjusting learning rate and initial weights, 3) Real-time updates showing the current iteration, weights, and error value, and 4) A data panel for choosing different environmental scenarios.
```

### Error Landscape Visualization Prompts

```
Create a 3D visualization of an error surface for a [YOUR_ENVIRONMENT_PROBLEM] prediction model with two weights. Use Three.js to create a smooth, interactive surface where the x and y axes represent the two weights, and the z-axis represents the error value.
```

```
Add the ability to rotate, zoom, and pan the 3D error landscape. Color the surface using a gradient from red (high error) to blue (low error) to make it visually clear where the minimum is located.
```

### Gradient Descent Animation Prompts

```
Implement an animation that shows a sphere moving along the error surface following the gradient descent path. Display small flags or markers at each step to show the historical path.
```

```
Add a trace line that follows the gradient descent path down the error surface, with clear indicators for each iteration. The line should change color based on how quickly the error is decreasing.
```

### Interactive Controls Prompts

```
Create a control panel that allows users to adjust the learning rate between 0.0001 and 1.0 using a slider. Include a reset button that restarts gradient descent from randomly chosen initial weights.
```

```
Add a feature that compares multiple gradient descent runs with different learning rates simultaneously, showing different colored paths on the same error surface.
```

### Environmental Data Integration Prompts

```
Create preset environmental scenarios for my [YOUR_CHOSEN_DOMAIN] with realistic data values and a true underlying relationship. Include both simple cases (linear relationships) and complex cases (non-linear relationships).
```

```
Add a data visualization panel that shows the relationship between input features and the predicted output. When gradient descent completes, show how well the learned weights predict the actual values.
```

### Educational Elements Prompts

```
Add educational tooltips that explain key concepts of gradient descent when users hover over different parts of the visualization. Include explanations of learning rate, gradient calculation, and how weights are updated.
```

```
Create an information panel that displays the mathematical formulas used in gradient descent, with the current values plugged in at each step. This should include the prediction formula, error calculation, and gradient computation.
```

## Example Project Workflow: Carbon Sequestration Predictor

Here's an example workflow using Vibe Coding to build a Forest Carbon Sequestration gradient descent visualizer:

### 1. Create the Application Structure

```
Create a new web application for visualizing gradient descent in forest carbon sequestration prediction. The application should use a neural network with two weights to predict carbon absorption based on forest density and tree age.
```

### 2. Setup the Error Landscape

```
Create a 3D visualization of the error landscape for the carbon sequestration model. The x-axis should represent the weight for forest density (ranging from -1 to 1), the y-axis should represent the weight for tree age (ranging from -1 to 1), and the z-axis should represent the mean squared error of predictions.
```

### 3. Generate Realistic Data

```
Generate a realistic dataset for forest carbon sequestration with 100 data points. Each point should have:
- Forest density (trees per hectare, ranging from 100 to 1000)
- Average tree age (years, ranging from 5 to 100)
- Carbon absorption rate (tons per hectare per year)

The true relationship should be: carbon_absorption = 0.02 * density + 0.15 * age + some random noise.
```

### 4. Implement Gradient Descent

```
Implement gradient descent for the carbon sequestration model with the following features:
1. Calculate predictions using the current weights
2. Calculate mean squared error
3. Calculate gradients for both weights
4. Update weights using the learning rate
5. Visualize the step on the error surface

Display the numerical values for current weights, error, and gradients at each step.
```

### 5. Add Interactive Controls

```
Add interactive controls for the gradient descent visualization:
1. A slider for learning rate (0.0001 to 0.1)
2. Input fields for setting initial weights manually
3. A "Run" button to start the animation
4. A "Step" button to advance one iteration at a time
5. A "Reset" button to start over

Also add a checkbox to show/hide the gradient vectors on the error surface.
```

### 6. Create an Educational Panel

```
Create an educational sidebar that explains what's happening during gradient descent:
1. Show the mathematical formulas being used
2. Explain what the current error means in terms of prediction accuracy
3. Explain why the gradient points in a particular direction
4. Highlight when the algorithm is converging or struggling

Include intuitive explanations of how this relates to real forest carbon sequestration modeling.
```

## Advanced Features to Try

Once you've built the basic application, try using Vibe Coding to implement these advanced features:

1. **Add Momentum**: Implement and visualize gradient descent with momentum, showing how it helps navigate flat or noisy parts of the error surface

2. **Local Minima Challenge**: Create an error surface with multiple local minima and show how different initializations or learning rates can lead to different solutions

3. **Stochastic vs. Batch**: Add the option to switch between stochastic gradient descent (one sample at a time) and batch gradient descent (all samples at once)

4. **Learning Rate Scheduler**: Implement a learning rate scheduler that automatically decreases the learning rate over time

5. **Contour Map**: Add a 2D contour map below the 3D visualization that shows the error surface from a top-down view

## Create Your Own Gradient Descent Application

Now it's your turn to apply what you've learned about gradient descent to an environmental challenge that interests you!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an application that applies gradient descent to solve or visualize a unique environmental problem.

Some ideas to inspire you:

- A climate model that uses gradient descent to find the optimal parameters for predicting temperature changes
- A renewable energy optimizer that finds the best position for wind turbines using gradient descent
- A water conservation system that uses gradient descent to optimize irrigation schedules
- A biodiversity prediction model that learns the relationship between habitat features and species diversity

The key is to use Vibe Coding to direct the AI to build your application, focusing on clear prompts that explain what you want rather than writing every line of code yourself.

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A document containing the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Documentation**: A brief explanation of how your application works and how it demonstrates gradient descent

4. **Demo**: A link to a live demo or a video showcasing your application in action

5. **Reflection**: A short reflection (2-3 paragraphs) on your experience using Vibe Coding with AI tools to build this application. What worked well? What was challenging? How did it compare to traditional coding?

## Tips for Effective Vibe Coding

- **Start With Structure**: First focus on creating the overall structure and components before diving into specific functionality

- **Be Specific About Mathematics**: When prompting about gradient descent, be explicit about the mathematical formulas and how they should be implemented

- **Iterative Refinement**: Start with basic prompts and progressively refine them based on what the AI generates

- **Visual Elements Are Key**: Be detailed when describing visualizations, including colors, shapes, labels, and interactions

- **Educational Context**: Remember to include educational elements that explain the gradient descent process

- **Check for Accuracy**: Verify that the gradient calculations and weight updates are implemented correctly

- **Learn From the Code**: Study the code the AI generates to deepen your understanding of gradient descent

## Final Note

Gradient descent is the beating heart of how neural networks learn. By building this visualization, you're not just creating an interesting application—you're developing an intuitive understanding of the learning process that drives modern AI. This understanding will serve you well whether you're building environmental models, analyzing ecological data, or developing cutting-edge AI systems.

Happy Vibe Coding!
