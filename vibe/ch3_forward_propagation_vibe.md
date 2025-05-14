# Vibe Coding Assignment: Forward Propagation with AI-Powered Natural Language Programming

> *"The future of coding isn't just about knowing the syntax—it's about knowing how to explain what you want to build."* —Dr. Ernesto Lee

## What is Vibe Coding?

Vibe Coding is a revolutionary approach to software development where you describe what you want in natural language, and AI-powered tools generate the code for you. Instead of writing code line by line, you collaborate with AI by providing high-level instructions, refining them iteratively, and guiding the AI to build your application.

Tools like [Cursor](https://www.cursor.com/), [Windsurf](https://windsurf.com/), and [Bolt.new](https://bolt.new/) are at the forefront of this movement, letting you build complex applications without writing every line of code yourself.

## Assignment: Build a Neural Network Visualizer using Vibe Coding

In this assignment, you'll use an AI-powered development environment (Cursor, Windsurf, or Bolt.new) to build an interactive application that demonstrates forward propagation in neural networks. You'll focus on using natural language to guide the AI in creating your application, rather than writing all the code manually.

### Your Task

Create an interactive web application that:

1. Visualizes a neural network's architecture (input layer, hidden layers, output layer)
2. Demonstrates forward propagation step-by-step with animations
3. Allows users to input their own test cases and see the prediction process
4. Uses environmental/ecological data consistent with Dr. Lee's focus

### Getting Started

#### Step 1: Choose Your AI Coding Environment

Select one of these AI-powered development environments:

- **[Cursor](https://www.cursor.com/)**: A code editor built on VSCode with powerful AI capabilities
- **[Windsurf](https://windsurf.com/)**: An AI agent-powered IDE focused on keeping developers in flow
- **[Bolt.new](https://bolt.new/)**: A browser-based AI coding environment for building web applications

#### Step 2: Select an Environmental Dataset

Choose one of these ecological prediction scenarios for your neural network visualizer:

1. **Climate Impact Predictor**: Predict how climate variables affect biodiversity
2. **Water Quality Analysis**: Predict water quality from environmental factors
3. **Renewable Energy Forecaster**: Predict energy generation from weather conditions
4. **Forest Fire Risk Assessment**: Predict wildfire risk from environmental variables

## Vibe Coding Prompts Guide

Below are examples of natural language prompts you can use with your chosen AI coding tool to build your application. These are starting points—you'll need to refine and expand on them as you develop your project.

### Project Setup Prompts

```
Create a new React application for visualizing neural networks with forward propagation. The app should demonstrate how a neural network processes environmental data to make predictions.
```

```
Set up a basic layout for my neural network visualizer with three main sections: 1) Network Architecture Visualization, 2) Data Input Panel, and 3) Propagation Visualization.
```

### Neural Network Architecture Prompts

```
Create a component that visualizes a neural network with 3 input nodes, 4 hidden nodes, and 1 output node. Use SVG to draw the network with circles for nodes and lines for connections.
```

```
Add the ability to display weight values on the connections between neurons when the user hovers over them.
```

### Forward Propagation Visualization Prompts

```
Implement a step-by-step visualization of forward propagation. Show how the input values propagate through the network with animations highlighting the active connections and nodes.
```

```
Create a function that calculates the weighted sum for each neuron and displays the calculation steps in a sidebar.
```

### Data Input Interface Prompts

```
Add an input form where users can enter environmental data values (like temperature, humidity, etc.) for the neural network to process.
```

```
Include a feature that lets users choose from preset sample data representing different environmental scenarios.
```

### Results and Analysis Prompts

```
Create a results panel that shows the prediction output and explains how each input contributed to the final prediction.
```

```
Add visualizations that show the relative importance of each input feature using bar charts or radar plots.
```

## Sample Project Workflow

Here's how you might build this project using Vibe Coding with an AI-powered coding tool:

1. **Initial Setup**: Use natural language to describe the application structure and have the AI generate the boilerplate code.

2. **Neural Network Model**: Ask the AI to create a neural network model with predefined weights focused on your environmental prediction task.

3. **Visualization Components**: Describe the visualizations you want and have the AI generate the visualization code.

4. **Interactive Elements**: Request user interface components that allow interaction with the network.

5. **Animation and Steps**: Describe how you want to animate the forward propagation process.

6. **Styling and Polish**: Use natural language to refine the appearance and user experience.

7. **Documentation**: Ask the AI to generate explanations of how forward propagation works in your application.

## Example: Building a Climate Impact Predictor

Here's an example workflow for building a Climate Impact Predictor using Cursor AI:

1. **Setup Request**:
```
Create a new React application that visualizes a neural network for predicting biodiversity impact from climate variables. The inputs should be temperature change, precipitation change, and CO2 levels. The output should be a biodiversity impact score.
```

2. **Model Implementation Request**:
```
Implement a forward propagation model using TensorFlow.js with the following architecture:
- 3 input nodes (temperature change, precipitation change, CO2 levels)
- 4 hidden nodes with ReLU activation
- 1 output node (biodiversity impact score)

Use these predefined weights:
- Input to hidden: [[0.1, 0.2, -0.1, 0.15], [-0.15, 0.25, 0.1, -0.2], [0.05, -0.1, -0.15, 0.3]]
- Hidden to output: [[0.4], [0.3], [-0.5], [0.2]]
```

3. **Visualization Request**:
```
Create an interactive SVG visualization of the neural network that shows the forward propagation process. When a user enters input values, animate how the signal flows through the network with these features:
- Nodes should change color intensity based on activation values
- Connection lines should change thickness based on the weight × input
- Show the calculation steps in a sidebar
```

4. **User Interface Request**:
```
Add an input panel where users can:
- Enter custom values for temperature change (-2°C to +4°C)
- Enter precipitation change (-30% to +30%)
- Enter CO2 levels (400ppm to 800ppm)

Also add a "Run Simulation" button and preset scenarios like "Current Trend", "Mitigation Scenario", and "Worst Case".
```

5. **Explanation Component Request**:
```
Add an explanation panel that describes what's happening at each step of forward propagation with both technical details and ecological interpretation. For example, explain how increased temperature combined with decreased precipitation creates a stronger negative signal in certain pathways.
```

## Advanced Features (Optional)

After completing the basic application, try using Vibe Coding to add these advanced features:

1. **Multiple Hidden Layers**: Extend your network to include multiple hidden layers

2. **Different Activation Functions**: Add the ability to switch between activation functions (ReLU, Sigmoid, Tanh)

3. **Weight Adjustment**: Allow users to modify the weights and see how it affects predictions

4. **Comparison View**: Let users compare propagation for different inputs side-by-side

5. **Dataset Integration**: Connect to a real environmental dataset for more realistic predictions

## Create Your Own Forward Propagation Application

Now that you've built a neural network visualizer, use Vibe Coding to create your own application that applies forward propagation concepts to an environmental challenge you care about!

**Your Challenge**: Using natural language prompts with your chosen AI coding tool, build an application that addresses a unique environmental use case. Some ideas:

- A coral reef health predictor that visualizes how different ocean conditions affect reef ecosystems
- A sustainable agriculture planning tool that predicts crop yields based on farming practices
- A wildlife conservation app that predicts animal population changes based on habitat variables
- A renewable energy optimizer that predicts optimal placement of solar panels or wind turbines

The key is to focus on using natural language to direct the AI to build your application, rather than coding everything manually.

## Submission Guidelines

Your submission should include:

1. **Prompt Log**: A document containing the key natural language prompts you used to create your application

2. **Application Code**: The complete code of your application (either as files or a link to a repository)

3. **Documentation**: A brief explanation of how your application works and how it demonstrates forward propagation

4. **Demo**: A link to a live demo or a video showcasing your application in action

5. **Reflection**: A short reflection (2-3 paragraphs) on your experience using Vibe Coding with AI tools to build this application. What worked well? What was challenging? How did it compare to traditional coding?

## Tips for Successful Vibe Coding

- **Be Clear and Specific**: The more specific your natural language prompts are, the better the AI will understand what you want.

- **Iterate**: Don't expect perfect results on the first try. Refine your prompts based on what the AI generates.

- **Understand the Fundamentals**: Even though the AI writes code for you, understanding the concepts of forward propagation will help you guide the AI effectively.

- **Break It Down**: Request complex features in smaller, manageable chunks rather than all at once.

- **Review and Learn**: Take time to understand the code the AI generates - this is still a learning opportunity!

By completing this Vibe Coding assignment, you'll not only deepen your understanding of forward propagation but also gain valuable experience with the future of programming—where your ability to communicate with AI is as important as your coding skills!
