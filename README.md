# PCBench - System Performance Benchmarking Tool

PCBench is a versatile and user-friendly Python-based system performance benchmarking tool designed to evaluate the computing capabilities of your system's CPU and GPU. Whether you're a tech enthusiast, a PC gamer, or a developer optimizing your code, PCBench provides valuable insights into your hardware's performance.

## Key Features

- **CPU Benchmarking:** PCBench offers CPU benchmarking capabilities, allowing you to measure your CPU's processing power and performance. It calculates a variety of metrics to provide a comprehensive view of your CPU's capabilities.

- **GPU Benchmarking:** If you have a dedicated GPU, PCBench can benchmark its performance as well. It gathers critical information about your GPU and runs tests to assess its computational power.

- **Easy-to-Use Interface:** PCBench comes with an intuitive and interactive command-line interface (CLI). It guides you through the benchmarking process step by step, making it accessible for users of all experience levels.

- **Customization:** The tool allows you to customize benchmarking parameters, such as the number of samples, iterations, or other relevant settings, so you can tailor the tests to your specific needs.

- **Performance Metrics:** After each benchmarking session, PCBench provides you with detailed performance metrics and stability indicators, helping you understand how your system performs under different conditions.

- **Rich Visualizations:** PCBench leverages the Rich library to create visually appealing and informative outputs, making it easy to interpret the benchmark results.

## Prerequisites

To get started with PCBench, you'll need:

- Python 3.11.5 or higher
- The torch library
- The Rich library (installable via pip)

You can install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

## CPU Bench

The CPU benchmarker in your "PCBench" tool uses the Monte Carlo method to estimate the value of π (pi) by performing random simulations. The benchmarking process involves generating random points within a square and determining how many of those points fall within a quarter-circle. The ratio of points within the quarter-circle to the total number of points generated provides an approximation of π/4. The estimated value of π can then be calculated by multiplying this ratio by 4.

Here's a step-by-step explanation of how the CPU benchmarker works, including relevant code and formulas:

### 1. Monte Carlo Simulation

The benchmarking process relies on the Monte Carlo method, which is a statistical technique for estimating numerical results through random sampling.

#### Code (Part of your `monte_carlo_pi` function):

```python
def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return (inside_circle / num_samples) * 4
```

#### Formula:

The ratio of points inside the quarter-circle to the total number of points generated is given by:

```formula
\[ \frac{\text{inside_circle}}{\text{num_samples}} \]
```

To estimate π, we multiply this ratio by 4:

```formula
\[ \text{Estimated π} = 4 \times \frac{\text{inside_circle}}{\text{num_samples}} \]
```

### 2. Benchmarking π Calculation

The benchmarking process repeats the Monte Carlo simulation for a specified number of iterations to obtain accurate performance metrics.

#### Code (Part of your `benchmark_pi_calculation` function):

```python
def benchmark_pi_calculation(num_samples, num_iterations):
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        pi_estimate = monte_carlo_pi(num_samples)
        end_time = time.time()
        total_time += end_time - start_time
        print(
            f"Iteration {_:2}: π ≈ {pi_estimate:.8f} (Time: {end_time - start_time:.4f} seconds)")
    average_time = total_time / num_iterations
    print(
        f"Average Time for {num_iterations} iterations: {average_time:.4f} seconds")
```

### 3. Benchmarking Options

The tool allows users to customize the number of samples and iterations for benchmarking. Users can choose to change both parameters or use default values.

#### Code (Part of your `cpu_bench` function):

```python
def cpu_bench():
    # ...
    print(cpu_menu)
    opt = input("Choose an option: ")
    match opt:
        case "1":
            num_samples = input("Enter the Num Samples:")
            bench(int(num_samples), int(num_iterations))
        case "2":
            num_iterations = input("Enter the Num Iterations:")
            bench(int(num_samples), int(num_iterations))
        case "3":
            num_samples = input("Enter the Num Samples:")
            num_iterations = input("Enter the Num Iterations:")
            bench(int(num_samples), int(num_iterations))
        case "4":
            bench(int(num_samples), int(num_iterations))
```

### 4. Results and Metrics

The benchmarking process generates performance metrics, including the estimated value of π and the average time taken for calculations. These metrics provide insights into the CPU's performance in carrying out the Monte Carlo simulations.

#### Code (Part of your `benchmark_pi_calculation` function):

```python
print(
    f"Iteration {_:2}: π ≈ {pi_estimate:.8f} (Time: {end_time - start_time:.4f} seconds)")

average_time = total_time / num_iterations
print(
    f"Average Time for {num_iterations} iterations: {average_time:.4f} seconds")
```

The CPU benchmarker helps users understand their CPU's processing power by quantifying its performance in this computational task. The estimated value of π is a useful metric that provides insights into the CPU's ability to perform complex calculations efficiently. Users can also use the tool to compare CPU performance under different configurations or settings.

## GPU Bench

GPU benchmarking in your "PCBench" tool assesses the performance of your GPU using a deep learning model. In this case, a simple neural network is used to classify images from a dataset. We'll explain the AI model algorithm and how it's useful for testing, along with code snippets and illustrations to help you understand.

### AI Model Algorithm

The AI model used for GPU benchmarking is a basic neural network for image classification. Here's a simplified outline of the algorithm:

1. **Data Loading:** Load a dataset of images (e.g., CIFAR-10) for training.

2. **Model Architecture:** Define a neural network with multiple layers, including fully connected (linear) layers and activation functions.

3. **Loss Function:** Use a loss function (e.g., Cross-Entropy) to measure the error between predicted and actual labels.

4. **Optimization:** Employ an optimization algorithm (e.g., Stochastic Gradient Descent) to update the model's weights and minimize the loss.

5. **Training Loop:** Iterate through the dataset for multiple epochs (training cycles), updating the model's weights after each batch of images.

6. **Performance Metrics:** Measure performance using metrics such as loss, accuracy, and training time.

### Code Snippets

Here are code snippets to illustrate how the AI model is implemented and used for GPU benchmarking:

#### Data Loading and Model Definition:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

#### Training and Benchmarking:

```python
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

# Benchmark GPU training
def benchmark_gpu_training(num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        loss = train(model, trainloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

# Call the benchmarking function with desired parameters
benchmark_gpu_training(num_epochs=5, batch_size=64)
```

### Usefulness for Testing

Using an AI model for GPU benchmarking provides several benefits:

1. **Realistic Workload:** Deep learning models represent real-world workloads that demand significant computational power, making them ideal for GPU testing.

2. **Multi-Purpose:** GPU benchmarking with AI models can assess not only raw performance but also how well the GPU handles complex calculations required for AI tasks.

3. **Generalizability:** AI models can assess GPU performance across various workloads, allowing users to gauge their GPU's capabilities for different applications.

4. **Practical Insights:** Users can evaluate GPU stability, training times, and accuracy, which are essential factors for AI applications.

Illustrating these concepts with code snippets and performance metrics provides users with a comprehensive understanding of how their GPU performs in AI-related tasks. Additionally, visualizations and metrics such as loss curves and training times can help users interpret benchmark results effectively.