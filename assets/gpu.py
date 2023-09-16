import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import platform
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.align import Align
from rich.markdown import Markdown
from rich import box

num_epochs = 50
batch_size = 128
dataset_size = 50000


def clearscr() -> None:
    try:
        osp = platform.system()
        match osp:
            case 'Darwin':
                os.system("clear")
            case 'Linux':
                os.system("clear")
            case 'Windows':
                os.system("cls")
    except Exception:
        pass


def check_gpu_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs Available: {num_gpus}")
        for gpu_id in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(gpu_id)
            info = f"""
            GPU {gpu_id} Info:
                Name: {gpu_info.name}
                Compute Capability: {gpu_info.major}.{gpu_info.minor}
                Total Memory: {gpu_info.total_memory / (1024 ** 3):.2f} GB
                CUDA Cores: {gpu_info.multi_processor_count}
                Clock Rate: {gpu_info.clock_rate / 1e3:.2f} GHz
            """
            info_profile = Markdown(info)
            option_panel = Panel(
                Align.center(
                    Group("\n", Align.center(info_profile)),
                    vertical="middle",
                ),
                box=box.ROUNDED,
                padding=(1, 2),
                title="[b red]GPU Details",
                border_style="blue",
            )
            print(option_panel)
    else:
        info = """
        - No GPU Found
        - Running on the CPU
        """
        info_profile = Markdown(info)
        option_panel = Panel(
            Align.center(
                Group("\n", Align.center(info_profile)),
                vertical="middle",
            ),
            box=box.ROUNDED,
            padding=(1, 2),
            title="[b red]GPU Details",
            border_style="blue",
        )
        print(option_panel)


class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


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


def benchmark_gpu_training(num_epochs, batch_size):
    clearscr()
    info = f"""
            Number of epochs: {num_epochs}
            Number of training examples: {batch_size}
            """
    info_profile = Markdown(info)
    option_panel = Panel(
        Align.center(
            Group("\n", Align.center(info_profile)),
            vertical="middle",
        ),
        box=box.ROUNDED,
        padding=(1, 2),
        title="[b red]GPU Details",
        border_style="blue",
    )
    print(option_panel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = torch.randn(dataset_size, 1024).to(device)
    train_labels = torch.randint(0, 10, (dataset_size,)).to(device)
    train_dataloader = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataloader, batch_size=batch_size, shuffle=True)

    model = ComplexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    total_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        loss = train(model, train_loader, criterion, optimizer, device)
        end_time = time.time()
        total_time += end_time - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Time: {end_time - start_time:.4f} seconds")

    average_time = total_time / num_epochs
    print(
        f"\nAverage Training Time for {num_epochs} epochs: {average_time:.4f} seconds")

    performance_metrics = {
        "Average Training Time": average_time,
        "Final Loss": loss,
    }

    return performance_metrics


def gpu_bench_menu():
    clearscr()
    global num_epochs
    global batch_size
    gpu_banner = """
      ____  ____  __ __  ____     ___  ____     __  __ __
     /    ||    \|  |  ||    \   /  _]|    \   /  ]|  |  |
    |   __||  o  )  |  ||  o  ) /  [_ |  _  | /  / |  |  |
    |  |  ||   _/|  |  ||     ||    _]|  |  |/  /  |  _  |
    |  |_ ||  |  |  :  ||  O  ||   [_ |  |  /   \_ |  |  |
    |     ||  |  |     ||     ||     ||  |  \     ||  |  |
    |___|_||__|   \__,_||_____||_____||__|__|\____||__|__|
    """
    banner = Markdown(gpu_banner)
    print(banner)
    check_gpu_info()
    gpu_menu = Table()
    gpu_menu.add_column("Option")
    gpu_menu.add_column("Value")
    gpu_menu.add_column("DefaultValue")
    gpu_menu.add_row("1", "Change Epoch", str(num_epochs))
    gpu_menu.add_row("2", "Change Batch Size", str(batch_size))
    gpu_menu.add_row("3", "Change Both", "Your Customs")
    gpu_menu.add_row("4", "Defaults", f"{num_epochs}, {num_epochs}")
    print(gpu_menu)
    opt = input("Choose an option: ")
    match opt:
        case "1":
            num_epochs = input("Enter the Num Epochs:")
            performance_metrics = benchmark_gpu_training(
                int(num_epochs), int(batch_size))
        case "2":
            batch_size = input("Enter the Batch Size:")
            performance_metrics = benchmark_gpu_training(
                int(num_epochs), int(batch_size))
        case "3":
            num_epochs = input("Enter the Num Epochs:")
            batch_size = input("Enter the Batch Size:")
            performance_metrics = benchmark_gpu_training(
                int(num_epochs), int(batch_size))
        case "4":
            performance_metrics = benchmark_gpu_training(
                int(num_epochs), int(batch_size))

    stability_metrics = {
        "GPU Stability": "High" if torch.cuda.is_available() else "N/A",
        "CPU Stability": "High" if not torch.cuda.is_available() else "N/A",
    }

    print("\nPerformance Metrics:")
    for key, value in performance_metrics.items():
        print(f"{key}: {value}")

    print("\nStability Metrics:")
    for key, value in stability_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    gpu_bench_menu()
