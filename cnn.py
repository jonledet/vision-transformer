import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda import is_available
import time

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if is_available() else "cpu")

class CNN(nn.Module):
    """
    Convolutional Neural Network model for image classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool1 (nn.MaxPool2d): First max-pooling layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        pool2 (nn.MaxPool2d): Second max-pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer.

    Methods:
        forward(x): Forward pass through the model.
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # Max pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Output tensor and layer processing times.
        """
        start_time = time.time()

        x = torch.relu(self.conv1(x))
        layer1_time = time.time() - start_time

        start_time = time.time()
        x = torch.relu(self.conv2(x))
        layer2_time = time.time() - start_time

        start_time = time.time()
        x = self.pool1(x)
        pool1_time = time.time() - start_time

        start_time = time.time()
        x = torch.relu(self.conv3(x))
        layer3_time = time.time() - start_time

        start_time = time.time()
        x = self.pool2(x)
        pool2_time = time.time() - start_time

        x = x.view(-1, 32 * 8 * 8)

        start_time = time.time()
        x = torch.relu(self.fc1(x))
        fc1_time = time.time() - start_time

        start_time = time.time()
        x = torch.relu(self.fc2(x))
        fc2_time = time.time() - start_time

        start_time = time.time()
        x = self.fc3(x)
        fc3_time = time.time() - start_time

        layer_times = {
            "Conv1": layer1_time,
            "Conv2": layer2_time,
            "Pool1": pool1_time,
            "Conv3": layer3_time,
            "Pool2": pool2_time,
            "FC1": fc1_time,
            "FC2": fc2_time,
            "FC3": fc3_time,
        }

        return x, layer_times


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    # Define data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, layer_times = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    layer_processing_times = {}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, layer_times = model(inputs)

            # Accumulate layer processing times
            for layer, time_elapsed in layer_times.items():
                if layer in layer_processing_times:
                    layer_processing_times[layer] += time_elapsed
                else:
                    layer_processing_times[layer] = time_elapsed

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print validation accuracy
    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    # Print average layer processing times
    print("\nAverage Layer Processing Times:")
    for layer, total_time in layer_processing_times.items():
        average_time = total_time / len(test_loader)
        print(f"{layer}: {average_time:.5f} seconds")


