import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.multiprocessing as mp
import time

# Use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvTransformer(nn.Module):
    """
    Convolutional Transformer model for image classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool1 (nn.MaxPool2d): First max-pooling layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        pool2 (nn.MaxPool2d): Second max-pooling layer.
        fc1 (nn.Linear): First fully connected layer.
        transformer_layer (nn.TransformerEncoderLayer): Transformer layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer.

    Methods:
        forward(x): Forward pass through the model.
    """
    def __init__(self):
        super(ConvTransformer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Output tensor and layer processing times.
        """
        start_time = time.time()

        x = self.pool1(torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        layer1_time = time.time() - start_time

        start_time = time.time()
        x = self.pool2(torch.relu(self.conv3(x)))
        layer2_time = time.time() - start_time

        x = x.view(-1, 32 * 8 * 8)

        start_time = time.time()
        x = torch.relu(self.fc1(x))
        layer3_time = time.time() - start_time

        x = x.unsqueeze(0)

        start_time = time.time()
        x = self.transformer_layer(x)
        transformer_time = time.time() - start_time

        x = x.squeeze(0)

        start_time = time.time()
        x = torch.relu(self.fc2(x))
        layer4_time = time.time() - start_time

        start_time = time.time()
        x = self.fc3(x)
        layer5_time = time.time() - start_time

        layer_times = {
            "Conv1": layer1_time,
            "Conv2": layer2_time,
            "FC1": layer3_time,
            "Transformer": transformer_time,
            "FC2": layer4_time,
            "FC3": layer5_time,
        }

        return x, layer_times


if __name__ == '__main__':
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn')

    # Instantiate ConvTransformer model and move it to the selected device
    model = ConvTransformer().to(device)

    # Define loss function, optimizer, and learning rate scheduler
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Load CIFAR-10 dataset and create DataLoader instances for training and validation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_size = int(0.8 * len(cifar10_dataset))
    val_size = len(cifar10_dataset) - train_size
    train_dataset, val_dataset = random_split(cifar10_dataset, [train_size, val_size])
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Number of training epochs
    num_epochs = 100

    # Lists to store validation loss and accuracy
    validation_losses = []
    validation_accuracies = []

    # Early stopping parameters
    early_stopping_patience = 100
    early_stopping_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Iterate over batches in the training DataLoader
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # Forward pass through the model
            outputs, layer_times = model(inputs)

            # Compute loss and perform backward pass
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

        # Print training loss for the current epoch
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(trainloader)}")

    # Validation and average layer processing times
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    layer_processing_times = {}

    # Perform validation on the validation DataLoader
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, layer_times = model(images)

            # Accumulate layer processing times
            for layer, time_elapsed in layer_times.items():
                if layer in layer_processing_times:
                    layer_processing_times[layer] += time_elapsed
                else:
                    layer_processing_times[layer] = time_elapsed

            # Compute validation loss and accuracy
            val_loss += criteria(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print validation loss and accuracy
    val_loss /= len(valloader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss}, Validation Acc: {val_accuracy:.2f}%")

    # Print average layer processing times
    print("\nAverage Layer Processing Times:")
    for layer, total_time in layer_processing_times.items():
        average_time = total_time / len(valloader)
        print(f"{layer}: {average_time:.5f} seconds")
