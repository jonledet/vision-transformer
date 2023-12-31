import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.multiprocessing as mp
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time

class Timing:
    """
    Class to measure execution time of different components in the ViT model.

    Attributes:
        input_patching_time (float): Time spent on input patching.
        pos_embedding_time (float): Time spent on positional encoding.
        trans_encoder_time (float): Time spent on the transformer encoder.
        linear_embedding_time (float): Time spent on linear embedding.
        avg_pooling_time (float): Time spent on pooling.
        avg_connected_layers_time (float): Time spent on connected layers.
        avg_output_time (float): Time spent on output layer.
    """
    def __init__(self):
        self.input_patching_time = 0
        self.pos_embedding_time = 0
        self.trans_encoder_time = 0
        self.linear_embedding_time = 0
        self.avg_pooling_time = 0
        self.avg_connected_layers_time = 0
        self.avg_output_time = 0

    def calculate_avg(self, num_epochs, num_batches):
        """
        Calculate average timing values.

        Args:
            num_epochs (int): Number of training epochs.
            num_batches (int): Number of batches in each epoch.
        """
        total = num_epochs * num_batches
        self.avg_pooling_time = self.avg_pooling_time / total
        self.avg_connected_layers_time = self.avg_connected_layers_time / total
        self.avg_output_time = self.avg_output_time / total


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    Args:
        image_size (tuple): Input image dimensions (height, width).
        patch_size (tuple): Patch dimensions (height, width).
        num_classes (int): Number of classes for classification.
        dim (int): Dimensionality of the model's embedding space.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of the MLP (FeedForward) layers.
        timing (Timing): Timing object for measuring execution time.
        pool (str): Pooling type, either 'cls' (cls token) or 'mean' (mean pooling).
        channels (int): Number of input channels (default is 3 for RGB images).
        dim_head (int): Dimensionality of each attention head.
        dropout (float): Dropout rate for model layers.
        emb_dropout (float): Dropout rate for the embedding layer.
    """
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, timing, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # Measure time for Input Patching
        input_patching_start = time.time()
        self.input_patching = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1])
        self.to_patch_embedding = nn.Sequential(
            self.input_patching,
            nn.Linear(channels * patch_size[0] * patch_size[1], dim),
        )
        input_patching_end = time.time()
        timing.input_patching_time = input_patching_end - input_patching_start

        # Measure time for Positional Encoding
        pos_embedding_start = time.time()
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) + 1, dim))
        pos_embedding_end = time.time()
        timing.pos_embedding_time = pos_embedding_end - pos_embedding_start

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Measure time for Transformer Encoder
        trans_encoder_start = time.time()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        trans_encoder_end = time.time()
        timing.trans_encoder_time = trans_encoder_end - trans_encoder_start

        self.pool = pool
        self.to_latent = nn.Identity()

        # Measure time for Linear Embedding
        linear_embedding_start = time.time()
        self.linear_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(channels * patch_size[0] * patch_size[1], dim),
        )
        linear_embedding_end = time.time()
        timing.linear_embedding_time = linear_embedding_end - linear_embedding_start

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, timing):
        """
        Forward pass of the ViT model.

        Args:
            img (torch.Tensor): Input image tensor.
            timing (Timing): Timing object for measuring execution time.

        Returns:
            list: List containing model outputs and timestamp for average pooling start.
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        avg_pooling_start = time.time()
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        avg_pooling_end = time.time()
        timing.avg_pooling_time += avg_pooling_end - avg_pooling_start

        avg_connected_layers_start = time.time()
        x = self.to_latent(x)
        avg_connected_layers_end = time.time()
        timing.avg_connected_layers_time += avg_connected_layers_end - avg_connected_layers_start

        avg_output_start = time.time()
        return [self.mlp_head(x), avg_output_start]


class Transformer(nn.Module):
    """
    Transformer layer for the ViT model.

    Args:
        dim (int): Dimensionality of the input and output feature vectors.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
        mlp_dim (int): Dimensionality of the MLP (FeedForward) layers.
        dropout (float): Dropout rate for model layers.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Forward pass of the transformer layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed output tensor.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    """
    Pre-Normalization module.

    Args:
        dim (int): Dimensionality of the input feature vectors.
        fn (nn.Module): Sub-module to be applied after normalization.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the Pre-Normalization module.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    FeedForward (MLP) module.

    Args:
        dim (int): Dimensionality of the input and output feature vectors.
        hidden_dim (int): Dimensionality of the hidden layer.
        dropout (float): Dropout rate for the module.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Attention module.

    Args:
        dim (int): Dimensionality of the input and output feature vectors.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
        dropout (float): Dropout rate for the module.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__ == "__main__":
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Create model and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')
    timing = Timing()
    model = ViT(image_size=(32, 32), patch_size=(4, 4), num_classes=10, dim=256, depth=3, heads=8, mlp_dim=512,
                channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1, timing=timing).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 100
    num_batches = len(cifar_loader)

    for epoch in range(num_epochs):
        model.train()
        # Iterate over batches
        for data in tqdm(cifar_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the gradients, forward pass, backward pass, and optimization
            optimizer.zero_grad()
            out_arr = model(inputs, timing=timing)
            timing.avg_output_time += time.time() - out_arr[1]
            outputs = out_arr[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Calculate average timing values
    timing.calculate_avg(num_epochs, num_batches)
    print(f'Input Patching Time: {timing.input_patching_time:.12f} seconds\n'
          f'Positional Encoding Time: {timing.pos_embedding_time:.12f} seconds\n'
          f'Transformer Encoder Time: {timing.trans_encoder_time:.12f} seconds\n'
          f'Linear Embedding Time: {timing.linear_embedding_time:.12f} seconds\n'
          f'Global Average Pooling Time: {timing.avg_pooling_time:.12f} seconds\n'
          f'Fully Connected Layers Time: {timing.avg_connected_layers_time:.12f} seconds\n'
          f'Output Layer Time: {timing.avg_output_time:.12f} seconds\n'
          )

    # Evaluate the model
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        # Iterate over batches for evaluation
        for data in tqdm(cifar_loader, desc='Evaluating'):
            inputs, labels = data[0].to(device), data[1].to(device)
            out_arr = model(inputs, timing=timing)
            outputs = out_arr[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

