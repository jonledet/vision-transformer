# Optimized Vision Transformer Training using GPU and Multi-threading

This repository contains an optimized implementation of Convolutional Neural Networks (CNN), Transformer, and Vision Transformer (ViT) models.

## Authors
Anonymous during paper submission process.

## Overview

This project focuses on optimizing Vision Transformer training using GPU acceleration and multi-threading techniques. It provides implementations of popular deep learning models, including Convolutional Neural Networks (CNN), Transformer, and a customized version of Vision Transformer (ViT) tailored for improved performance.

## Contents

- [CNN](cnn.py): Implementation of Convolutional Neural Networks.
- [Transformer](transformer.py): Implementation of the Transformer model.
- [ViT](vit.py): Customized version of the Vision Transformer (ViT) model, based on the [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10) repository.

## Getting Started

### Prerequisites

- Python (>=3.6)
- Anaconda 3
- PyTorch
- CUDA-enabled GPU (for GPU acceleration)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jonledet/vision-transformer.git
   ```

2. Create and activate a new Anaconda environment:

   ```bash
   conda create --name your-env-name python=3.6
   conda activate your-env-name
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

- To run the models, execute the corresponding Python script:

```bash
python cnn.py
```

```bash
python transformer.py
```

```bash
python vit.py
```

## Acknowledgments

- The Vision Transformer (ViT) model is based on the work from the [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10) repository.

## License

This project is licensed under the [MIT License](LICENSE).
