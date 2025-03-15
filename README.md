# Cutting Edge: Garment Pattern Recognition System

## Overview

Cutting Edge is a computer vision system that analyzes garment patterns and cloth materials using deep learning. The system helps in garment manufacturing by recognizing pattern types, detecting key features, and estimating dimensions for optimal pattern placement.

![System Diagram](https://via.placeholder.com/800x400?text=Cutting+Edge+System+Architecture)

## Features

- **Pattern Recognition**: Identifies pattern types from images
- **Cloth Material Analysis**: Analyzes cloth properties and contours
- **Dimension Estimation**: Extracts accurate dimensions for both patterns and cloth
- **Smart Placement**: (Planned feature) Automatically suggests optimal pattern placement

## How It Works

### 1. Pattern Recognition Module

The Pattern Recognition Module processes garment pattern images using multiple deep learning components:

- **CNN Backbone (ResNet50)**: Extracts visual features from pattern images
- **Classification Head**: Identifies the pattern type (shirt, pants, dress, etc.)
- **Corner Detection (LSTM)**: Locates corner points on the pattern
- **Dimension Predictor**: Estimates the real-world dimensions of the pattern

![Pattern Recognition](https://via.placeholder.com/600x300?text=Pattern+Recognition+Process)

### 2. Cloth Recognition Module

The Cloth Recognition Module analyzes cloth materials using:

- **EfficientNet-B0**: Classifies cloth material types
- **U-Net Segmentation**: Creates pixel-level cloth segmentation masks
- **Contour Detection**: Identifies cloth boundaries with traditional computer vision
- **Dimension Mapper**: Estimates cloth dimensions for cutting planning

![Cloth Analysis](https://via.placeholder.com/600x300?text=Cloth+Analysis+Process)

### 3. Dataset System

The system works with the GarmentCodeData dataset, which includes:

- Sewing pattern specifications in JSON format
- 2D pattern images
- Design parameters in YAML format
- 3D garment mesh segmentation data

## Technical Architecture

### Key Components

1. **Pattern Recognition Module**: `pattern_recognition_module.py`
   - Based on ResNet50 with custom heads for different tasks
   - Trained with supervised learning on pattern types and dimensions

2. **Cloth Recognition Module**: `cloth_recognition_module.py`
   - Uses EfficientNet and U-Net architectures
   - Combines deep learning with traditional computer vision techniques

3. **Dataset Handling**: `dataset.py`
   - Loads and preprocesses the GarmentCodeData dataset
   - Handles train/validation/test splits
   - Transforms data for model training

4. **Main Application**: `main.py`
   - Command-line interface for the system
   - Handles training, inference, and visualization

### Deep Learning Models

The system uses several state-of-the-art deep learning architectures:

- **ResNet50**: A residual neural network with 50 layers, effective for image classification tasks
- **LSTM**: Long Short-Term Memory network for sequence modeling (corner detection)
- **EfficientNet-B0**: Lightweight but powerful CNN for cloth classification
- **U-Net**: Encoder-decoder architecture with skip connections for semantic segmentation

## Installation

```bash
# Clone the repository
git clone https://github.com/gdxbs/CuttingEdge
cd CuttingEdge

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

To exit the environment:

```bash
deactivate
```

## Usage

### Training

```bash
python -m cutting_edge.main --dataset_path /path/to/garment_data --train --epochs 50
```

### Inference

```bash
python -m cutting_edge.main --model_path models/pattern_recognition_model.pth --pattern_image path/to/pattern.jpg --cloth_image path/to/cloth.jpg
```

## Development

1. Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

2. Run tests:

```bash
pytest
```

3. Format code:

```bash
black .
isort .
```

4. Run linter:

```bash
ruff check .
mypy .
```

## Technical Details

### Pattern Recognition Pipeline

1. **Input**: Pattern image (PNG/JPG format)
2. **Preprocessing**: Resize to 512x512, normalize using ImageNet statistics
3. **Feature Extraction**: ResNet50 backbone extracts 2048-dimensional features
4. **Classification**: Fully connected layer predicts pattern type
5. **Corner Detection**: LSTM processes features to detect corners
6. **Dimension Estimation**: MLP predicts width and height from features
7. **Output**: Pattern type, dimensions, and contour information

### Cloth Analysis Pipeline

1. **Input**: Cloth material image (PNG/JPG format)
2. **Preprocessing**: Resize, normalize, and prepare for multiple models
3. **Feature Extraction**: EfficientNet extracts features
4. **Cloth Type Classification**: Identifies material type
5. **Semantic Segmentation**: U-Net generates pixel-level cloth masks
6. **Contour Detection**: OpenCV processes edges and contours
7. **Output**: Cloth properties, dimensions, and contour information

## Scientific Foundation

This project builds on several research papers:

- **Deep Residual Learning**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." <https://arxiv.org/abs/1512.03385>
- **EfficientNet**: Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for CNNs." <https://arxiv.org/abs/1905.11946>
- **U-Net**: Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." <https://arxiv.org/abs/1505.04597>
- **GarmentCode**: Korosteleva, J. & Lee, S. (2021). "GarmentCode: Physics-based automatic patterning of 3D garment models." <https://doi.org/10.1145/3478513.3480489>
- **GarmentCodeData**: Korosteleva, J., et al. (2024). "GarmentCodeData: A Dataset of 3D Made-to-Measure Garments with Sewing Patterns." ECCV 2024.

## Future Development

- **Pattern Placement Optimization**: Algorithmic placement of patterns on cloth to minimize waste
- **Multi-Pattern Layout**: Handling multiple patterns on a single cloth piece
- **Material-Specific Adjustments**: Adapting pattern placement based on cloth properties
- **3D Visualization**: Providing 3D preview of the final garment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The GarmentCodeData dataset from ETH Zürich
- PyTorch and OpenCV communities
- Segmentation Models PyTorch library
