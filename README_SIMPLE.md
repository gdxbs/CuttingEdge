# Cutting Edge: Simplified Pattern Fitting System

## Overview

This is a simplified version of the Cutting Edge system designed to be **easy to understand** for beginners in computer vision and reinforcement learning. The system automatically fits garment patterns onto cloth materials to minimize waste.

## Key Features

- **Automatic Dimension Extraction**: Reads dimensions from filenames (e.g., `pattern_50x80.png` = 50cm x 80cm)
- **Fallback Dimension Estimation**: Uses OpenCV to estimate dimensions if not in filename
- **Simple Neural Networks**: Uses basic CNNs instead of complex pre-trained models
- **Easy-to-Modify Magic Numbers**: All default values are clearly marked and easy to change
- **Comprehensive Logging**: Detailed logs at every step for transparency
- **Multi-Pattern Support**: Can fit multiple patterns on a single cloth
- **Automatic Data Splitting**: Automatically splits data into train/test sets
- **Model Persistence**: Saves and loads models automatically

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

### 2. Prepare Your Images

Create two folders with your images:
- `images/shape/`: Pattern images
- `images/cloth/`: Cloth images

**Naming Convention**: Use format `name_WIDTHxHEIGHT.extension`
- Example: `pattern_50x80.png` (50cm wide, 80cm tall)
- Example: `cloth_200x300.jpeg` (200cm wide, 300cm tall)

If dimensions aren't in filename, the system will use defaults or estimate them.

### 3. Run the System

#### Single Pattern Fitting
```bash
# Fit a specific pattern to a specific cloth
python -m src.cutting_edge.simple_main \
    --pattern images/shape/pattern_50x80.png \
    --cloth images/cloth/cloth_200x300.jpeg

# Run on automatically selected test images
python -m src.cutting_edge.simple_main --mode inference
```

#### Multiple Pattern Fitting
```bash
# Fit multiple patterns to a single cloth
python -m src.cutting_edge.simple_main --multi_pattern

# This will automatically select patterns from test set
```

#### Training Mode
```bash
# Train and save models (simplified - just saves current model state)
python -m src.cutting_edge.simple_main --mode train

# Train and then run inference
python -m src.cutting_edge.simple_main --mode both
```

## System Architecture

### 1. Simple Pattern Recognition (`simple_pattern_recognition.py`)
- **SimplePatternCNN**: Basic 3-layer CNN for feature extraction
- **Dimension extraction**: From filename or OpenCV estimation
- **Default dimensions**: 100x150 cm (easily changeable)

### 2. Simple Cloth Recognition (`simple_cloth_recognition.py`)
- **SimpleClothCNN**: Basic CNN for cloth analysis
- **Usable area calculation**: Accounts for edge margins (5cm default)
- **Default dimensions**: 200x300 cm (easily changeable)

### 3. Simple Pattern Fitting (`simple_pattern_fitting.py`)
- **SimpleFittingAgent**: Basic neural network for placement decisions
- **Reward system**: 
  - Bonus for edge placement (+2)
  - Bonus for good utilization (+5)
  - Penalty for overlaps (-10)
- **Visualization**: Automatic generation of result images

### 4. Main System (`simple_main.py`)
- **Automatic data splitting**: 80/20 train/test split
- **Model management**: Auto save/load from `models/` directory
- **Comprehensive logging**: Every step is logged
- **Metrics tracking**: Saves all results to JSON

## Magic Numbers (Easy to Modify)

All default values are clearly marked in the code:

```python
# In simple_pattern_recognition.py
DEFAULT_WIDTH = 100   # Default pattern width in cm
DEFAULT_HEIGHT = 150  # Default pattern height in cm
IMAGE_SIZE = 256      # Size for neural network input

# In simple_cloth_recognition.py
DEFAULT_WIDTH = 200   # Default cloth width in cm
DEFAULT_HEIGHT = 300  # Default cloth height in cm
EDGE_MARGIN = 5       # Safety margin from edges in cm

# In simple_pattern_fitting.py
GRID_SIZE = 10        # Placement grid resolution
MAX_ATTEMPTS = 100    # Attempts to place each pattern
OVERLAP_PENALTY = -10 # Penalty for overlapping patterns
EDGE_BONUS = 2        # Bonus for efficient edge placement
UTILIZATION_BONUS = 5 # Bonus for good space usage
```

## Output Files

The system generates several outputs:

1. **Visualizations**: `output/fitting_result_*.png`
   - Shows cloth with placed patterns
   - Different colors for each pattern
   - Displays utilization percentage

2. **Reports**: `output/fitting_result_*_report.txt`
   - Detailed placement information
   - Material utilization statistics
   - Individual pattern positions

3. **Metrics**: `output/fitting_metrics.json`
   - Historical record of all fittings
   - Utilization percentages
   - Success rates

4. **Data Split**: `data/data_split.json`
   - Record of train/test split
   - Ensures reproducibility

5. **Models**: `models/`
   - `pattern_model.pth`: Pattern recognition model
   - `cloth_model.pth`: Cloth recognition model
   - `fitting_model.pkl`: Pattern fitting agent

## Example Output

```
PATTERN FITTING REPORT
==================================================

Total Patterns: 3
Patterns Placed: 3
Success Rate: 100.0%
Material Utilization: 9.9%
Cloth Dimensions: 190.0 x 290.0 cm
Total Pattern Area: 5438.6 cm²
Wasted Area: 49661.4 cm²

PLACEMENT DETAILS
--------------------------------------------------
Pattern 1: Position (77.0, 108.4), Rotation 2.8°
Pattern 2: Position (97.4, 143.7), Rotation 3.6°
Pattern 3: Position (50.4, 66.8), Rotation -14.9°
```

## Understanding the Code

The system is designed for beginners:

1. **Clear Comments**: Every function has detailed comments
2. **Simple Architecture**: Basic CNNs instead of complex models
3. **No Complex RL**: Uses simplified learning approach
4. **Verbose Logging**: See exactly what's happening at each step
5. **Modular Design**: Each module handles one specific task

## Customization

### Adding New Features
1. Modify magic numbers in respective files
2. Add new reward components in `PatternFitter.calculate_reward()`
3. Extend state representation in `PatternFitter.create_state()`

### Changing Neural Networks
- Pattern CNN: Modify `SimplePatternCNN` class
- Cloth CNN: Modify `SimpleClothCNN` class  
- Fitting Agent: Modify `SimpleFittingAgent` class

### Adjusting Dimensions
- Change `DEFAULT_WIDTH/HEIGHT` in respective modules
- Modify `pixel_to_cm` ratio in OpenCV estimation
- Adjust `EDGE_MARGIN` for safety margins

## Troubleshooting

1. **No dimensions found**: Add dimensions to filename or adjust defaults
2. **Poor fitting results**: Increase `MAX_ATTEMPTS` or adjust reward values
3. **Overlapping patterns**: Check `OVERLAP_PENALTY` value
4. **Low utilization**: Try smaller patterns or larger cloth

## Development

```bash
# Run tests
pytest

# Format code
black .
isort .

# Lint
ruff check .
mypy .
```

## Key Differences from Original System

1. **Simplified Models**: Basic CNNs instead of ResNet50/EfficientNet
2. **Clear Defaults**: All magic numbers are explicit and documented
3. **Simple RL**: Basic neural network instead of complex HRL
4. **Automatic Workflow**: Handles everything from folder scanning to results
5. **Better Logging**: Much more detailed output for learning

This simplified version maintains the core functionality while being much easier to understand and modify for beginners in computer vision and reinforcement learning.