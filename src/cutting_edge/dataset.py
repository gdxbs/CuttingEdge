import json
import os
from typing import Dict

import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image


class DatasetLoader:
    """Loader for GarmentCodeData dataset

    This class handles loading and preprocessing of the GarmentCodeData dataset,
    which contains 3D garment models with corresponding sewing patterns.

    References:
    - GarmentCodeData: A Dataset of 3D Made-to-Measure Garments with Sewing Patterns
      (Korosteleva et al., ECCV 2024)
    - Dataset available at: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
    """

    def __init__(self, dataset_path: str):
        """Initialize the dataset loader

        Args:
            dataset_path: Path to the root directory of the GarmentCodeData dataset
        """
        self.dataset_path = dataset_path
        self.train_valid_test_split = self._load_split()

    def _load_split(self) -> Dict:
        """Load official train/valid/test split from the dataset

        Returns:
            Dictionary containing train/valid/test splits
        """
        # Reference to the official ETH Zürich dataset repository
        # REF: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
        # os.chdir("/mnt/d/downloads/690432/690432")
        # Use dataset_path instead of hardcoded path
        split_path = os.path.join(
            "GarmentCodeData_v2_official_train_valid_test_data_split_filtered.json",
        )

        with open(split_path, "r") as f:
            return json.load(f)

    def load_pattern(self, pattern: str) -> Dict:
        """Load pattern data for a specific garment element

        Args:
            element_name: Name of the garment element (e.g., 'rand_RVS1QWPXD0')
            batch_id: Batch ID for the garment data

        Returns:
            Dictionary containing pattern specification, image, and metadata
        """
        pattern = f"{pattern}/{pattern.split('/')[-1]}"

        base_path = os.path.join(self.dataset_path, pattern)

        with open(
            f"{base_path.rsplit('/rand_', 2)[0]}/dataset_properties_default_body.yaml",
            "r",
        ) as f:
            dataset_properties = yaml.safe_load(f)
            type = dataset_properties["generator"]["stats"]["garment_types"][
                base_path.split("/")[-1]
            ]["main"]

        # Load pattern specification
        # Sewing pattern specification follows the format described in:
        # "GarmentCode: Physics-based automatic patterning of 3D garment models" [Korosteleva and Lee 2021]
        # REF: https://doi.org/10.1145/3478513.3480489
        # with open(f"{base_path}_specification.json", "r") as f:
        #     specification = json.load(f)

        # Load pattern image
        # Contains 2D vector graphics of sewing pattern panels and their relative locations
        pattern_img = Image.open(f"{base_path}_pattern.png")

        # Load design parameters
        # These parameters control the garment's design aspects (e.g., sleeve length, collar width)
        with open(f"{base_path}_design_params.yaml", "r") as f:
            design_params = yaml.safe_load(f)

        # Load segmentation
        # Contains panel labels for each vertex in the 3D mesh, with stitch vertices labeled separately
        # This enables mapping between 3D garment and 2D pattern pieces
        # with open(f"{base_path}_sim_segmentation.txt", "r") as f:
        #     segmentation = f.read().splitlines()

        # Extract dimensions from design parameters
        # pattern_size parameter contains width and height in centimeters (real-world scale)
        dimensions = design_params.get("pattern_size", [256, 256])

        # If dimensions are not in the expected format, log and use default values
        # Default of 256x256 is a common standardized size for pattern visualization
        if not isinstance(dimensions, list) or len(dimensions) != 2:
            print(
                f"Warning: Invalid dimensions format in {pattern}_design_params.yaml. Using default [256, 256]."
            )
            dimensions = [256, 256]

        return {
            "type": type.split("_")[0],
            # "specification": specification,
            "pattern_image": pattern_img,
            # "design_params": design_params,
            # "segmentation": segmentation,
            "dimensions": dimensions,
        }


class PatternDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for garment pattern recognition training

    This class processes the GarmentCodeData dataset into a format suitable for
    training deep learning models for pattern recognition. It handles data loading,
    preprocessing, and batch generation.

    Implements the PyTorch Dataset interface for compatibility with DataLoader.
    """

    def __init__(self, dataset_loader: DatasetLoader, split: str = "train"):
        """Initialize the pattern dataset

        Args:
            dataset_loader: Instance of DatasetLoader to load pattern data
            split: Dataset split to use ('train', 'valid', or 'test')
        """
        self.loader = dataset_loader
        self.split = split
        self.pattern_list = self.loader.train_valid_test_split[split]

        # Create mapping of pattern types to numerical indices for classification
        self.pattern_types = self._create_pattern_type_mapping()

    def __len__(self) -> int:
        """Return the number of patterns in the dataset

        Required for PyTorch DataLoader compatibility
        """
        return len(self.pattern_list)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single pattern sample by index

        Required for PyTorch DataLoader compatibility. Loads and preprocesses
        a pattern sample for model training or evaluation.

        Args:
            idx: Index of the pattern in the dataset

        Returns:
            Dictionary containing processed image tensor, label, and metadata
        """

        try:
            pattern_info = self.pattern_list[idx]
            pattern_data = self.loader.load_pattern(pattern_info)

            # Process pattern image for model input
            pattern_img = self._preprocess_image(pattern_data["pattern_image"])

            # Convert pattern type to numerical label using mapping
            pattern_type = pattern_data["type"]
            if pattern_type not in self.pattern_types:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            label = self.pattern_types[pattern_type]

            # Convert dimensions to tensor for regression tasks
            dimensions = torch.tensor(pattern_data["dimensions"], dtype=torch.float32)
            ## print({"image": pattern_img,
            ##    "label": label,
            ##    "specification": pattern_data["specification"],
            ##    "design_params": pattern_data["design_params"],
            ##    "dimensions": dimensions})
            # Return only the essential fields needed for training
            # Skip unnecessary fields like specification to avoid collation issues
            return {
                "image": pattern_img,
                "label": label,
                # "specification": pattern_data["specification"],
                ## "design_params": pattern_data["design_params"],
                "dimensions": dimensions,
            }
        except (KeyError, ValueError, Exception) as e:
            print(e)
            return {
                "image": torch.zeros((3, 512, 512)),  # Default empty image
                "label": 0,  # Default label
                "dimensions": torch.tensor(
                    [256.0, 256.0], dtype=torch.float32
                ),  # Default dimensions
            }

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        """Preprocess pattern image for model input

        Applies standard image preprocessing steps including resizing to a fixed size,
        tensor conversion, and normalization using ImageNet statistics.

        Args:
            image: PIL Image object of the pattern

        Returns:
            Preprocessed image as a normalized PyTorch tensor
        """

        # Standardized preprocessing pipeline for pattern images
        # - Resize to 512x512 to ensure consistent input dimensions
        # - Convert to tensor (0-1 range, CHW format)
        # - Normalize using ImageNet statistics for transfer learning compatibility
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),  # Standard size for model input
                transforms.ToTensor(),  # Convert to tensor (0-1 range)
                transforms.Normalize(
                    # ImageNet mean and std values (standard for pretrained models)
                    # REF: https://pytorch.org/vision/stable/models.html
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Convert RGBA images to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        return transform(image)

    def _create_pattern_type_mapping(self) -> Dict:
        """Create mapping of pattern types to numerical indices

        Builds a dictionary that maps textual pattern types to integer indices
        for model training. Ensures consistent labeling across training runs.

        Returns:
            Dictionary mapping pattern types (str) to indices (int)
        """

        # Collect all unique pattern types across the dataset split
        pattern_types = set()

        for pattern in self.pattern_list:
            # print(type(pattern))
            pattern_data = self.loader.load_pattern(pattern)
            # print(list(pattern_data['specification']['pattern']['panels'])[0].split('_')[0])
            pattern_types.add(pattern_data["type"])

        # Sort types alphabetically to ensure consistent indices
        # This is important for reproducibility and model loading/saving
        return {ptype: idx for idx, ptype in enumerate(sorted(pattern_types))}
