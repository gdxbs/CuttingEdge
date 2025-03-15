import json
import os
from typing import Dict

import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image


class DatasetLoader:
    """Loader for GarmentCodeData dataset"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.train_valid_test_split = self._load_split()

    def _load_split(self) -> Dict:
        """Load official train/valid/test split"""

        # REF: https://www.research-collection.ethz.ch/handle/20.500.11850/690432
        split_path = os.path.join(
            self.dataset_path,
            "GarmentCodeData_official_train_valid_test_data_split.json",
        )

        with open(split_path, "r") as f:
            return json.load(f)

    def load_pattern(self, element_name: str, batch_id: int) -> Dict:
        """Load pattern data for given element"""

        base_path = os.path.join(
            self.dataset_path,
            "GarmentCodeData",
            f"batch_{batch_id}",
            "default_body", # neutral body shape
            element_name,
        )

        # Load pattern specification
        # Sewing pattern specification in the [Korosteleva and Lee 2021] JSON format
        with open(f"{base_path}_specification.json", "r") as f:
            specification = json.load(f)

        # Load pattern image
        # Sewing pattern panels and their relative locations (projected onto 2D) in vector graphics format
        pattern_img = Image.open(f"{base_path}_pattern.png")

        # Load design parameters
        # List of design parameters and their values corresponding to the current design
        with open(f"{base_path}_design_params.yaml", "r") as f:
            design_params = yaml.safe_load(f)

        # Load segmentation
        # Segmentation of a garment mesh into panels: per-vertex panel labels, with stitch vertices labeled separately
        with open(f"{base_path}_sim_segmentation.txt", "r") as f:
            segmentation = f.read().splitlines()

        # TODO: Check if this is the correct dimension
        dimensions = design_params.get('pattern_size', [256, 256])

        return {
            "specification": specification,
            "pattern_image": pattern_img,
            "design_params": design_params,
            "segmentation": segmentation,
            "dimensions": dimensions,
        }


class PatternDataset(torch.utils.data.Dataset):
    """Dataset class for pattern recognition training"""

    def __init__(self, dataset_loader: DatasetLoader, split: str = "train"):
        self.loader = dataset_loader
        self.split = split
        self.pattern_list = self.loader.train_valid_test_split[split]

        # Create mapping of pattern types
        self.pattern_types = self._create_pattern_type_mapping()

    # Implement __len__ method similar to torch.utils.data.Dataset
    # Use can ignore it for now, as we primarily use it to make it compatible with PyTorch DataLoader 
    def __len__(self):
        return len(self.pattern_list)

    # Implement __getitem__ method similar to torch.utils.data.Dataset (UNUSED & CAN BE IGNORED FOR NOW)
    # Use can ignore it for now, as we primarily use it to make it compatible with PyTorch DataLoader 
    def __getitem__(self, idx):
        pattern_info = self.pattern_list[idx]
        pattern_data = self.loader.load_pattern(
            pattern_info["element_name"], pattern_info["batch_id"]
        )

        # Process pattern image
        pattern_img = self._preprocess_image(pattern_data["pattern_image"])

        # Create label tensor
        pattern_type = pattern_data["specification"]["type"]
        label = self.pattern_types[pattern_type]

        dimensions = torch.tensor(pattern_data["dimensions"], dtype=torch.float32)

        return {
            "image": pattern_img,
            "label": label,
            "specification": pattern_data["specification"],
            "design_params": pattern_data["design_params"],
            "dimensions": dimensions,
        }

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        """Preprocess pattern image for model input"""

        # TODO: Check for balance of normalization
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] # mean and stddev of ImageNet dataset
                ),
            ]
        )

        return transform(image)

    def _create_pattern_type_mapping(self) -> Dict:
        """Create mapping of pattern types to indices"""

        pattern_types = set()
        for pattern in self.pattern_list:
            pattern_data = self.loader.load_pattern(
                pattern["element_name"], pattern["batch_id"]
            )
            pattern_types.add(pattern_data["specification"]["type"])

        return {ptype: idx for idx, ptype in enumerate(sorted(pattern_types))}
