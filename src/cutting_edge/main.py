import logging

import cv2
import matplotlib.pyplot as plt
import torch

from cutting_edge.cloth_recognition_module import ClothRecognitionModule
from cutting_edge.pattern_recognition_module import PatternRecognitionModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""

    try:
        dataset_path = "path/to/GarmentCodeData" # Path to dataset (this will be sent for training, and after training this needs to be empty)
        model_path = "final_pattern_recognition_model.pth" # Path to saved model so that we dont have to keep training the model

        # Initialize modules
        pattern_recognizer = PatternRecognitionModule(
            dataset_path=dataset_path, model_path=model_path
        )
        cloth_recognizer = ClothRecognitionModule()

        # Check and train model
        pattern_recognizer.check_and_train(model_path)

        # Load images
        pattern_image = cv2.imread("pattern.jpg")
        cloth_image = cv2.imread("cloth.jpg")

        # Process pattern and cloth
        pattern_data = pattern_recognizer.process_pattern(pattern_image)
        cloth_data = cloth_recognizer.process_cloth(cloth_image)

        print("Pattern data:", pattern_data)
        print("Cloth data:", cloth_data)

        # Visualize results
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(pattern_image)
        plt.title("Input Pattern")
        plt.subplot(132)
        plt.imshow(cloth_image)
        plt.title("Input Cloth")
        plt.subplot(133)
        # plt.imshow(final_output["visualization"])
        plt.title("Pattern Placement")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except torch.cuda.OutOfMemoryError:
        print("Error: GPU memory exceeded. Try reducing batch size.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
