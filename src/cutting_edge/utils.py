import cv2
import numpy as np
import torchvision.transforms as transforms

from cutting_edge.config import IMAGE_PROCESSING


def preprocess_image_for_model(
    image,
    device=None,
    is_cv2_image=True,
    size=IMAGE_PROCESSING["DEFAULT_PREPROCESS_SIZE"],
):
    """Preprocess image for model input

    Standard preprocessing pipeline for both cloth and pattern images:
    - Convert BGR to RGB if using cv2 image
    - Resize to standard size (default 512x512)
    - Convert to tensor (0-1 range, CHW format)
    - Normalize using ImageNet statistics
    - Add batch dimension and move to specified device

    Args:
        image: Input image (cv2 BGR format or PIL image)
        device: Computation device (CPU/GPU)
        is_cv2_image: Whether the image is in cv2 BGR format (True) or PIL/RGB (False)
        size: Target image size

    Returns:
        Preprocessed image tensor with batch dimension
    """
    transform = transforms.Compose(
        [
            transforms.ToPILImage() if is_cv2_image else transforms.Lambda(lambda x: x),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_PROCESSING["IMAGENET_MEAN"],  # ImageNet mean
                std=IMAGE_PROCESSING["IMAGENET_STD"],  # ImageNet std
            ),
        ]
    )

    # Convert BGR to RGB if needed
    if is_cv2_image and len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Apply transform
    processed_tensor = transform(image_rgb).float()

    # Add batch dimension
    processed_tensor = processed_tensor.unsqueeze(0)

    # Move to device if specified
    if device is not None:
        processed_tensor = processed_tensor.to(device)

    return processed_tensor


def extract_contours(image, min_area_ratio=IMAGE_PROCESSING["MIN_AREA_RATIO"]):
    """Extract contours from an image using multiple techniques

    This function applies several detection methods to find contours:
    1. Color-based segmentation
    2. Otsu thresholding
    3. Adaptive thresholding

    Args:
        image: Input image
        min_area_ratio: Minimum area ratio to filter small contours

    Returns:
        List of contours sorted by area (largest first)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Try HSV color segmentation first
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, binary = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If not enough pixels detected, try grayscale with inverse
    if np.sum(binary) / binary.size < IMAGE_PROCESSING["HSV_SATURATION_THRESHOLD"]:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up binary image
    kernel = np.ones(IMAGE_PROCESSING["MORPH_KERNEL_SIZE"], np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    min_area = min_area_ratio * gray.shape[0] * gray.shape[1]
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Try adaptive thresholding if no good contours found
    if not filtered_contours:
        binary_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            IMAGE_PROCESSING["ADAPTIVE_THRESHOLD_BLOCK_SIZE"],
            IMAGE_PROCESSING["ADAPTIVE_THRESHOLD_C"],
        )
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Sort by area and return
    if filtered_contours:
        return sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    # Fallback to simple rectangle if no contours found
    h, w = gray.shape[:2]
    margin_x, margin_y = int(w * IMAGE_PROCESSING["MARGIN_RATIO"]), int(
        h * IMAGE_PROCESSING["MARGIN_RATIO"]
    )
    simple_contour = np.array(
        [
            [[margin_x, margin_y]],
            [[w - margin_x, margin_y]],
            [[w - margin_x, h - margin_y]],
            [[margin_x, h - margin_y]],
        ],
        dtype=np.int32,
    )

    return [simple_contour]
