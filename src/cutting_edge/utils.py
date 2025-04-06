import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

def preprocess_image_for_model(image, device=None, is_cv2_image=True, size=(512, 512)):
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
    transform = transforms.Compose([
        transforms.ToPILImage() if is_cv2_image else transforms.Lambda(lambda x: x),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])
    
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

def extract_contours(image, min_area_ratio=0.01):
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
    if np.sum(binary) / binary.size < 0.05:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up binary image
    kernel = np.ones((5, 5), np.uint8)
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
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Sort by area and return
    if filtered_contours:
        return sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    
    # Fallback to simple rectangle if no contours found
    h, w = gray.shape[:2]
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    simple_contour = np.array([
        [[margin_x, margin_y]],
        [[w - margin_x, margin_y]],
        [[w - margin_x, h - margin_y]],
        [[margin_x, h - margin_y]],
    ], dtype=np.int32)
    
    return [simple_contour]