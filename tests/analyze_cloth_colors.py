
import cv2
import numpy as np
import os

def analyze_color(path):
    print(f"Analyzing: {os.path.basename(path)}")
    if not os.path.exists(path):
        print("File not found.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Failed to load image.")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate stats for the center area to ignore background/edges
    h, w = img.shape[:2]
    center = hsv[int(h/3):int(2*h/3), int(w/3):int(2*w/3)]
    
    mean_h = np.mean(center[:, :, 0])
    mean_s = np.mean(center[:, :, 1])
    mean_v = np.mean(center[:, :, 2])
    
    print(f"  Mean Hue: {mean_h:.1f}")
    print(f"  Mean Sat: {mean_s:.1f}")
    print(f"  Mean Val: {mean_v:.1f}")
    
    # Determine basic color category
    # OpenCV Hue is 0-180
    if mean_s < 30:
        print("  Category: Grayscale/White/Black")
    else:
        if 10 < mean_h < 30:
            print("  Category: Orange/Brown-ish")
        elif 90 < mean_h < 130:
            print("  Category: Blue-ish")
        else:
            print("  Category: Other")
    print("-" * 30)

if __name__ == "__main__":
    analyze_color(r"d:\cut\CuttingEdge\images\cloth\freeform\cloth_49_625x614.jpg") # Leather
    analyze_color(r"d:\cut\CuttingEdge\images\cloth\freeform\cloth_4_507x426.jpg")  # Cotton
