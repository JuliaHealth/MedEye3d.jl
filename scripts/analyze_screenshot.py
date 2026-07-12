import cv2
import numpy as np

def analyze_quadrants(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
        
    h, w = img.shape
    
    # Define the 4 quadrants (top-left, top-right, bottom-left, bottom-right)
    # Give a small margin to avoid any crosshairs or borders
    margin = 5
    quadrants = {
        "Axial (Top-Left)": img[margin:h//2 - margin, margin:w//2 - margin],
        "Coronal (Top-Right)": img[margin:h//2 - margin, w//2 + margin:w - margin],
        "Sagittal (Bottom-Left)": img[h//2 + margin:h - margin, margin:w//2 - margin],
        "Axial (Bottom-Right)": img[h//2 + margin:h - margin, w//2 + margin:w - margin]
    }
    
    for name, quad in quadrants.items():
        # Threshold to get the bright sphere (ignore background)
        _, thresh = cv2.threshold(quad, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"[{name}] No object found in quadrant!")
            continue
            
        # Get the largest contour (the sphere)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w_box) / h_box
        
        print(f"[{name}] Bounding Box: {w_box}x{h_box} pixels")
        print(f"[{name}] Aspect Ratio (Width/Height): {aspect_ratio:.3f}")
        
        if 0.9 <= aspect_ratio <= 1.1:
            print(f"[{name}] Pass: Object is circular (proportional).")
        else:
            print(f"[{name}] Fail: Object is severely stretched/squished.")
        print("-" * 40)

if __name__ == "__main__":
    analyze_quadrants("data/screenshot_medeye.png")
