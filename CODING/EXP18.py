import cv2
import numpy as np

# ---------- Unicode-safe image loading ----------
image_path = r"C:/Users/msvis/Documents/CVISION/input.jpg"

with open(image_path, "rb") as f:
    data = np.frombuffer(f.read(), np.uint8)

img = cv2.imdecode(data, cv2.IMREAD_COLOR)

# Safety check
if img is None:
    print("Error: Image not loaded")
    exit()

# ---------- Convert to Grayscale ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Laplacian Mask with Positive Center ----------
laplacian_positive_kernel = np.array([
    [0, -1,  0],
    [-1,  5, -1],
    [0, -1,  0]
], dtype=np.float32)

# Apply Laplacian sharpening
sharpened = cv2.filter2D(gray, cv2.CV_64F, laplacian_positive_kernel)

# Convert to uint8 for display
sharpened_abs = cv2.convertScaleAbs(sharpened)

# ---------- Display Results ----------
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Sharpened Image (Positive Center Laplacian)", sharpened_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()
