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

# ---------- Laplacian Mask with Diagonal Neighbors ----------
laplacian_diagonal_kernel = np.array([
    [1,  1,  1],
    [1, -8,  1],
    [1,  1,  1]
], dtype=np.float32)

# Apply Laplacian filter
laplacian = cv2.filter2D(gray, cv2.CV_64F, laplacian_diagonal_kernel)

# Convert to absolute values
laplacian_abs = cv2.convertScaleAbs(laplacian)

# ---------- Sharpening ----------
sharpened = cv2.add(gray, laplacian_abs)

# ---------- Display Results ----------
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Laplacian (Diagonal Neighbors)", laplacian_abs)
cv2.imshow("Sharpened Image", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
