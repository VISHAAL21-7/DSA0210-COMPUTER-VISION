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

# ---------- Option 1: Standard Laplacian Sharpening ----------
# Use a proper Laplacian kernel for edge detection (positive center)
laplacian_kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

# Apply Laplacian filter - use float32 to preserve negative values
laplacian = cv2.filter2D(gray.astype(np.float32), -1, laplacian_kernel)

# ---------- Sharpening ----------
# Subtract Laplacian from original for sharpening
# sharpened = original - laplacian (to enhance edges)
sharpened = gray.astype(np.float32) - laplacian

# Clip values to 0-255 range and convert to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# For display, we need to convert laplacian to absolute values
laplacian_display = cv2.convertScaleAbs(laplacian)

# ---------- Alternative: Built-in Laplacian ----------
# You can also use OpenCV's built-in function
laplacian_cv2 = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_cv2_abs = cv2.convertScaleAbs(laplacian_cv2)

# ---------- Alternative: Unsharp Masking (often better) ----------
gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

# ---------- Display Results ----------
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Laplacian Image (Custom)", laplacian_display)
cv2.imshow("Laplacian Image (OpenCV)", laplacian_cv2_abs)
cv2.imshow("Sharpened Image (Laplacian)", sharpened)
cv2.imshow("Sharpened Image (Unsharp Mask)", unsharp_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Save results for comparison ----------
cv2.imwrite("sharpened_laplacian.jpg", sharpened)
cv2.imwrite("sharpened_unsharp.jpg", unsharp_mask)
