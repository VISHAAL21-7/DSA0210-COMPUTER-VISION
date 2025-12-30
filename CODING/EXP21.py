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

# ---------- Apply Sobel Gradient Masks ----------
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert to displayable format
gradient_abs = cv2.convertScaleAbs(gradient_magnitude)

# ---------- Sharpening using Gradient Mask ----------
sharpened = cv2.add(gray, gradient_abs)

# ---------- Display Results ----------
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Gradient Mask (Edges)", gradient_abs)
cv2.imshow("Sharpened Image (Gradient Masking)", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
