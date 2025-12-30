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

# ---------- Sobel Edge Detection ----------
# Sobel X
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute values
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# ---------- Combine X and Y ----------
sobel_xy = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Sobel Edge Detection - XY Axis", sobel_xy)

cv2.waitKey(0)
cv2.destroyAllWindows()
