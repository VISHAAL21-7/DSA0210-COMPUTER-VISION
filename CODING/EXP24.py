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

# ---------- Convert to Binary Image ----------
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# ---------- Structuring Element ----------
kernel = np.ones((5, 5), np.uint8)

# ---------- Apply Erosion ----------
eroded = cv2.erode(binary, kernel, iterations=1)

# ---------- Display Results ----------
cv2.imshow("Original Binary Image", binary)
cv2.imshow("Eroded Image", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()
