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

# ---------- Sobel Edge Detection (X-axis) ----------
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,  # output depth
    1, 0,        # dx=1, dy=0 → X direction
    ksize=3
)

# Convert to absolute values and uint8
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Sobel Edge Detection - X Axis", sobel_x_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()
