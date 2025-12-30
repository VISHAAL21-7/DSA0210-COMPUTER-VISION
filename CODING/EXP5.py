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

# Get image dimensions and center
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

# ---------- Clockwise Rotation (−45 degrees) ----------
rotation_matrix_cw = cv2.getRotationMatrix2D(center, -45, 1.0)
clockwise = cv2.warpAffine(img, rotation_matrix_cw, (w, h))

# ---------- Counter-Clockwise Rotation (+45 degrees) ----------
rotation_matrix_ccw = cv2.getRotationMatrix2D(center, 45, 1.0)
counter_clockwise = cv2.warpAffine(img, rotation_matrix_ccw, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Clockwise Rotation", clockwise)
cv2.imshow("Counter-Clockwise Rotation", counter_clockwise)

cv2.waitKey(0)
cv2.destroyAllWindows()
