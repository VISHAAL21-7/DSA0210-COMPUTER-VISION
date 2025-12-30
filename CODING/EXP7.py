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

# Get image size
(h, w) = img.shape[:2]

# ---------- Define points ----------
# Original image points (3 points)
pts1 = np.float32([
    [50, 50],
    [200, 50],
    [50, 200]
])

# Destination image points (shifted & tilted)
pts2 = np.float32([
    [10, 100],
    [220, 80],
    [80, 250]
])

# ---------- Affine transformation ----------
affine_matrix = cv2.getAffineTransform(pts1, pts2)

affine_transformed = cv2.warpAffine(img, affine_matrix, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Affine Transformed Image", affine_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()
