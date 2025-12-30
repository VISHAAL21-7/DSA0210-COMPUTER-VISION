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

# Image dimensions
h, w = img.shape[:2]

# ---------- Define Corresponding Points ----------
# Source points (original image)
src_pts = np.float32([
    [50, 50],
    [w - 50, 50],
    [50, h - 50],
    [w - 50, h - 50]
])

# Destination points (mapped positions)
dst_pts = np.float32([
    [100, 120],
    [w - 150, 80],
    [120, h - 100],
    [w - 120, h - 80]
])

# ---------- Direct Linear Transformation ----------
# findHomography internally uses DLT
H, status = cv2.findHomography(src_pts, dst_pts, 0)

# ---------- Apply Transformation ----------
dlt_output = cv2.warpPerspective(img, H, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("DLT Transformed Image", dlt_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
