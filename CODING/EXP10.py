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
h, w = img.shape[:2]

# ---------- Define corresponding points ----------
# Source points (from original image)
src_pts = np.float32([
    [50, 50],
    [w - 50, 60],
    [60, h - 60],
    [w - 60, h - 50]
])

# Destination points (mapped positions)
dst_pts = np.float32([
    [100, 100],
    [w - 150, 80],
    [120, h - 120],
    [w - 100, h - 100]
])

# ---------- Compute Homography Matrix ----------
H, status = cv2.findHomography(src_pts, dst_pts)

# ---------- Apply Homography ----------
homography_output = cv2.warpPerspective(img, H, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Homography Transformed Image", homography_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
