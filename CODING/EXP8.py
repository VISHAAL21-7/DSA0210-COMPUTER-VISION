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

# ---------- Define 4 source points (from original image) ----------
pts1 = np.float32([
    [50, 50],        # top-left
    [w - 50, 50],    # top-right
    [50, h - 50],    # bottom-left
    [w - 50, h - 50] # bottom-right
])

# ---------- Define 4 destination points ----------
pts2 = np.float32([
    [100, 100],          # top-left
    [w - 150, 50],       # top-right
    [150, h - 100],      # bottom-left
    [w - 100, h - 50]    # bottom-right
])

# ---------- Perspective transformation ----------
perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

perspective_transformed = cv2.warpPerspective(img, perspective_matrix, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Perspective Transformed Image", perspective_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()
