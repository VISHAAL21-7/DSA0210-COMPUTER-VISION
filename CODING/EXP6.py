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

# Get image dimensions
(h, w) = img.shape[:2]

# ---------- Translation values ----------
tx = 100   # move right by 100 pixels
ty = 50    # move down by 50 pixels

# Translation matrix
translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# Apply translation
moved = cv2.warpAffine(img, translation_matrix, (w, h))

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Moved Image", moved)

cv2.waitKey(0)
cv2.destroyAllWindows()
