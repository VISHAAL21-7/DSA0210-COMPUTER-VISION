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

# ---------- Canny Edge Detection ----------
edges = cv2.Canny(gray, 100, 200)

# ---------- Display Results ----------
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edge Detection", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
