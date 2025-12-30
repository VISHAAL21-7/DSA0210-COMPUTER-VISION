import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Unicode-safe image loading ----------
image_path = r"C:/Users/msvis/Documents/CVISION/input.jpg"

with open(image_path, "rb") as f:
    data = np.frombuffer(f.read(), np.uint8)

img = cv2.imdecode(data, cv2.IMREAD_COLOR)

# Safety check
if img is None:
    print("Error: Image not loaded.")
    exit()

# ---------- a) Convert to Grayscale ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- b) Gaussian Blur ----------
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

# ---------- c) Canny Edge Detection ----------
edges = cv2.Canny(gray, 100, 200)

# ---------- d) Dilation ----------
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# ---------- e) Erosion ----------
eroded = cv2.erode(edges, kernel, iterations=1)

# ---------- Display Results ----------
titles = [
    "Original Image",
    "Grayscale Image",
    "Gaussian Blurred Image",
    "Canny Edge Detection",
    "Dilated Image",
    "Eroded Image"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    gray,
    gaussian_blur,
    edges,
    dilated,
    eroded
]

plt.figure(figsize=(15, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray" if i != 0 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
