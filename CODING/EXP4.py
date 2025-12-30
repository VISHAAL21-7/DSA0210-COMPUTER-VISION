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
    print("Error: Image not loaded")
    exit()

# Convert BGR to RGB for correct display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Scaling ----------
# Bigger image (2x scaling)
bigger = cv2.resize(
    img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
)

# Smaller image (0.5x scaling)
smaller = cv2.resize(
    img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
)

# ---------- Display ----------
titles = ["Original Image", "Bigger Image (2x)", "Smaller Image (0.5x)"]
images = [img_rgb, bigger, smaller]

plt.figure(figsize=(12, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
