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

# ---------- Convert BGR to RGB for Matplotlib display ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Watermark Text ----------
watermark_text = "JERIDS"

# Get image dimensions
h, w = img.shape[:2]

# Position of watermark (bottom-right corner)
position = (w - 300, h - 30)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
color = (255, 255, 255)   # White color
thickness = 2

# ---------- Insert Watermark ----------
# Make a copy for watermarking
watermarked_image = img.copy()
cv2.putText(
    watermarked_image,
    watermark_text,
    position,
    font,
    font_scale,
    color,
    thickness,
    cv2.LINE_AA
)

# Convert watermarked image to RGB for Matplotlib
watermarked_rgb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)

# ---------- Create Matplotlib Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Image Watermarking', fontsize=16, fontweight='bold')

# Plot 1: Original Image
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[0].text(0.5, 0.02, f'Dimensions: {w} x {h}', 
             transform=axes[0].transAxes, 
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Watermarked Image
axes[1].imshow(watermarked_rgb)
axes[1].set_title('Watermarked Image', fontsize=12, fontweight='bold')
axes[1].axis('off')

# Add watermark position info
position_info = f"Watermark: '{watermark_text}'\n"
position_info += f"Position: ({position[0]}, {position[1]})\n"
position_info += f"Font scale: {font_scale}\n"
position_info += f"Thickness: {thickness}"

axes[1].text(0.5, 0.02, position_info, 
             transform=axes[1].transAxes, 
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Highlight the watermark area with a rectangle
# Create a rectangle patch
from matplotlib.patches import Rectangle
rect = Rectangle((position[0]-10, position[1]-25),  # (x, y)
                 200, 30,  # width, height
                 linewidth=2, edgecolor='red', facecolor='none', 
                 linestyle='--', alpha=0.7)
axes[1].add_patch(rect)

# Add arrow pointing to watermark
axes[1].annotate('Watermark', 
                xy=(position[0]+100, position[1]), 
                xytext=(position[0]+100, position[1]-100),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                ha='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---------- Alternative: Side-by-side comparison ----------
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Top row: Full images
axes2[0, 0].imshow(img_rgb)
axes2[0, 0].set_title('Original Image (Full)', fontsize=11)
axes2[0, 0].axis('off')

axes2[0, 1].imshow(watermarked_rgb)
axes2[0, 1].set_title(f'With Watermark: "{watermark_text}"', fontsize=11)
axes2[0, 1].axis('off')

# Bottom row: Zoomed-in area around watermark
zoom_factor = 3
x1, y1 = max(0, position[0] - 150), max(0, position[1] - 50)
x2, y2 = min(w, position[0] + 150), min(h, position[1] + 50)

# Original zoomed
axes2[1, 0].imshow(img_rgb[y1:y2, x1:x2])
axes2[1, 0].set_title('Original (Zoomed)', fontsize=11)
axes2[1, 0].axis('off')
axes2[1, 0].text(0.5, 0.02, f'Region: [{x1}:{x2}, {y1}:{y2}]', 
                 transform=axes2[1, 0].transAxes, 
                 ha='center', fontsize=9)

# Watermarked zoomed
axes2[1, 1].imshow(watermarked_rgb[y1:y2, x1:x2])
axes2[1, 1].set_title('Watermarked (Zoomed)', fontsize=11)
axes2[1, 1].axis('off')

# Add grid for better visibility
for ax in axes2[1, :]:
    ax.grid(True, alpha=0.3, linestyle='--')

plt.suptitle('Watermark Comparison: Full View vs Zoomed Area', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---------- Print watermark details ----------
print("=" * 60)
print("WATERMARKING DETAILS")
print("=" * 60)
print(f"Image dimensions: {w} x {h}")
print(f"Watermark text: '{watermark_text}'")
print(f"Position: {position}")
print(f"Font: Hershey Simplex")
print(f"Font scale: {font_scale}")
print(f"Color: RGB{color}")
print(f"Thickness: {thickness}")
print(f"Antialiasing: Enabled (LINE_AA)")
print("=" * 60)

# Show all figures
plt.show()

# ---------- Save the results ----------
fig.savefig('watermark_comparison.png', dpi=150, bbox_inches='tight')
fig2.savefig('watermark_zoomed.png', dpi=150, bbox_inches='tight')
print("\nResults saved as:")
print("1. 'watermark_comparison.png' - Side-by-side comparison")
print("2. 'watermark_zoomed.png' - Zoomed view comparison")
