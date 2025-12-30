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

# ---------- Convert to Grayscale ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Step 1: Blur the image ----------
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

# ---------- Step 2: Create Unsharp Mask ----------
# FIXED: Use float32 to preserve negative values
gray_float = gray.astype(np.float32)
blurred_float = blurred.astype(np.float32)
unsharp_mask_float = gray_float - blurred_float

# For display, normalize mask to 0-255 range
unsharp_mask_display = cv2.normalize(unsharp_mask_float, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# ---------- Step 3: Add mask to original ----------
k = 1.0  # sharpening strength
sharpened_float = gray_float + k * unsharp_mask_float
sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)

# ---------- Alternative method using addWeighted ----------
sharpened_weighted = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

# ---------- Create Matplotlib figure with all images ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Image Sharpening using Unsharp Masking', fontsize=16, fontweight='bold')

# Convert BGR to RGB for original color image (if you want to show it)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot 1: Original Color Image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('1. Original Color Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Plot 2: Grayscale Image
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. Grayscale Image', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Plot 3: Blurred Image
axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. Blurred Image (Gaussian)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Plot 4: Unsharp Mask (enhanced for visualization)
axes[1, 0].imshow(unsharp_mask_display, cmap='gray')
axes[1, 0].set_title('4. Unsharp Mask\n(original - blurred)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Plot 5: Sharpened Image (Manual)
axes[1, 1].imshow(sharpened, cmap='gray')
axes[1, 1].set_title('5. Sharpened Image (Manual)\ngray + (gray - blurred)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Plot 6: Sharpened Image (addWeighted)
axes[1, 2].imshow(sharpened_weighted, cmap='gray')
axes[1, 2].set_title('6. Sharpened Image (addWeighted)\n1.5*gray - 0.5*blurred', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

# Add some spacing between subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Optional: Add a colorbar for the unsharp mask
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(unsharp_mask_float, cmap='seismic')  # seismic colormap shows positive and negative
ax2.set_title('Unsharp Mask with Colorbar\n(Red = Positive, Blue = Negative)', fontsize=12, fontweight='bold')
ax2.axis('off')
fig2.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()

# ---------- Alternative: Single row layout ----------
fig3, axes3 = plt.subplots(1, 6, figsize=(18, 4))
titles = ['Original', 'Grayscale', 'Blurred', 'Unsharp Mask', 'Sharpened\n(Manual)', 'Sharpened\n(addWeighted)']
images = [img_rgb, gray, blurred, unsharp_mask_display, sharpened, sharpened_weighted]

for i, (ax, img_data, title) in enumerate(zip(axes3, images, titles)):
    if i == 0:  # Original color image
        ax.imshow(img_data)
    else:  # Grayscale images
        ax.imshow(img_data, cmap='gray')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    # Add label
    ax.text(0.5, -0.1, f'({i+1})', transform=ax.transAxes, 
            ha='center', va='top', fontsize=9)

plt.suptitle('Image Processing Pipeline', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()

# ---------- Display histogram comparison ----------
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of original vs sharpened
ax1.hist(gray.ravel(), bins=256, range=(0, 256), alpha=0.7, color='blue', label='Original')
ax1.hist(sharpened.ravel(), bins=256, range=(0, 256), alpha=0.7, color='red', label='Sharpened')
ax1.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
ax1.set_xlabel('Pixel Intensity')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histogram of unsharp mask
ax2.hist(unsharp_mask_float.ravel(), bins=100, alpha=0.7, color='green')
ax2.set_title('Unsharp Mask Histogram', fontsize=12, fontweight='bold')
ax2.set_xlabel('Pixel Difference Value')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero line')
ax2.legend()

plt.tight_layout()

print("=" * 60)
print("IMAGE PROCESSING COMPLETE")
print("=" * 60)
print(f"Original image size: {gray.shape}")
print(f"Unsharp mask range: {unsharp_mask_float.min():.2f} to {unsharp_mask_float.max():.2f}")
print(f"Sharpening completed successfully!")

# Show all figures
plt.show()

# ---------- Save the main figure ----------
fig.savefig('sharpening_results.png', dpi=150, bbox_inches='tight')
print("Results saved as 'sharpening_results.png'")
