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

# ---------- Convert BGR to RGB for Matplotlib ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Boundary Detection Kernel ----------
boundary_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Apply convolution
boundary = cv2.filter2D(gray, cv2.CV_64F, boundary_kernel)

# Convert to displayable format
boundary_abs = cv2.convertScaleAbs(boundary)

# ---------- Alternative edge detection methods for comparison ----------
# Sobel edge detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_abs = cv2.convertScaleAbs(sobel_combined)

# Canny edge detection
canny_edges = cv2.Canny(gray, 100, 200)

# Laplacian edge detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# ---------- Create Matplotlib Figure ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Boundary/Edge Detection Methods Comparison', fontsize=16, fontweight='bold')

# Plot 1: Original Color Image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('1. Original Color Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
axes[0, 0].text(0.5, 0.02, f'Size: {gray.shape[1]}x{gray.shape[0]}', 
                transform=axes[0, 0].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Grayscale Image
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. Grayscale Image', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Plot 3: Boundary Kernel Visualization
ax_kernel = axes[0, 2]
im_kernel = ax_kernel.imshow(boundary_kernel, cmap='coolwarm', interpolation='nearest')
ax_kernel.set_title('3. Boundary Detection Kernel', fontsize=12, fontweight='bold')
ax_kernel.set_xticks([0, 1, 2])
ax_kernel.set_yticks([0, 1, 2])
ax_kernel.set_xticklabels(['-1', '0', '1'])
ax_kernel.set_yticklabels(['-1', '0', '1'])

# Add kernel values
for i in range(3):
    for j in range(3):
        value = boundary_kernel[i, j]
        color = 'white' if abs(value) >= 7 else 'black'
        ax_kernel.text(j, i, f'{value:.0f}', ha='center', va='center', 
                      color=color, fontsize=12, fontweight='bold')

# Plot 4: Custom Boundary Detection (Your method)
axes[1, 0].imshow(boundary_abs, cmap='gray')
axes[1, 0].set_title('4. Custom Boundary Detection\n8-neighbor kernel', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Plot 5: Sobel Edge Detection
axes[1, 1].imshow(sobel_abs, cmap='gray')
axes[1, 1].set_title('5. Sobel Edge Detection\n(Gradient-based)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Plot 6: Canny Edge Detection
axes[1, 2].imshow(canny_edges, cmap='gray')
axes[1, 2].set_title('6. Canny Edge Detection\n(Multi-stage algorithm)', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---------- Figure 2: Detailed Comparison ----------
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# Original grayscale
axes2[0, 0].imshow(gray, cmap='gray')
axes2[0, 0].set_title('Original Grayscale', fontsize=11)
axes2[0, 0].axis('off')

# Your boundary detection
axes2[0, 1].imshow(boundary_abs, cmap='gray')
axes2[0, 1].set_title('Custom Boundary Detection', fontsize=11)
axes2[0, 1].axis('off')

# Laplacian for comparison
axes2[0, 2].imshow(laplacian_abs, cmap='gray')
axes2[0, 2].set_title('Laplacian Edge Detection', fontsize=11)
axes2[0, 2].axis('off')

# Zoomed region for detailed comparison
zoom_x, zoom_y = gray.shape[1]//3, gray.shape[0]//3
zoom_size = 100

# Original zoomed
axes2[1, 0].imshow(gray[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size], cmap='gray')
axes2[1, 0].set_title('Original (Zoomed)', fontsize=11)
axes2[1, 0].axis('off')
axes2[1, 0].grid(True, alpha=0.3)

# Boundary zoomed
axes2[1, 1].imshow(boundary_abs[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size], cmap='gray')
axes2[1, 1].set_title('Boundary (Zoomed)', fontsize=11)
axes2[1, 1].axis('off')
axes2[1, 1].grid(True, alpha=0.3)

# Edge detection comparison
edge_comparison = np.zeros_like(boundary_abs)
edge_comparison = cv2.normalize(boundary_abs + sobel_abs, None, 0, 255, cv2.NORM_MINMAX)
axes2[1, 2].imshow(edge_comparison[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size], cmap='hot')
axes2[1, 2].set_title('Edge Overlay (Hot Colormap)', fontsize=11)
axes2[1, 2].axis('off')

plt.suptitle('Boundary Detection: Detailed Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# ---------- Figure 3: Intensity Profile Analysis ----------
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Select a horizontal line from the middle
row_idx = gray.shape[0] // 2
original_line = gray[row_idx, :]
boundary_line = boundary_abs[row_idx, :]

# Plot intensity profiles
x = np.arange(len(original_line))
ax1.plot(x, original_line, 'b-', linewidth=1, label='Original Grayscale', alpha=0.8)
ax1.plot(x, boundary_line, 'r-', linewidth=1, label='Boundary Detection', alpha=0.8)
ax1.set_title('Intensity Profile Comparison', fontsize=12, fontweight='bold')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Intensity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight edge regions
edges_indices = np.where(boundary_line > np.mean(boundary_line) * 1.5)[0]
for idx in edges_indices:
    if idx % 50 == 0:  # Show only some markers to avoid clutter
        ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.5, linewidth=0.5)

# Histogram comparison
ax2.hist(gray.ravel(), bins=256, range=(0, 256), alpha=0.7, color='blue', 
         label='Original', density=True)
ax2.hist(boundary_abs.ravel(), bins=256, range=(0, 256), alpha=0.7, color='red', 
         label='Boundary', density=True)
ax2.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
ax2.set_xlabel('Pixel Intensity')
ax2.set_ylabel('Normalized Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# ---------- Print Analysis Results ----------
print("=" * 70)
print("BOUNDARY DETECTION ANALYSIS")
print("=" * 70)
print(f"Image dimensions: {gray.shape}")
print(f"\nKernel used:")
print("[[-1, -1, -1],")
print(" [-1,  8, -1],")
print(" [-1, -1, -1]]")
print(f"\nKernel properties:")
print(f"  Sum of coefficients: {np.sum(boundary_kernel):.0f}")
print(f"  Center coefficient: {boundary_kernel[1, 1]:.0f}")
print(f"  Edge coefficients: {boundary_kernel[0, 1]:.0f}")
print(f"\nDetection results:")
print(f"  Min boundary value: {boundary_abs.min()}")
print(f"  Max boundary value: {boundary_abs.max()}")
print(f"  Mean boundary value: {boundary_abs.mean():.2f}")
print(f"  Std boundary value: {boundary_abs.std():.2f}")
print(f"\nEdge detection comparison:")
print(f"  Sobel mean: {sobel_abs.mean():.2f}")
print(f"  Canny mean: {canny_edges.mean():.2f}")
print("=" * 70)

# Show all figures
plt.show()

# ---------- Save the results ----------
fig.savefig('boundary_detection_comparison.png', dpi=150, bbox_inches='tight')
fig2.savefig('boundary_detailed_analysis.png', dpi=150, bbox_inches='tight')
fig3.savefig('boundary_intensity_analysis.png', dpi=150, bbox_inches='tight')

print("\nResults saved as:")
print("1. 'boundary_detection_comparison.png' - Method comparison")
print("2. 'boundary_detailed_analysis.png' - Detailed analysis")
print("3. 'boundary_intensity_analysis.png' - Intensity profiles")
