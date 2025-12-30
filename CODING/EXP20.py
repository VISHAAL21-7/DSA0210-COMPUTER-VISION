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

# ---------- Convert BGR to RGB for display ----------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Create a figure with multiple A values for comparison ----------
A_values = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
results = []

for A in A_values:
    # Create High-Boost kernel
    high_boost_kernel = np.array([
        [0, -1,  0],
        [-1, A + 4, -1],
        [0, -1,  0]
    ], dtype=np.float32)
    
    # Apply High-Boost filtering
    high_boost = cv2.filter2D(gray.astype(np.float32), -1, high_boost_kernel)
    
    # Clip and convert to uint8
    high_boost_clipped = np.clip(high_boost, 0, 255).astype(np.uint8)
    results.append((A, high_boost_kernel, high_boost_clipped))

# ---------- Display Results with Matplotlib ----------
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.suptitle('High-Boost Filtering for Image Sharpening', fontsize=16, fontweight='bold')

# Plot 1: Original Color Image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('1. Original Color Image', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Plot 2: Grayscale Image
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. Grayscale Image', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Plot 3: High-Boost Kernel Visualization (for A=2)
A = 2.0
kernel_example = np.array([
    [0, -1,  0],
    [-1, A + 4, -1],
    [0, -1,  0]
])
ax_kernel = axes[0, 2]
im = ax_kernel.imshow(kernel_example, cmap='coolwarm', interpolation='nearest')
ax_kernel.set_title(f'3. High-Boost Kernel\n(A = {A})', fontsize=11, fontweight='bold')
ax_kernel.set_xticks([0, 1, 2])
ax_kernel.set_yticks([0, 1, 2])
ax_kernel.set_xticklabels(['-1', '0', '1'])
ax_kernel.set_yticklabels(['-1', '0', '1'])

# Add kernel values as text
for i in range(3):
    for j in range(3):
        value = kernel_example[i, j]
        color = 'white' if abs(value) > 3 else 'black'
        ax_kernel.text(j, i, f'{value:.1f}', ha='center', va='center', 
                      color=color, fontsize=12, fontweight='bold')

# Plot 4: Empty or theory explanation
ax_theory = axes[0, 3]
ax_theory.axis('off')
theory_text = "High-Boost Filter:\n\n"
theory_text += "Kernel = Laplacian + (A-1)*Identity\n\n"
theory_text += "Result = A*Original - Blurred\n\n"
theory_text += "A = 1: Standard Laplacian\n"
theory_text += "A > 1: High-Boost (Enhanced)"
ax_theory.text(0.1, 0.5, theory_text, transform=ax_theory.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot results for different A values
for idx, (A, kernel, result) in enumerate(results[:4]):
    row = 1
    col = idx
    ax = axes[row, col]
    ax.imshow(result, cmap='gray')
    ax.set_title(f'A = {A}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Calculate and display metrics
    diff = cv2.absdiff(gray, result)
    avg_diff = np.mean(diff)
    ax.text(0.5, -0.1, f'Avg diff: {avg_diff:.1f}', 
            transform=ax.transAxes, ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---------- Figure 2: Compare Original vs Best Result ----------
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax1.imshow(gray, cmap='gray')
ax1.set_title('Original Grayscale', fontsize=12, fontweight='bold')
ax1.axis('off')

# High-Boost with A=2 (typically good default)
A_best = 2.0
best_kernel = np.array([
    [0, -1,  0],
    [-1, A_best + 4, -1],
    [0, -1,  0]
], dtype=np.float32)
best_result = cv2.filter2D(gray.astype(np.float32), -1, best_kernel)
best_result = np.clip(best_result, 0, 255).astype(np.uint8)

ax2.imshow(best_result, cmap='gray')
ax2.set_title(f'High-Boost Sharpened (A={A_best})', fontsize=12, fontweight='bold')
ax2.axis('off')

# Difference image
difference = cv2.absdiff(gray, best_result)
ax3.imshow(difference, cmap='hot')
ax3.set_title('Difference (Enhanced Areas)', fontsize=12, fontweight='bold')
ax3.axis('off')

plt.suptitle('High-Boost Filtering: Before and After', fontsize=14, fontweight='bold')
plt.tight_layout()

# ---------- Figure 3: Intensity Profile Comparison ----------
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Select a horizontal line from the middle of the image
row_idx = gray.shape[0] // 2
original_line = gray[row_idx, :]
boosted_line = best_result[row_idx, :]

# Plot intensity profiles
x = np.arange(len(original_line))
ax1.plot(x, original_line, 'b-', linewidth=1.5, label='Original', alpha=0.7)
ax1.plot(x, boosted_line, 'r-', linewidth=1.5, label=f'High-Boost (A={A_best})', alpha=0.7)
ax1.set_title('Intensity Profile Comparison', fontsize=12, fontweight='bold')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Intensity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot difference
ax2.plot(x, boosted_line - original_line, 'g-', linewidth=1.5)
ax2.set_title('Intensity Difference (Boosted - Original)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Pixel Position')
ax2.set_ylabel('Intensity Difference')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(x, 0, boosted_line - original_line, 
                 where=(boosted_line - original_line) > 0, 
                 color='green', alpha=0.3, label='Increased intensity')
ax2.fill_between(x, 0, boosted_line - original_line, 
                 where=(boosted_line - original_line) < 0, 
                 color='red', alpha=0.3, label='Decreased intensity')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# ---------- Figure 4: Kernel Values Visualization ----------
fig4, axes = plt.subplots(2, 3, figsize=(12, 8))

for idx, (A, kernel, result) in enumerate(results):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Display kernel as matrix
    ax.imshow(kernel, cmap='coolwarm', vmin=-1, vmax=A+3)
    ax.set_title(f'A = {A}', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add values
    for i in range(3):
        for j in range(3):
            value = kernel[i, j]
            color = 'white' if abs(value) > (A+3)/2 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')

axes[1, 2].axis('off')  # Remove empty subplot if less than 6
plt.suptitle('High-Boost Kernel Variations with Different A Values', fontsize=14, fontweight='bold')
plt.tight_layout()

# ---------- Print summary information ----------
print("=" * 70)
print("HIGH-BOOST FILTERING SUMMARY")
print("=" * 70)
print(f"Original image size: {gray.shape}")
print(f"Image dtype: {gray.dtype}")
print("\nKernel formula: [0  -1  0; -1  A+4  -1; 0  -1  0]")
print("\nA value effects:")
print("  A = 1.0: Standard Laplacian sharpening")
print("  A = 2.0: Moderate sharpening (recommended)")
print("  A > 2.0: Stronger sharpening (may amplify noise)")
print("\nProcessing completed successfully!")
print("=" * 70)

# Show all figures
plt.show()

# ---------- Save the main figure ----------
fig.savefig('high_boost_results.png', dpi=150, bbox_inches='tight')
print("\nResults saved as 'high_boost_results.png'")
