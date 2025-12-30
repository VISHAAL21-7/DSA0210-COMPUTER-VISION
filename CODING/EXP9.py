import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# ---------- Load video ----------
video_path = r"E:/VK CUTZ/Clap 2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Read first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read frame")
    exit()

h, w = frame.shape[:2]

# ---------- Convert first frame to RGB for Matplotlib ----------
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------- Define source points ----------
pts1 = np.float32([
    [50, 50],        # top-left
    [w - 50, 50],    # top-right
    [50, h - 50],    # bottom-left
    [w - 50, h - 50] # bottom-right
])

# ---------- Define destination points ----------
pts2 = np.float32([
    [100, 100],
    [w - 150, 80],
    [150, h - 100],
    [w - 100, h - 50]
])

# Perspective matrix
perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply transformation to first frame
transformed = cv2.warpPerspective(frame, perspective_matrix, (w, h))
transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

# ---------- Create Matplotlib Figure ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Perspective Transformation Analysis', fontsize=16, fontweight='bold')

# Plot 1: Original Frame with Source Points
axes[0, 0].imshow(frame_rgb)
axes[0, 0].set_title('1. Original Frame with Source Points', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Draw source points and polygon
for i, (x, y) in enumerate(pts1):
    axes[0, 0].plot(x, y, 'ro', markersize=8, markeredgewidth=2, markeredgecolor='white')
    axes[0, 0].text(x, y-15, f'P{i+1}', color='white', fontsize=10, 
                   fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Draw polygon connecting source points
source_poly = Polygon(pts1, closed=True, fill=False, 
                     edgecolor='yellow', linewidth=2, linestyle='--', alpha=0.8)
axes[0, 0].add_patch(source_poly)

# Plot 2: Transformed Frame with Destination Points
axes[0, 1].imshow(transformed_rgb)
axes[0, 1].set_title('2. Transformed Frame with Destination Points', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Draw destination points and polygon
for i, (x, y) in enumerate(pts2):
    axes[0, 1].plot(x, y, 'go', markersize=8, markeredgewidth=2, markeredgecolor='white')
    axes[0, 1].text(x, y-15, f'P{i+1}\'', color='white', fontsize=10, 
                   fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

# Draw polygon connecting destination points
dest_poly = Polygon(pts2, closed=True, fill=False, 
                   edgecolor='cyan', linewidth=2, linestyle='--', alpha=0.8)
axes[0, 1].add_patch(dest_poly)

# Plot 3: Transformation Matrix
ax_matrix = axes[0, 2]
ax_matrix.axis('off')

matrix_info = "Perspective Transformation Matrix:\n\n"
matrix_info += f"[[{perspective_matrix[0,0]:.3f}, {perspective_matrix[0,1]:.3f}, {perspective_matrix[0,2]:.3f}],\n"
matrix_info += f" [{perspective_matrix[1,0]:.3f}, {perspective_matrix[1,1]:.3f}, {perspective_matrix[1,2]:.3f}],\n"
matrix_info += f" [{perspective_matrix[2,0]:.3f}, {perspective_matrix[2,1]:.3f}, {perspective_matrix[2,2]:.3f}]]\n\n"
matrix_info += f"Image Size: {w} × {h}\n\n"
matrix_info += f"Transformation Type:\n3×3 Homography Matrix"

ax_matrix.text(0.1, 0.5, matrix_info, transform=ax_matrix.transAxes,
               fontsize=10, fontfamily='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot 4: Side-by-Side Comparison
axes[1, 0].imshow(frame_rgb)
axes[1, 0].set_title('Original', fontsize=11)
axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.02, f'Corners: {[(int(p[0]), int(p[1])) for p in pts1]}', 
                transform=axes[1, 0].transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[1, 1].imshow(transformed_rgb)
axes[1, 1].set_title('Transformed', fontsize=11)
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.02, f'Corners: {[(int(p[0]), int(p[1])) for p in pts2]}', 
                transform=axes[1, 1].transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 5: Grid Visualization
# Create a grid on original image
grid_size = 10
grid_original = frame_rgb.copy()
grid_transformed = transformed_rgb.copy()

# Draw grid on original
for i in range(0, w, w//grid_size):
    cv2.line(grid_original, (i, 0), (i, h), (255, 0, 0), 1)
for j in range(0, h, h//grid_size):
    cv2.line(grid_original, (0, j), (w, j), (255, 0, 0), 1)

# Draw grid on transformed
for i in range(0, w, w//grid_size):
    cv2.line(grid_transformed, (i, 0), (i, h), (0, 255, 0), 1)
for j in range(0, h, h//grid_size):
    cv2.line(grid_transformed, (0, j), (w, j), (0, 255, 0), 1)

axes[1, 2].imshow(grid_original)
axes[1, 2].set_title('Grid Visualization (Blue = Original)', fontsize=11)
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---------- Figure 2: Point Correspondence Visualization ----------
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Point correspondence plot
ax1.imshow(frame_rgb)
for i, (x, y) in enumerate(pts1):
    ax1.plot(x, y, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
    ax1.text(x, y-20, f'({int(x)}, {int(y)})', color='white', fontsize=9,
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Draw arrows showing transformation
colors = ['yellow', 'cyan', 'magenta', 'orange']
for i, ((x1, y1), (x2, y2), color) in enumerate(zip(pts1, pts2, colors)):
    # Draw arrow from source to destination (scaled for visualization)
    ax1.arrow(x1, y1, (x2 - x1)*0.8, (y2 - y1)*0.8, 
             head_width=15, head_length=20, fc=color, ec=color, alpha=0.7)
    ax1.text((x1 + x2)/2, (y1 + y2)/2, f'→ P{i+1}\'', color=color, fontsize=9,
            fontweight='bold', ha='center')

ax1.set_title('Point Correspondences with Transformation Vectors', fontsize=12, fontweight='bold')
ax1.axis('off')

# Warp visualization
ax2.imshow(transformed_rgb)
for i, (x, y) in enumerate(pts2):
    ax2.plot(x, y, 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
    ax2.text(x, y-20, f'({int(x)}, {int(y)})', color='white', fontsize=9,
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

ax2.set_title('Transformed Points', fontsize=12, fontweight='bold')
ax2.axis('off')

plt.suptitle('Perspective Transformation: Point Mapping', fontsize=14, fontweight='bold')
plt.tight_layout()

# ---------- Figure 3: Grid Warping Visualization ----------
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Create a more detailed grid
grid_detailed = np.zeros((h, w, 3), dtype=np.uint8)
grid_detailed[:] = frame_rgb

# Draw dense grid
grid_spacing = 50
for i in range(0, w, grid_spacing):
    cv2.line(grid_detailed, (i, 0), (i, h), (0, 255, 255), 2)  # Cyan vertical lines
for j in range(0, h, grid_spacing):
    cv2.line(grid_detailed, (0, j), (w, j), (0, 255, 255), 2)  # Cyan horizontal lines

# Warp the grid
grid_warped = cv2.warpPerspective(grid_detailed, perspective_matrix, (w, h))

# Original grid
axes3[0].imshow(grid_detailed)
axes3[0].set_title('Original Grid', fontsize=11)
axes3[0].axis('off')

# Warped grid
axes3[1].imshow(grid_warped)
axes3[1].set_title('Warped Grid', fontsize=11)
axes3[1].axis('off')

# Difference
difference = cv2.absdiff(grid_detailed, grid_warped)
axes3[2].imshow(difference)
axes3[2].set_title('Warping Difference', fontsize=11)
axes3[2].axis('off')

plt.suptitle('Grid Warping Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()

# ---------- Reset video for processing ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------- Process video for animation ----------
print("=" * 70)
print("PERSPECTIVE TRANSFORMATION ANALYSIS")
print("=" * 70)
print(f"Video dimensions: {w} × {h}")
print(f"\nSource points (original corners):")
for i, (x, y) in enumerate(pts1):
    print(f"  P{i+1}: ({int(x)}, {int(y)})")
print(f"\nDestination points (transformed corners):")
for i, (x, y) in enumerate(pts2):
    print(f"  P{i+1}': ({int(x)}, {int(y)})")
print(f"\nTransformation matrix computed successfully")
print("=" * 70)

# ---------- Create animation function ----------
def update_frame(frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return
    
    # Apply transformation
    transformed = cv2.warpPerspective(frame, perspective_matrix, (w, h))
    transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    
    # Update plots
    ax_orig.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax_trans.imshow(transformed_rgb)
    
    ax_orig.set_title(f'Original Frame {frame_num}', fontsize=10)
    ax_trans.set_title(f'Transformed Frame {frame_num}', fontsize=10)
    
    plt.draw()

# ---------- Create animation figure ----------
fig_anim, (ax_orig, ax_trans) = plt.subplots(1, 2, figsize=(12, 6))
fig_anim.suptitle('Perspective Transformation Animation', fontsize=14, fontweight='bold')

# Get total frames for animation
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nTotal frames in video: {total_frames}")
print("Close the animation window to continue...")

# Reset to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Create animation
ani = FuncAnimation(fig_anim, update_frame, frames=min(50, total_frames), 
                   interval=100, repeat=True)

plt.tight_layout()

# Show all figures
plt.show()

# Release video capture
cap.release()

# ---------- Save the results ----------
fig.savefig('perspective_transform_analysis.png', dpi=150, bbox_inches='tight')
fig2.savefig('perspective_point_correspondence.png', dpi=150, bbox_inches='tight')
fig3.savefig('perspective_grid_warping.png', dpi=150, bbox_inches='tight')

print("\nResults saved as:")
print("1. 'perspective_transform_analysis.png' - Transformation analysis")
print("2. 'perspective_point_correspondence.png' - Point mapping")
print("3. 'perspective_grid_warping.png' - Grid warping")
