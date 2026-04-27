#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# 1. Fake dataset
np.random.seed(0)
data = np.random.randn(8, 5)
row_labels = [f"G{i}" for i in range(8)]
col_labels = [f"M{j}" for j in range(5)]

# 2. Clustering
row_dist = pdist(data, metric="euclidean")
row_Z = linkage(row_dist, method="ward")
dg = dendrogram(row_Z, orientation="left", no_labels=True, color_threshold=None, no_plot=True)
row_order = dg["leaves"]
ordered_data = data[row_order, :]

# -----------------------------
# FIXED: Compute TRUE row centers from GridSpec geometry
# -----------------------------
fig = plt.figure(figsize=(6, 4))
outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 3.0], wspace=0.05)
split_idx = 3  # after row 2
heatmap_gs = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[split_idx, len(row_order) - split_idx], hspace=0.7
)

ax_dendro = fig.add_subplot(outer_gs[0, 0])
ax_heatmap_top = fig.add_subplot(heatmap_gs[0, 0])
ax_heatmap_bottom = fig.add_subplot(heatmap_gs[1, 0])


# Plot heatmaps using clustered order for both data and labels
top_indices = list(range(split_idx))
bottom_indices = list(range(split_idx, len(row_order)))

top_data = ordered_data[top_indices, :]
bottom_data = ordered_data[bottom_indices, :]
top_labels = [row_labels[row_order[i]] for i in top_indices]
bottom_labels = [row_labels[row_order[i]] for i in bottom_indices]

ax_heatmap_top.imshow(top_data, aspect="auto", cmap="RdBu_r")
ax_heatmap_top.set_yticks(range(len(top_labels)))
ax_heatmap_top.set_yticklabels(top_labels)
ax_heatmap_top.set_xticks([])
ax_heatmap_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

ax_heatmap_bottom.imshow(bottom_data, aspect="auto", cmap="RdBu_r")
ax_heatmap_bottom.set_yticks(range(len(bottom_labels)))
ax_heatmap_bottom.set_yticklabels(bottom_labels)
ax_heatmap_bottom.set_xticks(range(len(col_labels)))
ax_heatmap_bottom.set_xticklabels(col_labels, rotation=45, ha="right")

# Clean spines
for ax in [ax_heatmap_top, ax_heatmap_bottom]:
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)

ax_heatmap_top.set_ylim(len(top_labels) - 0.5, -0.5)
ax_heatmap_bottom.set_ylim(len(bottom_labels) - 0.5, -0.5)

# -----------------------------
# COMPUTE ACTUAL ROW CENTERS FROM AXIS POSITIONS
# -----------------------------
fig_height = fig.get_figheight()
heatmap_bbox = outer_gs[0, 1].get_position(fig)

row_pos_centers = []
# TOP BLOCK (y decreasing from top)
for i in range(split_idx):
    tick_pos_data = len(top_labels) - 0.5 - i  # 2.5,1.5,0.5 for 3 rows
    tick_pos_fig = ax_heatmap_top.transData.transform((0, tick_pos_data))[1]
    tick_pos_normalized = heatmap_bbox.y0 + (tick_pos_fig / fig_height)  # 0-1 figure frac
    row_pos_centers.append(tick_pos_normalized)

# BOTTOM BLOCK (y increasing from bottom)
for i in range(len(bottom_labels)):
    tick_pos_data = i + 0.5  # 0.5,1.5,...
    tick_pos_fig = ax_heatmap_bottom.transData.transform((0, tick_pos_data))[1]
    tick_pos_normalized = heatmap_bbox.y0 + (tick_pos_fig / fig_height)
    row_pos_centers.append(tick_pos_normalized)


print("Row centers:", [f"{p:.3f}" for p in row_pos_centers])
print("Heatmap bbox y:", heatmap_bbox.y0, "to", heatmap_bbox.y1)

# -----------------------------
# Map EACH dendrogram leaf to its PHYSICAL row_pos_centers index
# -----------------------------
n_leaves = len(dg["leaves"])
leaf_to_row_center_idx = []
for leaf_idx in dg["leaves"]:  # leaf_idx is original row index
    row_center_idx = row_order.index(leaf_idx)  # position in dendrogram order
    leaf_to_row_center_idx.append(row_center_idx)


# FIXED remap_y using leaf_to_row_center_idx
def remap_y(y):
    dendro_idx = np.clip((y - 5.0) / 10.0, 0, n_leaves - 1)
    physical_idx = int(np.floor(dendro_idx))
    physical_idx = min(physical_idx, len(row_pos_centers) - 1)
    low_physical = leaf_to_row_center_idx[physical_idx]
    if dendro_idx >= n_leaves - 1:
        return row_pos_centers[-1]
    high_physical_idx = leaf_to_row_center_idx[min(physical_idx + 1, n_leaves - 1)]
    alpha = dendro_idx - physical_idx
    low_center = row_pos_centers[low_physical]
    high_center = row_pos_centers[high_physical_idx]
    return (1 - alpha) * low_center + alpha * high_center


new_icoord = [[remap_y(yy) for yy in seg] for seg in dg["icoord"]]

# Debug prints to verify
print("Expected mapping:")
for leaf_idx, physical_idx in enumerate(leaf_to_row_center_idx[:5]):
    print(
        f"Dendro leaf {leaf_idx} (orig row {dg['leaves'][leaf_idx]}) -> physical {physical_idx} = {row_labels[row_order[physical_idx]]}"
    )

# Draw dendrogram
ax_dendro.clear()
for y_seg, x_seg, c in zip(new_icoord, dg["dcoord"], dg["color_list"]):
    ax_dendro.plot(x_seg, y_seg, color=c if c else "C0")

# Set dendro ylim to span entire heatmap height (including hspace gap)
heatmap_y_min = min(row_pos_centers) - 2
heatmap_y_max = max(row_pos_centers) + 2
ax_dendro.set_ylim(heatmap_y_min, heatmap_y_max)
ax_dendro.invert_xaxis()
ax_dendro.set_xticks([])
ax_dendro.set_yticks([])

# plt.tight_layout()
plt.show()
