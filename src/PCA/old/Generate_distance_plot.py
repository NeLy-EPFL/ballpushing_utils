import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance
import PCA.Config as Config

# --- MFA ---
from MFA_Stats_subsetcontrols import (
    scores_with_meta as mfa_scores,
    selected_dims as mfa_dims,
    results_df as mfa_results,
)

# --- PCA Static ---
from PCA_Static import pca_scores_with_meta as static_scores, selected_dims as static_dims, results_df as static_results

# --- PCA Temporal ---
from PCA.old.PCA_Temporal import (
    fpca_scores_with_meta as temporal_scores,
    selected_dims as temporal_dims,
    results_df as temporal_results,
)

control_nicknames_dict = Config.registries["control_nicknames_dict"]


def compute_fly_distance(row, df, selected_dims, control_nicknames_dict):
    split = row["Split"]
    control_nickname = control_nicknames_dict.get(split)
    if control_nickname is None:
        return np.nan
    control_data = df[df["Nickname"] == control_nickname][selected_dims]
    if control_data.empty:
        return np.nan
    control_centroid = control_data.mean(axis=0).values
    control_cov = np.cov(control_data.values, rowvar=False)
    inv_cov = np.linalg.pinv(control_cov)
    try:
        return distance.mahalanobis(row[selected_dims].values, control_centroid, inv_cov)
    except Exception:
        return np.nan


def plot_distance_boxplot(scores_with_meta, selected_dims, results_df, title, filename):
    scores_with_meta = scores_with_meta.copy()
    scores_with_meta["distance_to_control"] = scores_with_meta.apply(
        lambda row: compute_fly_distance(row, scores_with_meta, selected_dims, control_nicknames_dict), axis=1
    )
    plot_data = scores_with_meta.dropna(subset=["distance_to_control", "Brain region"])
    median_order = plot_data.groupby("Nickname")["distance_to_control"].median().sort_values().index

    plt.figure(figsize=(10, max(8, len(median_order) * 0.5)))
    ax = plt.subplot(111)
    sns.boxplot(
        data=plot_data,
        y="Nickname",
        x="distance_to_control",
        hue="Brain region",
        order=median_order,
        palette=getattr(Config, "color_dict", "Set2"),
        showfliers=False,
        dodge=False,
        width=0.4,
        ax=ax,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
    )
    sns.stripplot(
        data=plot_data,
        y="Nickname",
        x="distance_to_control",
        hue="Brain region",
        order=median_order,
        palette=getattr(Config, "color_dict", "Set2"),
        dodge=False,
        alpha=0.6,
        size=3,
        ax=ax,
        jitter=True,
        linewidth=0.5,
        edgecolor="gray",
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Brain region", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xlabel("Mahalanobis distance to control centroid")
    plt.ylabel("Nickname")
    plt.title(title)
    plt.tight_layout()
    try:
        for i, nickname in enumerate(median_order):
            if nickname in results_df[results_df["MannWhitney_any_dim_significant"]]["Nickname"].values:
                ax.get_yticklabels()[i].set_color("red")
    except Exception:
        pass
    plt.savefig(filename + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")
    plt.savefig(filename + ".eps", bbox_inches="tight")
    plt.show()


# --- Generate plots for each analysis ---
plot_distance_boxplot(
    mfa_scores,
    mfa_dims,
    mfa_results,
    "Deviation from control per Nickname (MFA, colored by Brain region)",
    "distance_to_control_horizontal_boxplot_MFA",
)
plot_distance_boxplot(
    static_scores,
    static_dims,
    static_results,
    "Deviation from control per Nickname (PCA Static, colored by Brain region)",
    "distance_to_control_horizontal_boxplot_PCA_Static",
)
plot_distance_boxplot(
    temporal_scores,
    temporal_dims,
    temporal_results,
    "Deviation from control per Nickname (PCA Temporal, colored by Brain region)",
    "distance_to_control_horizontal_boxplot_PCA_Temporal",
)
