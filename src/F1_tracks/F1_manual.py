import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the CSV data
df = pd.read_csv("/home/durrieu/ballpushing_utils/src/F1_tracks/250924_F1_Analysis - Feuille 1.csv")

# Clean and prepare the data
print("Original data shape:", df.shape)

# Remove rows with critical NaN values and clean up the data
df_clean = df.copy()

# Handle the 'Has_finished' column - remove rows where it's NaN
df_clean = df_clean.dropna(subset=["Has_finished"])


# Create the three conditions based on Pretraining and Unlocked
def create_condition(row):
    if row["Pretraining"] == "n":
        return "No Pretraining"
    elif row["Pretraining"] == "y" and row["Unlocked"] == "y":
        return "Pretraining Unlocked"
    elif row["Pretraining"] == "y" and row["Unlocked"] == "n":
        return "Pretraining Not Unlocked"
    else:
        return "Other"  # for rows with missing Unlocked data


df_clean["Condition"] = df_clean.apply(create_condition, axis=1)

# Remove 'Other' category if it exists
df_clean = df_clean[df_clean["Condition"] != "Other"]

print("Cleaned data shape:", df_clean.shape)
print("\nCondition counts:")
print(df_clean["Condition"].value_counts())

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("F1 Ball Pushing Analysis: Pretraining Effects", fontsize=16, fontweight="bold")

# 1. Has_finished as a function of Pretraining (simple binary comparison)
ax1 = axes[0, 0]
pretraining_finished = df_clean.groupby("Pretraining")["Has_finished"].agg(["mean", "count", "std"]).reset_index()
pretraining_finished["sem"] = pretraining_finished["std"] / np.sqrt(pretraining_finished["count"])

sns.barplot(data=df_clean, x="Pretraining", y="Has_finished", ax=ax1, palette=["#FF6B6B", "#4ECDC4"], alpha=0.7)
ax1.set_title("Success Rate by Pretraining Status")
ax1.set_xlabel("Pretraining")
ax1.set_ylabel("Proportion Finished")
ax1.set_ylim(0, 1)

# Add individual data points
sns.stripplot(data=df_clean, x="Pretraining", y="Has_finished", ax=ax1, color="black", size=4, alpha=0.6)

# 2. Has_finished for the three conditions
ax2 = axes[0, 1]
condition_finished = df_clean.groupby("Condition")["Has_finished"].agg(["mean", "count", "std"]).reset_index()

sns.barplot(
    data=df_clean, x="Condition", y="Has_finished", ax=ax2, palette=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.7
)
ax2.set_title("Success Rate by Detailed Condition")
ax2.set_xlabel("Condition")
ax2.set_ylabel("Proportion Finished")
ax2.set_ylim(0, 1)
ax2.tick_params(axis="x", rotation=45)

# Add individual data points
sns.stripplot(data=df_clean, x="Condition", y="Has_finished", ax=ax2, color="black", size=4, alpha=0.6)

# 3. Time_to_move as a function of Pretraining
ax3 = axes[1, 0]
# Only include rows where Time_to_move is not NaN and Has_finished is 1
df_time = df_clean[(df_clean["Time_to_move"].notna()) & (df_clean["Has_finished"] == 1)]

if len(df_time) > 0:
    sns.boxplot(data=df_time, x="Pretraining", y="Time_to_move", ax=ax3, palette=["#FF6B6B", "#4ECDC4"])
    sns.stripplot(data=df_time, x="Pretraining", y="Time_to_move", ax=ax3, color="black", size=4, alpha=0.6)
    ax3.set_title("Time to Move by Pretraining Status\n(Only successful trials)")
    ax3.set_xlabel("Pretraining")
    ax3.set_ylabel("Time to Move (seconds)")
else:
    ax3.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Time to Move by Pretraining Status")

# 4. Time_to_move for the three conditions
ax4 = axes[1, 1]
df_time_condition = df_clean[(df_clean["Time_to_move"].notna()) & (df_clean["Has_finished"] == 1)]

if len(df_time_condition) > 0:
    sns.boxplot(
        data=df_time_condition, x="Condition", y="Time_to_move", ax=ax4, palette=["#FF6B6B", "#4ECDC4", "#45B7D1"]
    )
    sns.stripplot(data=df_time_condition, x="Condition", y="Time_to_move", ax=ax4, color="black", size=4, alpha=0.6)
    ax4.set_title("Time to Move by Detailed Condition\n(Only successful trials)")
    ax4.set_xlabel("Condition")
    ax4.set_ylabel("Time to Move (seconds)")
    ax4.tick_params(axis="x", rotation=45)
else:
    ax4.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("Time to Move by Detailed Condition")

plt.tight_layout()

# Save the plot as PNG
output_filename = "F1_pretraining_analysis.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Plot saved as: {output_filename}")

plt.show()

# Print summary statistics
print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)

print("\n1. Success rates by condition:")
for condition in df_clean["Condition"].unique():
    subset = df_clean[df_clean["Condition"] == condition]
    success_rate = subset["Has_finished"].mean()
    n = len(subset)
    print(f"{condition}: {success_rate:.2%} ({subset['Has_finished'].sum():.0f}/{n})")

if len(df_time_condition) > 0:
    print("\n2. Time to move statistics (successful trials only):")
    for condition in df_time_condition["Condition"].unique():
        subset = df_time_condition[df_time_condition["Condition"] == condition]
        if len(subset) > 0:
            mean_time = subset["Time_to_move"].mean()
            std_time = subset["Time_to_move"].std()
            n = len(subset)
            print(f"{condition}: {mean_time:.1f} ± {std_time:.1f} seconds (n={n})")

# Statistical tests
print("\n3. Statistical comparisons:")

# A. Pooled analysis: Pretraining vs No Pretraining (ignoring unlocked status)
print("\nA. Pooled Analysis (Pretraining vs No Pretraining):")

from scipy.stats import chi2_contingency

# Chi-square test for success rates (pooled)
contingency_pooled = pd.crosstab(df_clean["Pretraining"], df_clean["Has_finished"])
chi2_pooled, p_val_pooled, dof_pooled, expected_pooled = chi2_contingency(contingency_pooled)
print(f"   Chi-square test for success rates: χ² = {chi2_pooled:.3f}, p = {p_val_pooled:.3f}")

# Mann-Whitney U test for time differences (pooled)
df_time_pooled = df_clean[(df_clean["Time_to_move"].notna()) & (df_clean["Has_finished"] == 1)]
if len(df_time_pooled) > 2:
    pretraining_times = df_time_pooled[df_time_pooled["Pretraining"] == "y"]["Time_to_move"]
    no_pretraining_times = df_time_pooled[df_time_pooled["Pretraining"] == "n"]["Time_to_move"]

    if len(pretraining_times) > 0 and len(no_pretraining_times) > 0:
        from scipy.stats import mannwhitneyu

        statistic_pooled, p_val_pooled_time = mannwhitneyu(
            pretraining_times, no_pretraining_times, alternative="two-sided"
        )
        print(f"   Mann-Whitney U test for time to move: U = {statistic_pooled:.3f}, p = {p_val_pooled_time:.3f}")
        print(
            f"   Pretraining group: {pretraining_times.mean():.1f} ± {pretraining_times.std():.1f} s (n={len(pretraining_times)})"
        )
        print(
            f"   No pretraining group: {no_pretraining_times.mean():.1f} ± {no_pretraining_times.std():.1f} s (n={len(no_pretraining_times)})"
        )
    else:
        print("   Insufficient data for Mann-Whitney U test")
else:
    print("   Insufficient data for time comparisons")

# B. Detailed analysis: Three conditions
print("\nB. Detailed Analysis (Three Conditions):")

# Create contingency table for success rates
contingency_success = pd.crosstab(df_clean["Condition"], df_clean["Has_finished"])
chi2, p_val, dof, expected = chi2_contingency(contingency_success)
print(f"   Chi-square test for success rates: χ² = {chi2:.3f}, p = {p_val:.3f}")

# Mann-Whitney U test for time differences (if enough data)
if len(df_time_condition) > 4:  # Need at least some data in each group
    conditions = df_time_condition["Condition"].unique()
    if len(conditions) >= 2:
        from scipy.stats import mannwhitneyu

        # Compare the first two conditions with enough data
        cond1_data = df_time_condition[df_time_condition["Condition"] == conditions[0]]["Time_to_move"]
        cond2_data = df_time_condition[df_time_condition["Condition"] == conditions[1]]["Time_to_move"]

        if len(cond1_data) > 0 and len(cond2_data) > 0:
            statistic, p_val = mannwhitneyu(cond1_data, cond2_data, alternative="two-sided")
            print(f"   Mann-Whitney U test ({conditions[0]} vs {conditions[1]}): U = {statistic:.3f}, p = {p_val:.3f}")

        # If we have 3 conditions, do pairwise comparisons
        if len(conditions) == 3:
            print("   Pairwise comparisons for time to move:")
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    cond_i_data = df_time_condition[df_time_condition["Condition"] == conditions[i]]["Time_to_move"]
                    cond_j_data = df_time_condition[df_time_condition["Condition"] == conditions[j]]["Time_to_move"]
                    if len(cond_i_data) > 0 and len(cond_j_data) > 0:
                        stat, p = mannwhitneyu(cond_i_data, cond_j_data, alternative="two-sided")
                        print(f"     {conditions[i]} vs {conditions[j]}: U = {stat:.3f}, p = {p:.3f}")
else:
    print("   Insufficient data for detailed time comparisons")

print("\n" + "=" * 50)
