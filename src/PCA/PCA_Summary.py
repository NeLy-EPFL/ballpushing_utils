import pandas as pd

# Load significant hits from each approach
mfa_df = pd.read_csv("mfa_stats_results.csv")
fpca_df = pd.read_csv("fpca_temporal_stats_results.csv")
static_df = pd.read_csv("static_pca_stats_results.csv")
static_sparse_df = pd.read_csv("static_sparsepca_stats_results.csv")

# Filter for significant results
mfa_hits = mfa_df[mfa_df["FDR_significant"]]
fpca_hits = fpca_df[fpca_df["FDR_significant"]]
static_hits = static_df[static_df["FDR_significant"]]
sparse_hits = static_sparse_df[static_sparse_df["FDR_significant"]]

# Add a column to indicate the source
mfa_hits["Approach"] = "MFA"
fpca_hits["Approach"] = "FPCA_Temporal"
static_hits["Approach"] = "PCA_Static"
sparse_hits["Approach"] = "PCA_Static_Sparse"


# Standardize columns for concatenation
def standardize(df):
    cols = ["Nickname", "Control", "Dimension", "FDR_pval", "Approach"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    return df[cols]


mfa_hits = standardize(mfa_hits)
fpca_hits = standardize(fpca_hits)
static_hits = standardize(static_hits)
sparse_hits = standardize(sparse_hits)

# Concatenate all hits
all_hits = pd.concat([mfa_hits, fpca_hits, static_hits, sparse_hits], ignore_index=True)

# Group by Nickname and aggregate details
summary = (
    all_hits.groupby("Nickname")
    .agg(
        Approaches=("Approach", lambda x: ", ".join(sorted(set(x)))),
        N_Approaches=("Approach", "nunique"),
        Details=("Dimension", lambda x: "; ".join(x.astype(str))),
        Min_FDR_pval=("FDR_pval", "min"),
    )
    .reset_index()
)

# Add a column for which approaches were significant (1, 2, or 3)
summary["Consistency"] = summary["N_Approaches"]

# Save to CSV
summary.to_csv("final_consistent_hits_summary.csv", index=False)
print("Summary saved to final_consistent_hits_summary.csv")
