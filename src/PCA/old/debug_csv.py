#!/usr/bin/env python3

import pandas as pd
import os

# Change to the correct directory
os.chdir("/home/matthias/ballpushing_utils/src/PCA")

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check the Permutation_FDR_significant column
print("\nPermutation_FDR_significant value counts:")
print(df["Permutation_FDR_significant"].value_counts())

# Check the data types
print(f"\nType of Permutation_FDR_significant: {df['Permutation_FDR_significant'].dtype}")

# Get significant genotypes
significant_mask = df["Permutation_FDR_significant"] == True
significant_genotypes = df[significant_mask]

print(f"\nNumber of significant genotypes: {len(significant_genotypes)}")
print("First 10 significant genotypes:")
for i, row in significant_genotypes.head(10).iterrows():
    print(f"  {row['Nickname']} - FDR p={row['Permutation_FDR_pval']:.4f}")

# Also check by FDR p-value
print("\nChecking by FDR p-value < 0.05:")
fdr_significant = df[df["Permutation_FDR_pval"] < 0.05]
print(f"Number with FDR p < 0.05: {len(fdr_significant)}")

# Check by raw permutation p-value (what the heatmap uses)
print("\nChecking by raw permutation p-value < 0.05 (heatmap criterion):")
raw_significant = df[df["Permutation_pval"] < 0.05]
print(f"Number with raw permutation p < 0.05: {len(raw_significant)}")

print("\nFirst 10 raw significant genotypes:")
for i, row in raw_significant.head(10).iterrows():
    print(
        f"  {row['Nickname']} - raw p={row['Permutation_pval']:.4f}, FDR p={row['Permutation_FDR_pval']:.4f}, FDR sig={row['Permutation_FDR_significant']}"
    )
