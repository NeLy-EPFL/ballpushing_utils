# Install required packages if not already installed
if (!requireNamespace("FactoMineR", quietly = TRUE)) {
  install.packages("FactoMineR")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}

library(FactoMineR)
library(data.table)

# Get suffix from command line argument (default empty string)
args <- commandArgs(trailingOnly=TRUE)
suffix <- ifelse(length(args) > 0, args[[1]], "")

# Read data (expects a CSV with all metrics and a group vector)
data <- fread(paste0("mfa_input", suffix, ".csv"))
groups <- scan(paste0("mfa_groups", suffix, ".txt"), what=integer(), quiet=TRUE)

# Run MFA
mfa_res <- MFA(data, group=groups, type=rep("s", length(groups)), ncp=30, name.group=c("temporal", "static"))

# Save results
write.csv(mfa_res$ind$coord, paste0("mfa_scores", suffix, ".csv"), row.names=FALSE)
write.csv(mfa_res$quanti.var$coord, paste0("mfa_loadings", suffix, ".csv"), row.names=TRUE)
write.csv(mfa_res$group$contrib, paste0("mfa_block_contrib", suffix, ".csv"), row.names=TRUE)
write.csv(mfa_res$eig, paste0("mfa_eigenvalues", suffix, ".csv"), row.names=FALSE)

# In R after MFA
plot(mfa_res, choix="group")