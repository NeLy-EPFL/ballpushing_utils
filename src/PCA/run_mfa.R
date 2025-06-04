library(FactoMineR)
library(data.table)

# Read data (expects a CSV with all metrics and a group vector)
data <- fread("mfa_input.csv")
groups <- scan("mfa_groups.txt", what=integer(), quiet=TRUE)

# Run MFA
mfa_res <- MFA(data, group=groups, type=rep("s", length(groups)), ncp=30, name.group=c("temporal", "static"))

# Save results
write.csv(mfa_res$ind$coord, "mfa_scores.csv", row.names=FALSE)
write.csv(mfa_res$quanti.var$coord, "mfa_loadings.csv", row.names=TRUE)
write.csv(mfa_res$group$contrib, "mfa_block_contrib.csv", row.names=TRUE)
# Add to run_mfa.R
write.csv(mfa_res$eig, "mfa_eigenvalues.csv", row.names=FALSE)

# In R after MFA
plot(mfa_res, choix="group")