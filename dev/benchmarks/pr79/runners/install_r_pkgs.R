# Install R packages needed for PR79 benchmark references
pkgs <- c("survival", "glmnet", "plm", "sandwich", "lmtest", "jsonlite")
install.packages(pkgs, repos = "https://cloud.r-project.org", quiet = TRUE)
cat("Installed packages:\n")
print(installed.packages()[pkgs, "Version"])
