#!/usr/bin/env Rscript
# R comparison script for statgpu benchmark

# Read command line args
args <- commandArgs(trailingOnly = TRUE)
csv_path <- ifelse(length(args) > 0, args[1], "/tmp/statgpu_benchmark_data.csv")

# Check if file exists
if (!file.exists(csv_path)) {
    cat("Error: CSV file not found:", csv_path, "\n")
    quit(status = 1)
}

# Read data
cat("Reading data from:", csv_path, "\n")
data <- read.csv(csv_path)
n_features <- ncol(data) - 2  # exclude y and y_binary
X <- data[, 1:n_features]
y <- data$y
y_binary <- data$y_binary

cat("Dataset:", nrow(data), "samples x", n_features, "features\n\n")

# ============================================================================
# 1. Linear Regression
# ============================================================================
cat("================================================================================\n")
cat("1. LINEAR REGRESSION (R lm)\n")
cat("================================================================================\n\n")

start_time <- proc.time()
lm_model <- lm(y ~ ., data=data)
lm_time <- (proc.time() - start_time)["elapsed"] * 1000

cat("Time:", round(lm_time, 2), "ms\n")
cat("R-squared:", round(summary(lm_model)$r.squared, 6), "\n")
cat("Coefficients (first 5):\n")
print(round(coef(lm_model)[1:6], 6))

# Save coefficients for comparison
lm_coef <- coef(lm_model)[-1]  # exclude intercept
lm_intercept <- coef(lm_model)[1]

# ============================================================================
# 2. Logistic Regression
# ============================================================================
cat("\n================================================================================\n")
cat("2. LOGISTIC REGRESSION (R glm)\n")
cat("================================================================================\n\n")

start_time <- proc.time()
logit_model <- glm(y_binary ~ ., data=data, family=binomial(link="logit"))
logit_time <- (proc.time() - start_time)["elapsed"] * 1000

cat("Time:", round(logit_time, 2), "ms\n")
cat("Coefficients (first 5):\n")
print(round(coef(logit_model)[1:6], 6))

# Summary
cat("\n================================================================================\n")
cat("SUMMARY\n")
cat("================================================================================\n\n")

cat("Model          | Time (ms) | R²/Deviance\n")
cat("---------------|-----------|-------------\n")
cat(sprintf("%-14s | %9.2f | %11.6f\n", "Linear (lm)", lm_time, summary(lm_model)$r.squared))
cat(sprintf("%-14s | %9.2f | %11.4f\n", "Logistic (glm)", logit_time, logit_model$deviance))

cat("\n================================================================================\n")
cat("R results saved for comparison with Python\n")
cat("================================================================================\n")

# Save results to file for Python to read
results <- list(
    linear = list(
        time_ms = lm_time,
        r_squared = summary(lm_model)$r.squared,
        coef = as.numeric(lm_coef),
        intercept = as.numeric(lm_intercept)
    ),
    logistic = list(
        time_ms = logit_time,
        deviance = logit_model$deviance,
        coef = as.numeric(coef(logit_model)[-1]),
        intercept = as.numeric(coef(logit_model)[1])
    )
)

# Print coefficients in format easy to parse
 cat("\n=== LINEAR COEFFICIENTS ===\n")
cat(lm_coef, sep=", ")
cat("\n")

cat("\n=== LOGISTIC COEFFICIENTS ===\n")
cat(coef(logit_model)[-1], sep=", ")
cat("\n")
