# Elastic Net Benchmark: R (glmnet) vs Python (statgpu)
# Run on remote GPU server with R and glmnet installed

library(glmnet)
library(jsonlite)

# ========== Configuration ==========
HOST <- "hz-4.matpool.com"
PORT <- 27609

cat("=" , rep("=", 69), "\n", sep="")
cat("Elastic Net Benchmark: R glmnet\n")
cat("=", rep("=", 69), "\n", sep="")

# ========== Test configurations ==========
test_configs <- list(
  list(name="small_data", n=200, p=20, true_nonzero=5, noise=0.5),
  list(name="medium_data", n=1000, p=50, true_nonzero=10, noise=0.5),
  list(name="large_data", n=5000, p=100, true_nonzero=20, noise=0.5),
  list(name="high_dim_data", n=100, p=200, true_nonzero=20, noise=0.5),
  list(name="sparse_coef", n=500, p=100, true_nonzero=5, noise=0.5),
  list(name="high_noise", n=500, p=50, true_nonzero=10, noise=2.0)
)

all_results <- list()

for (config in test_configs) {
  cat("\n", rep("=", 60), "\n", sep="")
  cat(sprintf("Test: %s (n=%d, p=%d)\n", config$name, config$n, config$p))
  cat(rep("=", 60), "\n", sep="")

  # Generate reproducible data
  set.seed(42)
  X <- matrix(rnorm(config$n * config$p), nrow=config$n, ncol=config$p)
  true_coef <- c(rnorm(config$true_nonzero), rep(0, config$p - config$true_nonzero))
  y <- X %*% true_coef + rnorm(config$n) * config$noise

  # Save data for Python comparison
  write.csv(X, sprintf("/root/benchmark_X_%s.csv", config$name), row.names=FALSE)
  write.csv(y, sprintf("/root/benchmark_y_%s.csv", config$name), row.names=FALSE)
  write.csv(true_coef, sprintf("/root/benchmark_true_coef_%s.csv", config$name), row.names=FALSE)

  # ========== Run glmnet ==========
  cat("\n[glmnet]\n")

  start_time <- Sys.time()
  fit <- glmnet(X, y, alpha=0.5, lambda=1.0, thresh=1e-8, maxit=5000)
  end_time <- Sys.time()
  fit_time <- as.numeric(difftime(end_time, start_time, units="secs")) * 1000

  # Extract results
  coef_glmnet <- as.vector(coef(fit))[-1]
  intercept_glmnet <- as.vector(coef(fit))[1]

  cat(sprintf("  coef_norm: %.6f\n", norm(coef_glmnet, "2")))
  cat(sprintf("  intercept: %.6f\n", intercept_glmnet))
  cat(sprintf("  fit_time: %.2f ms\n", fit_time))

  # Save results
  result <- list(
    name = config$name,
    n_samples = config$n,
    n_features = config$p,
    backend = "glmnet",
    coef = coef_glmnet,
    intercept = intercept_glmnet,
    coef_norm = norm(coef_glmnet, "2"),
    fit_time_ms = fit_time,
    n_iterations = fit$df
  )

  # Save individual result
  json_file <- sprintf("/root/glmnet_result_%s.json", config$name)
  writeLines(toJSON(result, digits=16, auto_unbox=TRUE), json_file)

  all_results[[config$name]] <- result
}

# ========== Save combined results ==========
combined_results <- list(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  backend = "glmnet",
  results = all_results
)

output_file <- "/root/benchmark_glmnet_all.json"
writeLines(toJSON(combined_results, digits=16, auto_unbox=TRUE, pretty=TRUE), output_file)

cat("\n", rep("=", 60), "\n", sep="")
cat(sprintf("Results saved to: %s\n", output_file))
cat(rep("=", 60), "\n", sep="")
