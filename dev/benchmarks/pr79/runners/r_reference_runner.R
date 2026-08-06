#!/usr/bin/env Rscript
# PR79 benchmark — R reference runner.
# Reads session manifest (JSON) and runs R reference models on the same
# case data, writing JSONL checkpoint records.

library(jsonlite)
library(survival)
library(glmnet)
library(plm)
library(sandwich)
library(lmtest)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript r_reference_runner.R <results_dir>")
}
results_dir <- args[1]
dir.create(file.path(results_dir, "raw", "r"), showWarnings = FALSE, recursive = TRUE)

# ---- helpers ----

write_run <- function(run, out_file) {
  # Append one JSON record per line (JSONL)
  cat(toJSON(run, auto_unbox = TRUE, pretty = FALSE), "\n",
      file = out_file, append = TRUE)
}

safe_run <- function(expr, run_key, out_file) {
  result <- tryCatch(expr, error = function(e) {
    list(status = "failed", error = conditionMessage(e))
  })
  if (is.list(result) && !is.null(result[["status"]]) && result[["status"]] == "failed") {
    run <- list(
      run_key = run_key,
      status  = "failed",
      error   = result[["error"]]
    )
    write_run(run, out_file)
    return(NULL)
  }
  return(result)
}

# ---- Linear via lm ----

bench_linear_lm <- function(X, y, weights = NULL, cov_type = "nonrobust") {
  df <- as.data.frame(cbind(y = y, X))
  colnames(df)[-1] <- paste0("x", seq_len(ncol(X)))
  if (is.null(weights)) {
    fit <- lm(y ~ ., data = df)
  } else {
    fit <- lm(y ~ ., data = df, weights = weights)
  }

  coef_all <- coef(fit)
  n <- nrow(X)
  k <- length(coef_all)

  # Covariance
  if (cov_type == "hc1") {
    vcov_mat <- vcovHC(fit, type = "HC1")
  } else if (cov_type == "hc0") {
    vcov_mat <- vcovHC(fit, type = "HC0")
  } else {
    vcov_mat <- vcov(fit)
  }
  bse <- sqrt(diag(vcov_mat))

  list(
    coef_     = as.numeric(coef_all[-1]),
    intercept_= as.numeric(coef_all[1]),
    bse       = as.numeric(bse[-1]),
    rsquared  = as.numeric(summary(fit)$r.squared),
    aic       = as.numeric(AIC(fit)),
    bic       = as.numeric(BIC(fit)),
    loglik    = as.numeric(logLik(fit))
  )
}

# ---- Ridge via glmnet ----

bench_ridge_glmnet <- function(X, y, alpha = 1.0, weights = NULL) {
  # glmnet uses (1/2n)*RSS + lambda*||beta||_2^2
  # statgpu uses average loss: (1/n)*RSS + alpha*||beta||_2^2
  # So glmnet lambda = statgpu alpha (no n factor needed for average-loss convention,
  # but glmnet normalizes differently — we pass alpha * n / 2)
  n <- nrow(X)
  lambda <- alpha * n / 2  # convert statgpu alpha to glmnet lambda

  if (is.null(weights)) {
    fit <- glmnet(X, y, alpha = 0, lambda = lambda, standardize = FALSE)
  } else {
    fit <- glmnet(X, y, alpha = 0, lambda = lambda, standardize = FALSE,
                  weights = weights)
  }
  coefs <- as.numeric(coef(fit))
  list(
    coef_     = coefs[-1],
    intercept_= coefs[1]
  )
}

# ---- CoxPH via survival::coxph ----

bench_coxph_survival <- function(X, time, event, ties = "efron", entry = NULL,
                                 penalty = 0.0) {
  df <- as.data.frame(cbind(time = time, event = event, X))
  colnames(df)[-(1:2)] <- paste0("x", seq_len(ncol(X)))
  rhs <- paste0("x", seq_len(ncol(X)), collapse = " + ")
  form <- as.formula(paste0("Surv(time, event) ~ ", rhs))

  if (!is.null(entry)) {
    df$entry <- entry
    form <- as.formula(paste0("Surv(entry, time, event) ~ ", rhs))
  }

  fit <- coxph(form, data = df, ties = ties,
               control = coxph.control(iter.max = 30))

  list(
    coef_     = as.numeric(coef(fit)),
    bse       = as.numeric(sqrt(diag(vcov(fit)))),
    loglik    = as.numeric(fit$loglik[2]),
    converged = as.numeric(fit$info[["convergence"]] == 0)
  )
}

# ---- PooledOLS via plm ----

bench_pooled_plm <- function(X, y, entity, time_idx, cov_type = "nonrobust") {
  df <- as.data.frame(cbind(y = y, X))
  colnames(df)[-(1)] <- paste0("x", seq_len(ncol(X)))
  df$entity <- entity
  df$time   <- time_idx

  pdata <- pdata.frame(df, index = c("entity", "time"))
  rhs <- paste0("x", seq_len(ncol(X)), collapse = " + ")
  form <- as.formula(paste0("y ~ ", rhs))
  fit <- plm(form, data = pdata, model = "pooling")

  # Covariance matrix
  if (cov_type == "clustered") {
    vcv <- vcovHC(fit, type = "HC0", cluster = "group")
  } else if (cov_type == "hc1") {
    vcv <- vcovHC(fit, type = "HC1")
  } else {
    vcv <- vcovHC(fit, type = "HC0")
  }
  bse <- sqrt(diag(vcv))

  list(
    coef_     = as.numeric(coef(fit)),
    bse       = as.numeric(bse),
    rsquared  = as.numeric(summary(fit)$r.squared[1])
  )
}

# ---- Main ----

main <- function() {
  out_file <- file.path(results_dir, "raw", "r", "r_reference_runs.jsonl")

  # Generate data inline (same seeds as Python generators)
  set.seed(42)

  cat("=== R Reference Runner ===\n")

  # Linear full-rank
  cat("Linear (lm): ")
  n <- 1000; p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta <- rnorm(p); y <- as.numeric(X %*% beta + rnorm(n, sd = 0.3))
  res <- safe_run(bench_linear_lm(X, y), "ref-linear-lm", out_file)
  if (!is.null(res)) cat(sprintf("coef1=%.4f, R2=%.4f\n", res$coef_[1], res$rsquared))

  # Linear HC1
  cat("Linear HC1 (lm + sandwich): ")
  res <- safe_run(bench_linear_lm(X, y, cov_type = "hc1"), "ref-linear-lm-hc1", out_file)
  if (!is.null(res)) cat(sprintf("bse1=%.6f\n", res$bse[1]))

  # Ridge via glmnet
  cat("Ridge (glmnet): ")
  n_r <- 200; p_r <- 8
  Xr <- matrix(rnorm(n_r * p_r), n_r, p_r)
  yr <- as.numeric(Xr %*% rnorm(p_r) + rnorm(n_r, sd = 0.3))
  res <- safe_run(bench_ridge_glmnet(Xr, yr, alpha = 1.0), "ref-ridge-glmnet", out_file)
  if (!is.null(res)) cat(sprintf("intercept=%.4f\n", res$intercept_))

  # CoxPH via survival
  cat("CoxPH (survival): ")
  n_c <- 200; p_c <- 4
  Xc <- matrix(rnorm(n_c * p_c), n_c, p_c)
  eta <- as.numeric(Xc %*% c(0.5, -0.3, 0.2, 0.0))
  t_raw <- rexp(n_c) / exp(eta)
  c_time <- rexp(n_c, rate = 1 / quantile(t_raw, 0.7))
  event <- as.integer(t_raw <= c_time)
  time <- pmin(t_raw, c_time)
  res <- safe_run(bench_coxph_survival(Xc, time, event, ties = "efron"),
                  "ref-coxph-r", out_file)
  if (!is.null(res)) cat(sprintf("coef1=%.4f, ll=%.4f\n", res$coef_[1], res$loglik))

  # Panel PooledOLS via plm
  cat("Panel (plm): ")
  n_ent <- 30; n_per <- 5
  Xp <- matrix(rnorm(n_ent * n_per * 3), n_ent * n_per, 3)
  entity <- rep(1:n_ent, each = n_per)
  time_idx <- rep(1:n_per, n_ent)
  yp <- as.numeric(Xp %*% c(1.0, -0.5, 0.3) + rnorm(n_ent * n_per, sd = 0.2))
  res <- safe_run(bench_pooled_plm(Xp, yp, entity, time_idx),
                  "ref-pooled-plm", out_file)
  if (!is.null(res)) cat(sprintf("coef1=%.4f\n", res$coef_[1]))

  cat(sprintf("\nRuns saved to %s\n", out_file))
}

main()
