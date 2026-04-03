suppressPackageStartupMessages(library(survival))
set.seed(456)

gen_data <- function(n, p, tie_digits = 2L) {
  X <- matrix(rnorm(n * p), n, p)
  beta <- rnorm(p, sd = 0.35)
  lin <- X %*% beta
  u <- pmin(pmax(runif(n), 1e-12), 1 - 1e-12)
  base <- 0.03
  t_true <- -log(u) / (base * exp(pmin(pmax(lin, -20), 20)))
  censor <- rexp(n, rate = 1 / median(t_true))
  event <- as.integer(t_true <= censor)
  time_obs <- round(pmin(t_true, censor), tie_digits)
  d <- data.frame(time = time_obs, event = event, X)
  names(d) <- c("time", "event", paste0("x", seq_len(p)))
  d
}

bench_once <- function(n, p, ties, warmup = FALSE) {
  d <- gen_data(n, p)
  form <- as.formula(
    paste("Surv(time,event)~", paste(names(d)[3:ncol(d)], collapse = "+"))
  )
  tm <- system.time({
    fit <- coxph(form, data = d, ties = ties)
  })["elapsed"]
  if (!warmup) {
    cat(sprintf("%s n=%d p=%d elapsed=%.3fs loglik=%.4f\n", ties, n, p, as.numeric(tm), fit$loglik[2]))
  }
  invisible(as.numeric(tm))
}

run_case <- function(n, p, ties, reps = 3L) {
  # warmup to reduce one-time overhead noise
  bench_once(min(2000L, n), p, ties, warmup = TRUE)
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    times[i] <- bench_once(n, p, ties, warmup = FALSE)
  }
  cat(sprintf("SUMMARY %s n=%d p=%d median=%.3fs min=%.3fs max=%.3fs\n\n",
              ties, n, p, median(times), min(times), max(times)))
}

cat("=== R coxph large-scale benchmark ===\n")
cat(sprintf("R version: %s\n\n", R.version.string))

# Breslow: scale n and p
for (cfg in list(
  c(10000L, 20L),
  c(30000L, 30L),
  c(50000L, 40L),
  c(100000L, 50L)
)) {
  run_case(cfg[1], cfg[2], "breslow", reps = 3L)
}

# Efron: slightly smaller/larger mixed set
for (cfg in list(
  c(10000L, 20L),
  c(30000L, 30L),
  c(50000L, 40L)
)) {
  run_case(cfg[1], cfg[2], "efron", reps = 3L)
}
