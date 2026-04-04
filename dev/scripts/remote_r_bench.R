suppressPackageStartupMessages(library(survival))
set.seed(123)

bench <- function(n, p, ties) {
  X <- matrix(rnorm(n * p), n, p)
  beta <- rnorm(p, sd = 0.35)
  lin <- X %*% beta
  u <- pmin(pmax(runif(n), 1e-12), 1 - 1e-12)
  base <- 0.03
  t_true <- -log(u) / (base * exp(pmin(pmax(lin, -20), 20)))
  censor <- rexp(n, rate = 1 / median(t_true))
  event <- as.integer(t_true <= censor)
  time_obs <- round(pmin(t_true, censor), 2)

  d <- data.frame(time = time_obs, event = event, X)
  names(d) <- c("time", "event", paste0("x", 1:p))
  form <- as.formula(paste("Surv(time,event)~", paste(names(d)[3:ncol(d)], collapse = "+")))

  t <- system.time({
    fit <- coxph(form, data = d, ties = ties)
  })["elapsed"]

  cat(sprintf("%s n=%d p=%d elapsed=%.3fs loglik=%.4f\n", ties, n, p, as.numeric(t), fit$loglik[2]))
}

bench(400, 6, "breslow")
bench(1200, 10, "breslow")
bench(3000, 14, "breslow")
bench(800, 7, "efron")
