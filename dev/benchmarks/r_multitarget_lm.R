#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript r_multitarget_lm.R <train_csv> <test_csv>")
}

train_path <- args[[1]]
test_path <- args[[2]]

suppressWarnings({
  train <- read.csv(train_path)
  test <- read.csv(test_path)
})

x_cols <- grep("^x", names(train), value = TRUE)
y_cols <- grep("^y", names(train), value = TRUE)

if (length(x_cols) == 0 || length(y_cols) == 0) {
  stop("Expected columns x1..xp and y1..yt in train/test CSV")
}

formula_str <- paste0("cbind(", paste(y_cols, collapse = ","), ") ~ ", paste(x_cols, collapse = "+"))
f <- as.formula(formula_str)

t0 <- proc.time()
m <- lm(f, data = train)
t1 <- proc.time()

pred_t0 <- proc.time()
pred <- predict(m, newdata = test)
pred_t1 <- proc.time()

coef_mat <- coef(m)
coef_json <- lapply(seq_len(ncol(coef_mat)), function(j) as.numeric(coef_mat[, j]))

out <- list(
  fit_ms = as.numeric((t1 - t0)[3]) * 1000,
  pred_ms = as.numeric((pred_t1 - pred_t0)[3]) * 1000,
  coeffs = coef_json,
  pred = unname(pred)
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE))
