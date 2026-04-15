# ============================================================
# TELEMEDICINE SIMULATION — METRICS EXTRACTION
# Loads JSON results, computes all evaluation metrics,
# exports a long-format CSV ready for ggplot2
# ============================================================
# 
# DEPENDENCIES
#   install.packages(c("jsonlite", "dplyr", "tidyr"))
#
# INPUT FILES
#   1. JSON results file — path set in JSON_PATH below
#   2. DGP variables (optional, for SNR) — see DGP SECTION below
#
# OUTPUT
#   telemed_metrics_long.csv  — one row per learner per metric
#   telemed_dgp_summary.csv   — DGP-level diagnostics (one row)
# ============================================================

library(jsonlite)
library(dplyr)
library(tidyr)
library(MASS)

# ============================================================
# CONFIGURATION
# ============================================================

JSON_PATH   <- "telemed_simulation_200.json"   # path to your JSON file
OUTPUT_LONG <- "telemed_metrics_long.csv"
OUTPUT_DGP  <- "telemed_dgp_summary.csv"

LEARNERS <- c("rlasso","rboost","rkern",
              "slasso","sboost","skern",
              "tlasso","tboost","tkern",
              "xlasso","xboost","xkern",
              "ols_inter","zero_pred")

# ============================================================
# LOAD JSON
# ============================================================

cat("Loading JSON...\n")
raw      <- fromJSON(JSON_PATH, simplifyVector = FALSE)
all_iters <- raw$iterations

# Filter to valid (non-skipped) iterations
valid <- Filter(function(it) !it$skipped[[1]], all_iters)
n_valid <- length(valid)
cat("Valid iterations:", n_valid, "\n")

# ============================================================
# HELPER — safely extract a scalar from a possibly-list value
# ============================================================

s <- function(x) {
  if (is.null(x)) return(NA_real_)
  v <- unlist(x)
  if (length(v) == 0) return(NA_real_)
  as.numeric(v[[1]])
}

b <- function(x) {
  # same as s() but returns logical
  if (is.null(x)) return(NA)
  v <- unlist(x)
  if (length(v) == 0) return(NA)
  as.logical(v[[1]])
}

# ============================================================
# EXTRACT PER-ITERATION PER-LEARNER DATA INTO A FLAT DATA FRAME
# ============================================================

cat("Extracting per-iteration results...\n")

rows <- list()

for (it in valid) {
  
  seed         <- s(it$seed)
  true_ate_test <- s(it$true_ate_test)
  tau_sd       <- s(it$tau_sd)
  true_sg1     <- s(it$true_sg1)
  true_sg2     <- s(it$true_sg2)
  
  # Per-iteration zero_pred MSE — used as dynamic baseline
  zero_raw_mse <- s(it$results$zero_pred$raw_mse)
  zero_nmse    <- s(it$results$zero_pred$norm_mse)

  for (lname in LEARNERS) {

    r <- it$results[[lname]]

    rows[[length(rows) + 1]] <- list(
      seed          = seed,
      learner       = lname,
      true_ate_test = true_ate_test,
      tau_sd        = tau_sd,
      true_sg1      = true_sg1,
      true_sg2      = true_sg2,
      zero_raw_mse  = zero_raw_mse,
      zero_nmse     = zero_nmse,
      raw_mse       = s(r$raw_mse),
      norm_mse      = s(r$norm_mse),
      rank_corr     = s(r$rank_corr),
      mean_tau      = s(r$mean_tau),
      est_sg1       = s(r$est_sg1),
      est_sg2       = s(r$est_sg2),
      rec_sg1       = s(r$rec_sg1),
      rec_sg2       = s(r$rec_sg2),
      sign_sg1      = b(r$sign_sg1),
      sign_sg2      = b(r$sign_sg2),
      failed        = b(r$failed)
    )
  }
}

df_iter <- bind_rows(lapply(rows, as.data.frame))

# Remove failed learner runs
df_iter <- df_iter %>% filter(!failed | is.na(failed))

cat("Total learner-iteration rows:", nrow(df_iter), "\n")

# ============================================================
# DERIVED COLUMNS
# ============================================================

df_iter <- df_iter %>%
  mutate(
    # Ratio vs zero predictor — below 1.0 means beats zero prediction
    mse_vs_zero    = raw_mse  / zero_raw_mse,
    nmse_vs_zero   = norm_mse / zero_nmse,
    
    # ATE bias — how far the learner's mean prediction is from true ATE
    ate_bias        = mean_tau - true_ate_test,
    
    # ATE quartile — used for SG2 sign recovery stratification
    ate_quartile    = ntile(true_ate_test, 4),
    
    # Sign of both subgroups correct simultaneously
    sign_both       = sign_sg1 & sign_sg2
  )

# ============================================================
# METRIC COMPUTATION
# Per-learner summaries, returned in long format
# ============================================================

cat("Computing metrics...\n")

pct <- function(x, p) quantile(x, p / 100, na.rm = TRUE)

compute_metrics <- function(df) {
  
  learner_name <- unique(df$learner)
  
  tau_var_mean <- mean(df$tau_sd^2, na.rm = TRUE)

  # --- MSE distribution ---
  mse_vals     <- df$raw_mse

  # --- nMSE distribution ---
  nmse_vals    <- df$norm_mse

  # --- Beats zero predictor ---
  beats_zero   <- mean(df$mse_vs_zero < 1.0, na.rm = TRUE)
  pct_bad      <- mean(df$mse_vs_zero > 1.0, na.rm = TRUE)
  
  # --- Rank correlation ---
  rc_vals      <- df$rank_corr[!is.na(df$rank_corr)]
  rc_mean      <- mean(rc_vals)
  rc_sd        <- sd(rc_vals)
  rc_p10       <- pct(rc_vals, 10)
  rc_p50       <- pct(rc_vals, 50)
  rc_p90       <- pct(rc_vals, 90)
  
  # --- Sign recovery ---
  sign_sg1_rate  <- mean(df$sign_sg1, na.rm = TRUE)
  sign_sg2_rate  <- mean(df$sign_sg2, na.rm = TRUE)
  sign_both_rate <- mean(df$sign_both, na.rm = TRUE)
  
  # --- SG2 sign recovery by ATE quartile ---
  # Q1 = most negative ATE (hardest to detect harmful subgroup)
  # Q4 = least negative / near-zero ATE
  sg2_by_q <- df %>%
    group_by(ate_quartile) %>%
    summarise(sg2_rate = mean(sign_sg2, na.rm = TRUE), .groups = "drop") %>%
    arrange(ate_quartile)
  
  sg2_q1 <- sg2_by_q$sg2_rate[sg2_by_q$ate_quartile == 1]
  sg2_q4 <- sg2_by_q$sg2_rate[sg2_by_q$ate_quartile == 4]
  sg2_q1 <- ifelse(length(sg2_q1) == 0, NA, sg2_q1)
  sg2_q4 <- ifelse(length(sg2_q4) == 0, NA, sg2_q4)
  
  # --- Recovery ratios ---
  rec1 <- df$rec_sg1
  rec2 <- df$rec_sg2
  
  # --- ATE bias ---
  bias_vals  <- df$ate_bias
  bias_mean  <- mean(bias_vals, na.rm = TRUE)
  bias_sd    <- sd(bias_vals,   na.rm = TRUE)
  
  # --- Bias-variance decomposition (population level, raw MSE scale) ---
  # mse = bias^2 + variance
  # ATE-level bias^2 using raw MSE
  bias_sq   <- mean(bias_vals^2, na.rm = TRUE)
  mse_mean  <- mean(mse_vals,    na.rm = TRUE)
  variance  <- max(0, mse_mean - bias_sq)

  # --- Bias-variance decomposition (normalised MSE scale, for reference) ---
  norm_bias_sq  <- mean(bias_vals^2, na.rm = TRUE) / tau_var_mean
  norm_mse_mean <- mean(nmse_vals,   na.rm = TRUE)
  norm_variance <- max(0, norm_mse_mean - norm_bias_sq)

  # Assemble into named vector — will become rows in long format
  list(
    learner = learner_name,

    # MSE (raw)
    mse_median = pct(mse_vals, 50),
    mse_p10    = pct(mse_vals, 10),
    mse_p25    = pct(mse_vals, 25),
    mse_p75    = pct(mse_vals, 75),
    mse_p90    = pct(mse_vals, 90),
    mse_mean   = mse_mean,

    # nMSE (normalised, kept for reference)
    nmse_median = pct(nmse_vals, 50),
    nmse_p10    = pct(nmse_vals, 10),
    nmse_p25    = pct(nmse_vals, 25),
    nmse_p75    = pct(nmse_vals, 75),
    nmse_p90    = pct(nmse_vals, 90),
    nmse_mean   = norm_mse_mean,

    # vs zero predictor
    pct_beats_zero = beats_zero,
    pct_worse_zero = pct_bad,
    
    # rank correlation
    rank_corr_mean = rc_mean,
    rank_corr_sd   = rc_sd,
    rank_corr_p10  = rc_p10,
    rank_corr_p50  = rc_p50,
    rank_corr_p90  = rc_p90,
    
    # sign recovery
    sign_sg1_rate  = sign_sg1_rate,
    sign_sg2_rate  = sign_sg2_rate,
    sign_both_rate = sign_both_rate,
    
    # SG2 sign recovery by ATE quartile (substantive HTE finding)
    sign_sg2_ate_q1 = sg2_q1,   # hardest: most negative ATE
    sign_sg2_ate_q4 = sg2_q4,   # easiest: near-zero ATE
    
    # Recovery ratios — SG1
    rec_sg1_p10 = pct(rec1, 10),
    rec_sg1_p25 = pct(rec1, 25),
    rec_sg1_p50 = pct(rec1, 50),
    rec_sg1_p75 = pct(rec1, 75),
    rec_sg1_p90 = pct(rec1, 90),
    
    # Recovery ratios — SG2
    rec_sg2_p10 = pct(rec2, 10),
    rec_sg2_p25 = pct(rec2, 25),
    rec_sg2_p50 = pct(rec2, 50),
    rec_sg2_p75 = pct(rec2, 75),
    rec_sg2_p90 = pct(rec2, 90),
    
    # ATE bias
    ate_bias_mean = bias_mean,
    ate_bias_sd   = bias_sd,
    
    # Bias-variance decomposition (raw MSE scale)
    bias_sq      = bias_sq,
    variance     = variance,

    # Bias-variance decomposition (normalised MSE scale, for reference)
    norm_bias_sq  = norm_bias_sq,
    norm_variance = norm_variance
  )
}

# Apply to each learner
metrics_list <- df_iter %>%
  group_by(learner) %>%
  group_map(~ compute_metrics(.x), .keep = TRUE)

# Combine into wide data frame
df_wide <- bind_rows(lapply(metrics_list, as.data.frame))

# ============================================================
# PIVOT TO LONG FORMAT FOR GGPLOT
# One row per learner per metric
# Columns: learner | metric | value
# ============================================================

df_long <- df_wide %>%
  pivot_longer(
    cols      = -learner,
    names_to  = "metric",
    values_to = "value"
  ) %>%
  # Add a grouping column so plots can facet by metric family
  mutate(
    metric_group = case_when(
      grepl("^mse",         metric) ~ "MSE",
      grepl("^nmse",        metric) ~ "nMSE",
      grepl("zero",         metric) ~ "vs_zero_pred",
      grepl("rank_corr",    metric) ~ "rank_correlation",
      grepl("sign",         metric) ~ "sign_recovery",
      grepl("^rec_sg1",     metric) ~ "recovery_ratio_sg1",
      grepl("^rec_sg2",     metric) ~ "recovery_ratio_sg2",
      grepl("ate_bias",     metric) ~ "ate_bias",
      grepl("norm_bias|norm_var", metric) ~ "bias_variance",
      TRUE                          ~ "other"
    ),
    # Learner base method and meta-learner type for faceting
    base_method = case_when(
      grepl("lasso", learner)               ~ "lasso",
      grepl("boost", learner)               ~ "boost",
      grepl("kern",  learner)               ~ "kernel",
      learner == "ols_inter"                ~ "ols",
      learner == "zero_pred"                ~ "baseline",
      TRUE                                  ~ "other"
    ),
    meta_type = case_when(
      grepl("^r", learner) & learner != "rkern" ~ "R-learner",
      grepl("^r", learner)                      ~ "R-learner",
      grepl("^s", learner)                      ~ "S-learner",
      grepl("^t", learner)                      ~ "T-learner",
      grepl("^x", learner)                      ~ "X-learner",
      learner == "ols_inter"                    ~ "OLS",
      learner == "zero_pred"                    ~ "Baseline",
      TRUE                                      ~ "other"
    )
  ) %>%
  # Round values for cleaner output
  mutate(value = round(value, 6))

# Fix meta_type for r-learners (the regex above catches rkern correctly but
# let's be explicit)
df_long <- df_long %>%
  mutate(
    meta_type = case_when(
      startsWith(learner, "r") & learner != "zero_pred" ~ "R-learner",
      startsWith(learner, "s")                          ~ "S-learner",
      startsWith(learner, "t")                          ~ "T-learner",
      startsWith(learner, "x")                          ~ "X-learner",
      learner == "ols_inter"                            ~ "OLS",
      learner == "zero_pred"                            ~ "Baseline",
      TRUE                                              ~ meta_type
    )
  )

# ============================================================
# EXPORT METRICS CSV
# ============================================================

write.csv(df_long, OUTPUT_LONG, row.names = FALSE)
cat("Metrics written to:", OUTPUT_LONG, "\n")
cat("Rows:", nrow(df_long), "| Learners:", n_distinct(df_long$learner),
    "| Metrics:", n_distinct(df_long$metric), "\n\n")

# ============================================================
# DGP SUMMARY — simulation-level diagnostics
# ============================================================

cat("Computing DGP summary...\n")

sigma <- 5.0   # noise SD — must match your simulation config

dgp_summary <- data.frame(
  
  # Signal-to-noise
  tau_sd_mean    = mean(sapply(valid, function(it) s(it$tau_sd))),
  tau_sd_sd      = sd( sapply(valid, function(it) s(it$tau_sd))),
  sigma          = sigma,
  snr_mean       = mean(sapply(valid, function(it) s(it$tau_sd))) / sigma,
  
  # True ATE distribution
  true_ate_mean  = mean(sapply(valid, function(it) s(it$true_ate_test))),
  true_ate_sd    = sd(  sapply(valid, function(it) s(it$true_ate_test))),
  pct_ate_lt_0   = mean(sapply(valid, function(it) s(it$true_ate_test)) < 0),
  pct_ate_abs_lt1 = mean(abs(sapply(valid, function(it) s(it$true_ate_test))) < 1.0),
  
  # True subgroup effect magnitudes
  sg1_true_mean  = mean(sapply(valid, function(it) s(it$true_sg1))),
  sg1_true_sd    = sd(  sapply(valid, function(it) s(it$true_sg1))),
  sg1_true_min   = min( sapply(valid, function(it) s(it$true_sg1))),
  
  sg2_true_mean  = mean(sapply(valid, function(it) s(it$true_sg2))),
  sg2_true_sd    = sd(  sapply(valid, function(it) s(it$true_sg2))),
  sg2_true_min   = min( sapply(valid, function(it) s(it$true_sg2))),
  
  # Subgroup sizes
  sg1_size_mean  = mean(sapply(valid, function(it) s(it$sg_sizes$sg1))),
  sg1_size_min   = min( sapply(valid, function(it) s(it$sg_sizes$sg1))),
  sg2_size_mean  = mean(sapply(valid, function(it) s(it$sg_sizes$sg2))),
  sg2_size_min   = min( sapply(valid, function(it) s(it$sg_sizes$sg2))),
  
  # Pct beneficial
  pct_beneficial_mean = mean(sapply(valid, function(it) s(it$pct_beneficial))),
  pct_beneficial_sd   = sd(  sapply(valid, function(it) s(it$pct_beneficial))),
  
  n_valid   = n_valid,
  n_skipped = length(all_iters) - n_valid
)

# ============================================================
# DGP SECTION — PARK RAW DGP VARIABLES HERE
# ============================================================
# If you want to compute additional DGP metrics that require
# the raw simulation variables (propensity scores, scaled
# covariates, individual-level tau values), park them here.
#
# Example structure expected:
#
#   propensity_all  — numeric vector of length n, propensity scores
#   tau_all         — numeric vector of length n, true tau per patient
#   x1_s, x2_s, ... — scaled covariate vectors
set.seed(1)

# ----------------------------------------------------------
# DATA GENERATION
# ----------------------------------------------------------

cor_matrix = matrix(c(
  1.00,  0.05, -0.10,  0.35,  0.25,
  0.05,  1.00,  0.05,  0.00, -0.05,
  -0.10,  0.05,  1.00, -0.20, -0.40,
  0.35,  0.00, -0.20,  1.00,  0.45,
  0.25, -0.05, -0.40,  0.45,  1.00
), nrow = 5, byrow = TRUE)

z = MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

x1_bp       = 110 + 80 * pnorm(z[,1])
x2_travel   = 20  + 60 * pnorm(z[,2])
x3_digital  = round(20 * pnorm(z[,3]))
x4_comorbid = round(5  * pnorm(z[,4]))
x5_age      = round(40 + 40 * pnorm(z[,5]))

x = cbind(x1_bp, x2_travel, x3_digital, x4_comorbid, x5_age)
colnames(x) = c("bp_baseline", "travel_time", "digital_prior",
                "comorbidities", "age")

# Scaled covariates for propensity and tau
x1_s = as.numeric(scale(x1_bp))
x2_s = as.numeric(scale(x2_travel))
x3_s = as.numeric(scale(x3_digital))
x4_s = as.numeric(scale(x4_comorbid))
x5_s = as.numeric(scale(x5_age))

# Propensity
log_odds_w = -0.5 + 1.5*x2_s + 0.5*x1_s - 0.4*x5_s
propensity_all = pmax(0.05, pmin(plogis(log_odds_w), 0.95))
w          = rbinom(n, 1, propensity)

# True treatment effect
tau_all = -2.0*x2_s + 2.0*x1_s + 1.5*x4_s - 0.8*x3_s + 0.5*x5_s - 0.7

# Baseline and outcome
b_x = 0.6*x1_bp + 2.5*x4_comorbid + 0.3*x5_age - 1.2*x3_digital + 20
y   = pmax(b_x + (w - propensity)*tau_x + rnorm(n, 0, sigma), 80)
#
# Then uncomment and run:
#
  dgp_summary$prop_tau_corr <- cor(propensity_all, tau_all)
  dgp_summary$overlap_min   <- min(propensity_all)
  dgp_summary$overlap_max   <- max(propensity_all)
  dgp_summary$pct_near_boundary <- mean(
    abs(propensity_all - 0.5) < 0.1
  )
# ============================================================

write.csv(dgp_summary, OUTPUT_DGP, row.names = FALSE)
cat("DGP summary written to:", OUTPUT_DGP, "\n\n")

# ============================================================
# CONSOLE SUMMARY — quick sanity check
# ============================================================

cat("=== QUICK SANITY CHECK ===\n\n")

key_metrics <- df_long %>%
  filter(metric %in% c("mse_median","nmse_median","pct_beats_zero",
                       "rank_corr_mean","sign_both_rate",
                       "rec_sg1_p50","rec_sg2_p50",
                       "ate_bias_mean")) %>%
  select(learner, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  arrange(mse_median)

print(as.data.frame(key_metrics), digits = 3)

cat("\n=== DGP SUMMARY ===\n")
cat(sprintf("SNR (tau_sd / sigma):     %.3f\n", dgp_summary$snr_mean))
cat(sprintf("True ATE mean:            %.3f\n", dgp_summary$true_ate_mean))
cat(sprintf("Pct |ATE| < 1 mmHg:       %.1f%%\n", dgp_summary$pct_ate_abs_lt1 * 100))
cat(sprintf("SG1 true tau mean:        %.3f mmHg\n", dgp_summary$sg1_true_mean))
cat(sprintf("SG2 true tau mean:        %.3f mmHg\n", dgp_summary$sg2_true_mean))
cat(sprintf("SG1 mean size (test set): %.1f\n",  dgp_summary$sg1_size_mean))
cat(sprintf("SG2 mean size (test set): %.1f\n",  dgp_summary$sg2_size_mean))
cat(sprintf("Valid / Skipped:          %d / %d\n", dgp_summary$n_valid,
            dgp_summary$n_skipped))

cat("\nDone.\n")