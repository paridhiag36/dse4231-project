# ============================================================
# TELEMEDICINE DGP DIAGNOSTICS — 200 ITERATIONS
# Extracts distributional properties of the data generating
# process: tau heterogeneity, treatment split, ATT, ATC,
# propensity-tau correlation, SNR, and overlap
# ============================================================

library(MASS)

set.seed(42)

n       <- 300
n_iter  <- 200
sigma   <- 5.0

cor_matrix <- matrix(c(
  1.00,  0.05, -0.10,  0.35,  0.25,
  0.05,  1.00,  0.05,  0.00, -0.05,
  -0.10,  0.05,  1.00, -0.20, -0.40,
  0.35,  0.00, -0.20,  1.00,  0.45,
  0.25, -0.05, -0.40,  0.45,  1.00
), nrow = 5, byrow = TRUE)

# Storage — one row per iteration
results <- data.frame(
  iter              = integer(n_iter),
  seed              = integer(n_iter),
  
  # Tau distribution (population level, all n patients)
  tau_mean          = numeric(n_iter),   # true ATE
  tau_sd            = numeric(n_iter),   # heterogeneity spread
  tau_p5            = numeric(n_iter),   # 5th percentile of tau
  tau_p25           = numeric(n_iter),
  tau_p50           = numeric(n_iter),
  tau_p75           = numeric(n_iter),
  tau_p95           = numeric(n_iter),   # 95th percentile of tau
  tau_min           = numeric(n_iter),
  tau_max           = numeric(n_iter),
  pct_beneficial    = numeric(n_iter),   # proportion with tau < 0
  
  # Treatment assignment
  n_treated         = integer(n_iter),
  n_control         = integer(n_iter),
  pct_treated       = numeric(n_iter),
  propensity_mean   = numeric(n_iter),
  propensity_sd     = numeric(n_iter),
  propensity_p5     = numeric(n_iter),
  propensity_p95    = numeric(n_iter),
  
  # ATT and ATC — the key estimands
  ATT               = numeric(n_iter),   # E[tau | W=1]
  ATC               = numeric(n_iter),   # E[tau | W=0]
  ATT_ATC_gap       = numeric(n_iter),   # ATT - ATC
  
  # Confounding diagnostics
  prop_tau_corr     = numeric(n_iter),   # cor(e*(X), tau*(X))
  
  # Signal to noise
  snr               = numeric(n_iter),   # tau_sd / sigma
  
  # Overlap
  overlap_min       = numeric(n_iter),
  overlap_max       = numeric(n_iter),
  pct_near_boundary = numeric(n_iter)    # |e - 0.5| < 0.1
)

cat("Running", n_iter, "DGP iterations...\n")

for (i in seq_len(n_iter)) {
  
  iter_seed <- i
  set.seed(iter_seed)
  
  # ----------------------------------------------------------
  # DATA GENERATION — identical to simulation script
  # ----------------------------------------------------------
  
  z <- MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)
  
  x1_bp       <- 110 + 80 * pnorm(z[, 1])
  x2_travel   <- 20  + 60 * pnorm(z[, 2])
  x3_digital  <- round(20 * pnorm(z[, 3]))
  x4_comorbid <- round(5  * pnorm(z[, 4]))
  x5_age      <- round(40 + 40 * pnorm(z[, 5]))
  
  x1_s <- as.numeric(scale(x1_bp))
  x2_s <- as.numeric(scale(x2_travel))
  x3_s <- as.numeric(scale(x3_digital))
  x4_s <- as.numeric(scale(x4_comorbid))
  x5_s <- as.numeric(scale(x5_age))
  
  log_odds_w    <- -0.5 + 1.5*x2_s + 0.5*x1_s - 0.4*x5_s
  propensity    <- pmax(0.05, pmin(plogis(log_odds_w), 0.95))
  w             <- rbinom(n, 1, propensity)
  
  tau_x <- -2.0*x2_s + 2.0*x1_s + 1.5*x4_s - 0.8*x3_s + 0.5*x5_s - 0.7
  
  # ----------------------------------------------------------
  # TAU DISTRIBUTION
  # ----------------------------------------------------------
  
  results$iter[i]           <- i
  results$seed[i]           <- iter_seed
  results$tau_mean[i]       <- mean(tau_x)
  results$tau_sd[i]         <- sd(tau_x)
  results$tau_p5[i]         <- quantile(tau_x, 0.05)
  results$tau_p25[i]        <- quantile(tau_x, 0.25)
  results$tau_p50[i]        <- quantile(tau_x, 0.50)
  results$tau_p75[i]        <- quantile(tau_x, 0.75)
  results$tau_p95[i]        <- quantile(tau_x, 0.95)
  results$tau_min[i]        <- min(tau_x)
  results$tau_max[i]        <- max(tau_x)
  results$pct_beneficial[i] <- mean(tau_x < 0)
  
  # ----------------------------------------------------------
  # TREATMENT ASSIGNMENT
  # ----------------------------------------------------------
  
  results$n_treated[i]         <- sum(w)
  results$n_control[i]         <- sum(1 - w)
  results$pct_treated[i]       <- mean(w)
  results$propensity_mean[i]   <- mean(propensity)
  results$propensity_sd[i]     <- sd(propensity)
  results$propensity_p5[i]     <- quantile(propensity, 0.05)
  results$propensity_p95[i]    <- quantile(propensity, 0.95)
  
  # ----------------------------------------------------------
  # ATT AND ATC
  # Based on realised treatment assignment W
  # ----------------------------------------------------------
  
  results$ATT[i]         <- mean(tau_x[w == 1])
  results$ATC[i]         <- mean(tau_x[w == 0])
  results$ATT_ATC_gap[i] <- results$ATT[i] - results$ATC[i]
  
  # ----------------------------------------------------------
  # CONFOUNDING AND OVERLAP
  # ----------------------------------------------------------
  
  results$prop_tau_corr[i]     <- cor(propensity, tau_x)
  results$snr[i]               <- sd(tau_x) / sigma
  results$overlap_min[i]       <- min(propensity)
  results$overlap_max[i]       <- max(propensity)
  results$pct_near_boundary[i] <- mean(abs(propensity - 0.5) < 0.1)
}

cat("Done.\n\n")

# ============================================================
# SUMMARY ACROSS ITERATIONS
# ============================================================

summarise_vec <- function(x, label) {
  cat(sprintf("%-22s mean=%6.3f  sd=%5.3f  p5=%6.3f  p95=%6.3f  min=%6.3f  max=%6.3f\n",
              label,
              mean(x, na.rm = TRUE),
              sd(x,   na.rm = TRUE),
              quantile(x, 0.05, na.rm = TRUE),
              quantile(x, 0.95, na.rm = TRUE),
              min(x, na.rm = TRUE),
              max(x, na.rm = TRUE)))
}

cat("=== TAU DISTRIBUTION ACROSS ITERATIONS ===\n")
summarise_vec(results$tau_mean,    "tau_mean (ATE)")
summarise_vec(results$tau_sd,      "tau_sd")
summarise_vec(results$tau_p5,      "tau_p5 (within-iter)")
summarise_vec(results$tau_p95,     "tau_p95 (within-iter)")
summarise_vec(results$tau_min,     "tau_min (within-iter)")
summarise_vec(results$tau_max,     "tau_max (within-iter)")
summarise_vec(results$pct_beneficial, "pct_beneficial")

cat("\n=== TREATMENT ASSIGNMENT ===\n")
summarise_vec(results$pct_treated,     "pct_treated")
summarise_vec(results$propensity_mean, "propensity_mean")
summarise_vec(results$propensity_sd,   "propensity_sd")
summarise_vec(results$propensity_p5,   "propensity_p5")
summarise_vec(results$propensity_p95,  "propensity_p95")

cat("\n=== ATT AND ATC ===\n")
summarise_vec(results$ATT,         "ATT")
summarise_vec(results$ATC,         "ATC")
summarise_vec(results$ATT_ATC_gap, "ATT - ATC gap")

cat("\n=== CONFOUNDING AND OVERLAP ===\n")
summarise_vec(results$prop_tau_corr,     "prop_tau_corr")
summarise_vec(results$snr,               "SNR (tau_sd/sigma)")
summarise_vec(results$pct_near_boundary, "pct_near_boundary")

# ============================================================
# EXPORT
# ============================================================

write.csv(results, "dgp_iter_diagnostics.csv", row.names = FALSE)
cat("\nPer-iteration results written to: dgp_iter_diagnostics.csv\n")