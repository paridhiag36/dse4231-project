# ============================================================
# SMS PREVENTIVE HEALTH CAMPAIGN — SINGLE ITERATION
# Outcome: Number of GP visits in next 12 months (floored at 0)
# Treatment: SMS reminder to attend preventive check-up (W=1) vs none (W=0)
# Key feature: tau(X) = 0.01 for everyone — genuine null effect
# Purpose: Test whether learners hallucinate heterogeneity when none exists
# ============================================================

library(MASS)      # for mvrnorm — correlated covariate generation
library(rlearner)  # for rlasso and rboost
install.packages("devtools")
devtools::install_github("xnie/KRLS")
library(KRLS2)

set.seed(1)
n = 200

# ============================================================
# SECTION 1: COVARIATE GENERATION
#
# X1: age (years)                          — range 30 to 75
# X2: GP visits in past 24 months (count)  — range 0 to 12
# X3: medication adherence ratio           — range 0 to 1 (continuous)
# X4: travel time to clinic (minutes)      — range 5 to 60
# X5: area deprivation index               — standardised, higher = more deprived in US
#
# Correlation structure:
#   age <-> past GP visits:       +0.35  older patients have more established care
#   age <-> medication adherence: +0.25  older patients on regular prescriptions
#   past visits <-> adherence:    +0.40  engaged patients visit and collect meds
#   deprivation <-> adherence:    -0.30  deprived patients face more barriers
#   deprivation <-> past visits:  -0.20  deprived patients attend less
#   travel time is largely independent of clinical/demographic variables
#   age <-> deprivation:          -0.15  working-age older adults tend less deprived
# ============================================================

cor_matrix = matrix(c(
  # X1_age  X2_visits  X3_adherence  X4_travel  X5_deprivation
   1.00,     0.35,       0.25,        -0.05,      -0.15,   # X1 age
   0.35,     1.00,       0.40,        -0.05,      -0.20,   # X2 past GP visits
   0.25,     0.40,       1.00,        -0.05,      -0.30,   # X3 med adherence
  -0.05,    -0.05,      -0.05,         1.00,       0.05,   # X4 travel time
  -0.15,    -0.20,      -0.30,         0.05,       1.00    # X5 deprivation
), nrow = 5, byrow = TRUE)

# Generate correlated standard normal data
z = mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

# Transform each variable to its realistic scale
# pnorm maps standard normal to (0,1) then scaled to target range

# X1: age — 30 to 75 years, rounded to integer
x1_age       = round(30 + 45 * pnorm(z[,1]))

# X2: GP visits in past 24 months — 0 to 12, rounded to integer
# Mean around 4-5 is realistic for a general adult population
x2_visits    = round(12 * pnorm(z[,2]))

# X3: medication adherence ratio — 0 to 1, continuous
# Kept continuous — proportion of days with active prescription dispensed
x3_adherence = pnorm(z[,3])

# X4: travel time to clinic — 5 to 60 minutes
# Urban and peri-urban population, most patients reasonably close
x4_travel    = 5 + 55 * pnorm(z[,4])

# X5: area deprivation index — standardised scale
# Higher values = more deprived
# Kept as standardised score for interpretability
x5_deprivation = as.numeric(scale(z[,5]))

# Combine into matrix for rlearner
x = cbind(x1_age, x2_visits, x3_adherence, x4_travel, x5_deprivation)
colnames(x) = c("age", "past_visits", "med_adherence",
                "travel_time", "deprivation")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))

cat("\n=== COVARIATE CORRELATION (should reflect design above) ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
#
# Clinical logic:
#   Age (X1) is the dominant driver — campaign targets older adults
#   Medication adherence (X3) contributes positively — less adherent patients
#     are more likely to receive reminders as campaign targets disengaged
#     NOTE: lower adherence = higher propensity, so coefficient is negative
#   Travel time, deprivation, and past visits NOT in propensity model
#     because the SMS targeting algorithm uses only age and clinical flags
#
# Expected enrolment: ~55-60% reflecting a broad population campaign
# Intercept 0.0 gives baseline probability of 50% before covariate effects
# ============================================================

# Scale covariates for propensity model
x1_s = scale(x1_age)
x2_s = scale(x2_visits)
x3_s = scale(x3_adherence)
x4_s = scale(x4_travel)
x5_s = scale(x5_deprivation)

# Propensity: older patients and LESS adherent patients more likely targeted
log_odds_w = 0.0 +
  0.60 * as.numeric(x1_s) +   # age — dominant driver
  -0.40 * as.numeric(x3_s) +   # adherence — less adherent targeted
  0.20 * as.numeric(x2_s) +   # past visits — engaged patients
  -0.15 * as.numeric(x5_s) +   # deprivation — mild negative
  -0.10 * as.numeric(x4_s)     # travel — mild negative

propensity = plogis(log_odds_w)

# Clip to avoid exact 0 or 1
propensity = pmax(0.05, pmin(propensity, 0.95))

w = rbinom(n, 1, propensity)

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion receiving SMS reminder:", round(mean(w), 3), "\n")
cat("Propensity range:",
    round(min(propensity), 3), "to", round(max(propensity), 3), "\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")

# Confounding check — treated group should be older and less adherent
cat("\nMean age         — treated:", round(mean(x1_age[w==1]), 1),
    "| control:", round(mean(x1_age[w==0]), 1), "\n")
cat("Mean adherence   — treated:", round(mean(x3_adherence[w==1]), 3),
    "| control:", round(mean(x3_adherence[w==0]), 3), "\n")
cat("Mean past visits — treated:", round(mean(x2_visits[w==1]), 1),
    "| control:", round(mean(x2_visits[w==0]), 1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)
#
# tau*(X) = 0.01 for ALL individuals — no heterogeneity whatsoever
# This represents a negligible universal nudge effect
# A meaningful reminder campaign would shift visits by 0.3 to 0.5 per year
# 0.01 is one fiftieth of that — clinically and practically meaningless
#
# Any variation in tau_hat(X) across individuals is BY DEFINITION spurious
# ============================================================

tau_x = rep(0.01, n)

cat("\n=== TRUE TREATMENT EFFECT ===\n")
cat("tau*(X) = 0.01 for all individuals (constant, no heterogeneity)\n")
cat("Mean:", round(mean(tau_x), 4), "\n")
cat("SD:  ", round(sd(tau_x), 4), "(should be exactly 0)\n")
cat("Range:", round(min(tau_x), 4), "to", round(max(tau_x), 4), "\n")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
#
# b(X) = expected GP visits in next 12 months under NO reminder
# Coefficients deliberately small to avoid overwhelming tau = 0.01
# Past visits (X2) is dominant predictor — behavioural consistency
# Adherence (X3) contributes positively — engaged patients visit more
# Travel time (X4) contributes negatively — distance reduces attendance
# Deprivation (X5) contributes negatively — structural access barriers
# Age (X1) contributes modestly and positively — clinical need
#
# Intercept 3.0 anchors mean at ~3 visits per year
# Noise SD = 1.0 dominates residual variation by design
# ============================================================

b_x = 3.0 +
  0.15 * as.numeric(x2_s) +   # halved from 0.30
  0.10 * as.numeric(x3_s) +   # halved from 0.20
  -0.08 * as.numeric(x4_s) +   # halved from 0.15
  -0.08 * as.numeric(x5_s) +   # halved from 0.15
  0.05 * as.numeric(x1_s)     # halved from 0.10

cat("\n=== BASELINE OUTCOME b(X) SUMMARY ===\n")
cat("Mean b(X):", round(mean(b_x), 3), "visits\n")
cat("SD b(X):  ", round(sd(b_x), 3), "visits\n")
cat("Range b(X):", round(min(b_x), 3),
    "to", round(max(b_x), 3), "visits\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
#
# Y = b(X) + (W - propensity) * tau(X) + epsilon
# Floored at 0 — cannot have negative GP visits
# Noise SD = 1.0 chosen so residual variation dominates tau = 0.01
# Signal-to-noise ratio is deliberately very low — this is a null study
# ============================================================

sigma   = 1.0
epsilon = rnorm(n, mean = 0, sd = sigma)

y_raw = b_x + (w - propensity) * tau_x + epsilon
y     = pmax(y_raw, 0)   # floor at zero

# Proportion floored — should be small
cat("\n=== OBSERVED OUTCOME Y ===\n")
cat("Proportion floored at zero:", round(mean(y_raw < 0), 3), "\n")
cat("Mean Y (all):      ", round(mean(y), 3), "\n")
cat("SD Y:              ", round(sd(y), 3), "\n")
cat("Mean Y (treated):  ", round(mean(y[w==1]), 3), "\n")
cat("Mean Y (control):  ", round(mean(y[w==0]), 3), "\n")
cat("Naive ATE (treated - control):",
    round(mean(y[w==1]) - mean(y[w==0]), 4),
    "(confounded — not the true ATE)\n")
cat("True ATE:", round(mean(tau_x), 4), "\n")

# ============================================================
# SECTION 6: DEFINE SUBGROUPS BEFORE FITTING
#
# All subgroups have the same true tau = 0.01
# Any learner assigning different tau_hat across groups is producing
# false heterogeneity by misattributing baseline differences
#
# Subgroup 1 — High health engagers
#   Above median past visits AND above median medication adherence
#   High baseline visit rate — learner may falsely assign higher tau
#
# Subgroup 2 — Remote low engagers
#   Above median travel time AND below median past visits
#   Low baseline — learner may falsely assign lower or negative tau
#
# Subgroup 3 — Older adherent patients (most heavily targeted)
#   Above 75th percentile age AND above median medication adherence
#   High propensity — learner may conflate propensity with treatment effect
# ============================================================

sg1 = x2_visits    >  median(x2_visits)    &
      x3_adherence >  median(x3_adherence)

sg2 = x4_travel    >  median(x4_travel)    &
      x2_visits    <  median(x2_visits)

sg3 = x1_age       >  quantile(x1_age, 0.75) &
      x3_adherence >  median(x3_adherence)

cat("\n=== SUBGROUP SIZES ===\n")
cat("SG1 (high engagers):          ", sum(sg1), "patients\n")
cat("SG2 (remote low engagers):    ", sum(sg2), "patients\n")
cat("SG3 (older adherent):         ", sum(sg3), "patients\n")

cat("\n=== TRUE TAU BY SUBGROUP (all should be 0.01) ===\n")
cat("SG1:", round(mean(tau_x[sg1]), 4), "\n")
cat("SG2:", round(mean(tau_x[sg2]), 4), "\n")
cat("SG3:", round(mean(tau_x[sg3]), 4), "\n")

# ============================================================
# SECTION 7: FIT LEARNERS
# ============================================================

cat("\n=== FITTING LEARNERS ===\n")

cat("Fitting rlasso...\n")
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

cat("Fitting rboost...\n")
rboost_fit = rboost(x, w, y,
                    num_search_rounds     = 5,
                    k_folds               = 5,
                    ntrees_max            = 300,
                    early_stopping_rounds = 5,
                    verbose               = FALSE)
rboost_est = predict(rboost_fit, x)

# --- Baseline: constant 0.01 predictor ---
# Theoretically optimal predictor given knowledge that tau is constant
# Any learner scoring worse on spread metrics is doing worse than
# simply knowing the truth is flat
const_pred = rep(0.01, n)

# --- Baseline: zero predictor ---
# Included for consistency with telemedicine study
zero_pred  = rep(0, n)

# --- Baseline: OLS with interactions ---
x_df = as.data.frame(x)
colnames(x_df) = c("age", "visits", "adherence", "travel", "deprivation")

ols_data = data.frame(
  y            = y,
  w            = w,
  x_df,
  w_age        = w * x_df$age,
  w_visits     = w * x_df$visits,
  w_adherence  = w * x_df$adherence,
  w_travel     = w * x_df$travel,
  w_deprivation= w * x_df$deprivation
)

ols_fit = lm(y ~ age + visits + adherence + travel + deprivation +
               w + w_age + w_visits + w_adherence +
               w_travel + w_deprivation,
             data = ols_data)

coef_w             = coef(ols_fit)["w"]
coef_w_age         = coef(ols_fit)["w_age"]
coef_w_visits      = coef(ols_fit)["w_visits"]
coef_w_adherence   = coef(ols_fit)["w_adherence"]
coef_w_travel      = coef(ols_fit)["w_travel"]
coef_w_deprivation = coef(ols_fit)["w_deprivation"]

ols_tau_est = coef_w +
              coef_w_age         * x_df$age +
              coef_w_visits      * x_df$visits +
              coef_w_adherence   * x_df$adherence +
              coef_w_travel      * x_df$travel +
              coef_w_deprivation * x_df$deprivation

learners_all = list(
  rlasso     = rlasso_est,
  rboost     = rboost_est,
  ols_inter  = as.numeric(ols_tau_est),
  const_pred = const_pred,
  zero_pred  = zero_pred
)

# ============================================================
# SECTION 8: EVALUATION
# Primary focus: calibration and suppression of false heterogeneity
# ============================================================

true_tau   = 0.01
tau_variance = var(tau_x)   # = 0 by construction, use for MSE reference

cat("\n===========================================\n")
cat("MEASURE 1: NORMALISED MSE\n")
cat("(denominator = variance of tau_hat for meaningful comparison\n")
cat(" since true tau variance = 0)\n")
cat("===========================================\n")

for (name in names(learners_all)) {
  est     = learners_all[[name]]
  raw_mse = mean((est - tau_x)^2)

  # Since true tau has zero variance, normalise by variance of estimates
  # to give a sense of how far estimates stray from the true constant
  # A learner predicting exactly 0.01 everywhere gets MSE = 0
  cat("Learner:", name, "\n")
  cat("  Raw MSE from true tau:  ", round(raw_mse, 6), "\n")
  cat("  Mean tau_hat:           ", round(mean(est), 4), "\n")
  cat("  Bias (mean - true):     ", round(mean(est) - true_tau, 4), "\n\n")
}

cat("===========================================\n")
cat("MEASURE 2: SPREAD OF tau_hat(X) — PRIMARY CALIBRATION METRIC\n")
cat("True tau is constant 0.01 — all spread is spurious\n")
cat("Lower spread = better calibration\n")
cat("===========================================\n")

# Calibration pass threshold = 0.20 for single iteration at n=200
# Ten times the true effect would be 0.10 — too tight for n=200 noise
# 0.20 threshold used here; tighter threshold for multi-iteration study
pass_threshold = 0.20

spread_table = data.frame()

for (name in names(learners_all)) {
  est = learners_all[[name]]

  iqr_est  = IQR(est)
  p05_est  = quantile(est, 0.05)
  p95_est  = quantile(est, 0.95)
  range_90 = p95_est - p05_est
  sd_est   = sd(est)
  passes   = iqr_est < pass_threshold

  cat("Learner:", name, "\n")
  cat("  IQR of tau_hat:           ", round(iqr_est, 4), "\n")
  cat("  5th percentile:           ", round(p05_est, 4), "\n")
  cat("  95th percentile:          ", round(p95_est, 4), "\n")
  cat("  90% range (P95 - P05):    ", round(range_90, 4), "\n")
  cat("  SD of tau_hat:            ", round(sd_est,   4), "\n")
  cat("  Calibration pass (IQR<0.20):", passes, "\n\n")

  spread_table = rbind(spread_table, data.frame(
    Learner  = name,
    IQR      = round(iqr_est,  4),
    P05      = round(p05_est,  4),
    P95      = round(p95_est,  4),
    Range_90 = round(range_90, 4),
    SD       = round(sd_est,   4),
    Passes   = passes
  ))
}

cat("===========================================\n")
cat("MEASURE 3: SUBGROUP PLACEBO CHECK\n")
cat("True tau = 0.01 in ALL subgroups\n")
cat("Deviation > 0.10 flagged as meaningful false positive\n")
cat("===========================================\n")

false_pos_threshold = 0.10

cat("True tau in all subgroups: 0.01\n\n")

for (name in names(learners_all)) {
  est = learners_all[[name]]

  est_all = mean(est)
  est_sg1 = mean(est[sg1])
  est_sg2 = mean(est[sg2])
  est_sg3 = mean(est[sg3])

  dev_all = abs(est_all - true_tau)
  dev_sg1 = abs(est_sg1 - true_tau)
  dev_sg2 = abs(est_sg2 - true_tau)
  dev_sg3 = abs(est_sg3 - true_tau)

  fp_sg1  = dev_sg1 > false_pos_threshold
  fp_sg2  = dev_sg2 > false_pos_threshold
  fp_sg3  = dev_sg3 > false_pos_threshold

  cat("Learner:", name, "\n")
  cat("  Overall mean tau_hat:", round(est_all, 4),
      "| deviation:", round(dev_all, 4), "\n")
  cat("  SG1 (high engagers):   ", round(est_sg1, 4),
      "| deviation:", round(dev_sg1, 4),
      "| false positive:", fp_sg1, "\n")
  cat("  SG2 (remote low eng):  ", round(est_sg2, 4),
      "| deviation:", round(dev_sg2, 4),
      "| false positive:", fp_sg2, "\n")
  cat("  SG3 (older adherent):  ", round(est_sg3, 4),
      "| deviation:", round(dev_sg3, 4),
      "| false positive:", fp_sg3, "\n\n")
}

# ============================================================
# SECTION 9: VISUALISATIONS
# Primary: KDE distribution of tau_hat(X) across learners
# Secondary: PDP plots — should be flat for all covariates
# ============================================================

# --- KDE Plot: distribution of tau_hat across all individuals ---
# Ideal shape: tight spike centred at 0.01
# Wide spread or multi-modal distribution = spurious heterogeneity

cat("=== GENERATING KDE PLOT ===\n")

# Define colours for each learner — consistent with telemedicine study
learner_colours = c(
  rlasso     = "blue",
  rboost     = "red",
  ols_inter  = "darkgreen",
  const_pred = "black",
  zero_pred  = "grey50"
)

learner_lty = c(
  rlasso     = 1,
  rboost     = 2,
  ols_inter  = 3,
  const_pred = 4,
  zero_pred  = 5
)

# Compute KDE for each learner
all_estimates = unlist(learners_all)
x_range = range(all_estimates)
x_grid  = seq(x_range[1] - 0.1, x_range[2] + 0.1, length.out = 500)

kde_list = lapply(learners_all, function(est) {
  if (sd(est) < 1e-10) return(NULL)   # skip constant predictors
  density(est, from = x_grid[1], to = x_grid[length(x_grid)], n = 500)
})

# Find y-axis range across all non-null KDEs
y_max = max(sapply(kde_list[!sapply(kde_list, is.null)],
                   function(d) max(d$y))) * 1.1

par(mfrow = c(1, 1))
plot(NULL,
     xlim = c(x_range[1] - 0.05, x_range[2] + 0.05),
     ylim = c(0, y_max),
     xlab = "Estimated tau(X) — additional GP visits per year",
     ylab = "Density",
     main = "KDE of tau_hat(X) Estimates\nIdeal: tight spike at 0.01")

# Vertical line at true tau = 0.01
abline(v = 0.01,  col = "black", lwd = 2, lty = 1)
abline(v = 0,     col = "grey70", lwd = 1, lty = 2)

# Draw KDE for each learner that has variation
for (name in names(kde_list)) {
  if (is.null(kde_list[[name]])) next
  lines(kde_list[[name]]$x,
        kde_list[[name]]$y,
        col = learner_colours[name],
        lwd = 2,
        lty = learner_lty[name])
}

legend("topright",
       legend = c("True tau = 0.01",
                  "rlasso", "rboost", "OLS interactions"),
       col    = c("black", "blue", "red", "darkgreen"),
       lwd    = 2,
       lty    = c(1, 1, 2, 3),
       cex    = 0.8)

# --- PDP Plots: should be flat at 0.01 for all covariates ---
# Any slope = false positive — learner is treating covariate as
# a treatment effect moderator when it is only a baseline predictor

cat("=== GENERATING PDP PLOTS ===\n")

covariate_names  = c("age", "past_visits", "med_adherence",
                     "travel_time", "deprivation")
covariate_labels = c("Age (years)",
                     "Past GP Visits (24 months)",
                     "Medication Adherence Ratio",
                     "Travel Time (minutes)",
                     "Area Deprivation Index")
covariate_vals   = list(x1_age, x2_visits, x3_adherence,
                        x4_travel, x5_deprivation)

par(mfrow = c(2, 3))

for (j in 1:5) {

  cov_vals = covariate_vals[[j]]
  ord      = order(cov_vals)

  # Y-axis range includes all learner estimates for this covariate ordering
  y_range = range(c(tau_x,
                    rlasso_est,
                    rboost_est,
                    as.numeric(ols_tau_est)))

  plot(NULL,
       xlim = range(cov_vals),
       ylim = y_range,
       xlab = covariate_labels[j],
       ylab = "Treatment effect (visits/year)",
       main = paste("PDP:", covariate_labels[j]))

  # True tau — flat horizontal line at 0.01
  abline(h = 0.01, col = "black", lwd = 2, lty = 1)
  abline(h = 0,    col = "grey70", lwd = 1, lty = 2)

  # Learner estimates sorted by covariate value
  lines(cov_vals[ord], rlasso_est[ord],
        col = "blue",      lwd = 2, lty = 2)
  lines(cov_vals[ord], rboost_est[ord],
        col = "red",       lwd = 2, lty = 3)
  lines(cov_vals[ord], as.numeric(ols_tau_est)[ord],
        col = "darkgreen", lwd = 2, lty = 4)

  legend("topleft",
         legend = c("True tau = 0.01", "rlasso", "rboost", "OLS"),
         col    = c("black", "blue", "red", "darkgreen"),
         lwd    = 2,
         lty    = c(1, 2, 3, 4),
         cex    = 0.65,
         bg     = "white")
}

# Scatter plot: true tau vs estimated tau
# All points should cluster horizontally at y = 0.01
# Any vertical spread = false heterogeneity
plot(NULL,
     xlim = c(0, 0.02),
     ylim = range(c(rlasso_est, rboost_est)),
     xlab = "True tau(X) = 0.01 for all",
     ylab = "Estimated tau_hat(X)",
     main = "True vs Estimated tau\nAll points should be at x = 0.01")

abline(h  = 0.01, col = "black",  lwd = 2, lty = 1)
abline(h  = 0,    col = "grey70", lwd = 1, lty = 2)

points(jitter(tau_x, factor = 0.5), rlasso_est,
       pch = 16, col = rgb(0, 0, 1, 0.4), cex = 0.8)
points(jitter(tau_x, factor = 0.5), rboost_est,
       pch = 17, col = rgb(1, 0, 0, 0.4), cex = 0.8)

legend("topleft",
       legend = c("rlasso", "rboost"),
       col    = c("blue", "red"),
       pch    = c(16, 17),
       cex    = 0.8)

par(mfrow = c(1, 1))

# ============================================================
# SECTION 10: FULL SUMMARY TABLE
# ============================================================

cat("\n===========================================\n")
cat("FULL SUMMARY TABLE\n")
cat("===========================================\n")

summary_rows = list()

for (name in names(learners_all)) {
  est = learners_all[[name]]

  raw_mse  = mean((est - tau_x)^2)
  bias     = mean(est) - true_tau
  iqr_est  = IQR(est)
  range_90 = quantile(est, 0.95) - quantile(est, 0.05)
  passes   = iqr_est < pass_threshold

  est_sg1  = mean(est[sg1])
  est_sg2  = mean(est[sg2])
  est_sg3  = mean(est[sg3])

  fp_sg1   = abs(est_sg1 - true_tau) > false_pos_threshold
  fp_sg2   = abs(est_sg2 - true_tau) > false_pos_threshold
  fp_sg3   = abs(est_sg3 - true_tau) > false_pos_threshold

  summary_rows[[name]] = data.frame(
    Learner   = name,
    Raw_MSE   = round(raw_mse,  6),
    Bias      = round(bias,     4),
    IQR       = round(iqr_est,  4),
    Range_90  = round(range_90, 4),
    Pass_Cal  = passes,
    FP_SG1    = fp_sg1,
    FP_SG2    = fp_sg2,
    FP_SG3    = fp_sg3
  )
}

summary_table = do.call(rbind, summary_rows)
print(summary_table, row.names = FALSE)

cat("\nInterpretation guide:\n")
cat("  Raw_MSE:  distance of mean estimate from true tau = 0.01\n")
cat("  Bias:     mean tau_hat minus 0.01 — should be near zero\n")
cat("  IQR:      spread of individual estimates — should be near zero\n")
cat("  Range_90: P95 minus P05 — captures tail spread\n")
cat("  Pass_Cal: TRUE if IQR < 0.20 (calibration pass at n=200)\n")
cat("  FP_SG:    TRUE if subgroup mean deviates > 0.10 from true tau\n")
cat("\nKey comparison with telemedicine study:\n")
cat("  Telemedicine tests POWER — can learner find real heterogeneity?\n")
cat("  This study tests CALIBRATION — does learner stay quiet when\n")
cat("  there is no heterogeneity to find?\n")