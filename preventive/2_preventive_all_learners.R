# ============================================================
# SMS PREVENTIVE HEALTH CAMPAIGN — SINGLE ITERATION
# Outcome: Number of GP visits in next 12 months (floored at 0)
# Treatment: SMS reminder to attend preventive check-up (W=1) vs none (W=0)
# Key feature: tau(X) = 0.01 for everyone — genuine null effect
# Purpose: Test whether learners hallucinate heterogeneity when none exists
#
# LEARNERS COMPARED:
#   R-learner:  rlasso, rboost, rkern
#   S-learner:  slasso, sboost, skern
#   T-learner:  tlasso, tboost, tkern
#   X-learner:  xlasso, xboost, xkern
#   Baselines:  constant 0.01 predictor, zero predictor, OLS with interactions
#
# TWO COMPARISON DIMENSIONS:
#   1. Across meta-learner types (R vs S vs T vs X) — same base method
#   2. Across base methods (lasso vs boost vs kern) — same meta-learner
# ============================================================

library(MASS)      # for mvrnorm — correlated covariate generation
library(rlearner)  # for all learners
library(KRLS2)     # for kernel ridge regression

set.seed(1)
n = 200

# ============================================================
# SECTION 1: COVARIATE GENERATION
#
# X1: age (years)                          — range 30 to 75
# X2: GP visits in past 24 months (count)  — range 0 to 12
# X3: medication adherence ratio           — range 0 to 1 (continuous)
# X4: travel time to clinic (minutes)      — range 5 to 60
# X5: area deprivation index               — standardised, higher = more deprived
#
# Correlation structure:
#   age <-> past GP visits:       +0.35
#   age <-> medication adherence: +0.25
#   past visits <-> adherence:    +0.40
#   deprivation <-> adherence:    -0.30
#   deprivation <-> past visits:  -0.20
#   travel time largely independent
#   age <-> deprivation:          -0.15
# ============================================================

cor_matrix = matrix(c(
  # X1_age  X2_visits  X3_adherence  X4_travel  X5_deprivation
  1.00,     0.35,       0.25,        -0.05,      -0.15,
  0.35,     1.00,       0.40,        -0.05,      -0.20,
  0.25,     0.40,       1.00,        -0.05,      -0.30,
  -0.05,    -0.05,      -0.05,         1.00,       0.05,
  -0.15,    -0.20,      -0.30,         0.05,       1.00
), nrow = 5, byrow = TRUE)

z = mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

x1_age         = round(30 + 45 * pnorm(z[,1]))
x2_visits      = round(12 * pnorm(z[,2]))
x3_adherence   = pnorm(z[,3])
x4_travel      = 5 + 55 * pnorm(z[,4])
x5_deprivation = as.numeric(scale(z[,5]))

x = cbind(x1_age, x2_visits, x3_adherence, x4_travel, x5_deprivation)
colnames(x) = c("age", "past_visits", "med_adherence",
                "travel_time", "deprivation")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))
cat("\n=== COVARIATE CORRELATION ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
# ============================================================

x1_s = scale(x1_age)
x2_s = scale(x2_visits)
x3_s = scale(x3_adherence)
x4_s = scale(x4_travel)
x5_s = scale(x5_deprivation)

log_odds_w = 0.0 +
  0.60 * as.numeric(x1_s) +
  -0.40 * as.numeric(x3_s) +
  0.20 * as.numeric(x2_s) +
  -0.15 * as.numeric(x5_s) +
  -0.10 * as.numeric(x4_s)

propensity = plogis(log_odds_w)
propensity = pmax(0.05, pmin(propensity, 0.95))
w          = rbinom(n, 1, propensity)

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion receiving SMS:", round(mean(w), 3), "\n")
cat("Propensity range:",
    round(min(propensity), 3), "to", round(max(propensity), 3), "\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")

cat("\nMean age         — treated:", round(mean(x1_age[w==1]), 1),
    "| control:", round(mean(x1_age[w==0]), 1), "\n")
cat("Mean adherence   — treated:", round(mean(x3_adherence[w==1]), 3),
    "| control:", round(mean(x3_adherence[w==0]), 3), "\n")
cat("Mean past visits — treated:", round(mean(x2_visits[w==1]), 1),
    "| control:", round(mean(x2_visits[w==0]), 1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT
# tau*(X) = 0.01 for ALL individuals — constant, no heterogeneity
# ============================================================

tau_x    = rep(0.01, n)
true_tau = 0.01

cat("\n=== TRUE TREATMENT EFFECT ===\n")
cat("tau*(X) = 0.01 constant for all — zero heterogeneity by design\n")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
# Small coefficients to avoid overwhelming tau = 0.01
# ============================================================

b_x = 3.0 +
  0.15 * as.numeric(x2_s) +
  0.10 * as.numeric(x3_s) +
  -0.08 * as.numeric(x4_s) +
  -0.08 * as.numeric(x5_s) +
  0.05 * as.numeric(x1_s)

cat("\n=== BASELINE b(X) ===\n")
cat("Mean b(X):", round(mean(b_x), 3), "| SD:", round(sd(b_x), 3), "\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
# ============================================================

sigma   = 1.0
epsilon = rnorm(n, mean = 0, sd = sigma)
y_raw   = b_x + (w - propensity) * tau_x + epsilon
y       = pmax(y_raw, 0)

cat("\n=== OBSERVED OUTCOME Y ===\n")
cat("Proportion floored at zero:", round(mean(y_raw < 0), 3), "\n")
cat("Mean Y:", round(mean(y), 3), "| SD:", round(sd(y), 3), "\n")
cat("Mean Y treated:", round(mean(y[w==1]), 3),
    "| control:", round(mean(y[w==0]), 3), "\n")
cat("Naive ATE:", round(mean(y[w==1]) - mean(y[w==0]), 4),
    "| True ATE:", round(true_tau, 4), "\n")
cat("Epsilon group diff (main noise source):",
    round(mean(epsilon[w==1]) - mean(epsilon[w==0]), 4), "\n")

# ============================================================
# SECTION 6: DEFINE SUBGROUPS
#
# All subgroups have true tau = 0.01
# Any learner returning different tau_hat across groups produces
# false heterogeneity by misattributing baseline differences
#
# SG1 — High health engagers
#        Above median past visits AND above median adherence
# SG2 — Remote low engagers
#        Above median travel time AND below median past visits
# SG3 — Older adherent patients (most heavily targeted by propensity)
#        Above 75th percentile age AND above median adherence
# ============================================================

sg1 = x2_visits    >  median(x2_visits)    &
  x3_adherence >  median(x3_adherence)

sg2 = x4_travel    >  median(x4_travel)    &
  x2_visits    <  median(x2_visits)

sg3 = x1_age       >  quantile(x1_age, 0.75) &
  x3_adherence >  median(x3_adherence)

cat("\n=== SUBGROUP SIZES ===\n")
cat("SG1 (high engagers):      ", sum(sg1), "\n")
cat("SG2 (remote low engagers):", sum(sg2), "\n")
cat("SG3 (older adherent):     ", sum(sg3), "\n")
cat("True tau in all subgroups: 0.01\n")

# ============================================================
# SECTION 7: FIT ALL LEARNERS
#
# Organised by meta-learner type x base method
# Reduced boosting settings for speed at n=200
# ============================================================

boost_args = list(
  num_search_rounds     = 5,
  k_folds               = 5,
  ntrees_max            = 300,
  early_stopping_rounds = 5,
  verbose               = FALSE
)

boost_args_others = list(
  num_search_rounds     = 5,
  ntrees_max            = 300,
  early_stopping_rounds = 5,
  verbose               = FALSE
)

cat("\n=== FITTING LEARNERS ===\n")

# --- R-learners ---
cat("R-learner: rlasso...\n")
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

cat("R-learner: rboost...\n")
rboost_fit = do.call(rboost, c(list(x=x, w=w, y=y), boost_args))
rboost_est = predict(rboost_fit, x)

cat("R-learner: rkern...\n")
rkern_fit  = rkern(x, w, y)
rkern_est  = predict(rkern_fit, x)

# --- S-learners ---
cat("S-learner: slasso...\n")
slasso_fit = slasso(x, w, y)
slasso_est = predict(slasso_fit, x)

cat("S-learner: sboost...\n")
sboost_fit = do.call(sboost, c(list(x=x, w=w, y=y), boost_args))
sboost_est = predict(sboost_fit, x)

cat("S-learner: skern...\n")
skern_fit  = skern(x, w, y)
skern_est  = predict(skern_fit, x)

# --- T-learners ---
cat("T-learner: tlasso...\n")
tlasso_fit = tlasso(x, w, y)
tlasso_est = predict(tlasso_fit, x)

cat("T-learner: tboost...\n")
tboost_fit = do.call(tboost, c(list(x=x, w=w, y=y), boost_args_others))
tboost_est = predict(tboost_fit, x)

cat("T-learner: tkern...\n")
tkern_fit  = tkern(x, w, y)
tkern_est  = predict(tkern_fit, x)

# --- X-learners ---
cat("X-learner: xlasso...\n")
xlasso_fit = xlasso(x, w, y)
xlasso_est = predict(xlasso_fit, x)

cat("X-learner: xboost...\n")
xboost_fit = do.call(xboost, c(list(x=x, w=w, y=y), boost_args_others))
xboost_est = predict(xboost_fit, x)

cat("X-learner: xkern...\n")
xkern_fit  = xkern(x, w, y)
xkern_est  = predict(xkern_fit, x)

# --- Baselines ---
const_pred = rep(0.01, n)   # theoretically optimal null predictor
zero_pred  = rep(0.00, n)   # for consistency with telemedicine study

x_df = as.data.frame(x)
colnames(x_df) = c("age", "visits", "adherence", "travel", "deprivation")

ols_data = data.frame(
  y             = y,
  w             = w,
  x_df,
  w_age         = w * x_df$age,
  w_visits      = w * x_df$visits,
  w_adherence   = w * x_df$adherence,
  w_travel      = w * x_df$travel,
  w_deprivation = w * x_df$deprivation
)

ols_fit = lm(y ~ age + visits + adherence + travel + deprivation +
               w + w_age + w_visits + w_adherence +
               w_travel + w_deprivation,
             data = ols_data)

ols_tau_est = coef(ols_fit)["w"] +
  coef(ols_fit)["w_age"]         * x_df$age +
  coef(ols_fit)["w_visits"]      * x_df$visits +
  coef(ols_fit)["w_adherence"]   * x_df$adherence +
  coef(ols_fit)["w_travel"]      * x_df$travel +
  coef(ols_fit)["w_deprivation"] * x_df$deprivation

# ============================================================
# SECTION 8: ORGANISE LEARNERS FOR EVALUATION
#
# Two groupings:
#   By meta-learner type — compare R vs S vs T vs X within each base method
#   By base method — compare lasso vs boost vs kern within each meta-learner
# ============================================================

# All learners in one flat list for shared evaluation loop
learners_all = list(
  # R-learners
  rlasso     = rlasso_est,
  rboost     = rboost_est,
  rkern      = rkern_est,
  # S-learners
  slasso     = slasso_est,
  sboost     = sboost_est,
  skern      = skern_est,
  # T-learners
  tlasso     = tlasso_est,
  tboost     = tboost_est,
  tkern      = tkern_est,
  # X-learners
  xlasso     = xlasso_est,
  xboost     = xboost_est,
  xkern      = xkern_est,
  # Baselines
  const_pred = const_pred,
  zero_pred  = zero_pred,
  ols_inter  = as.numeric(ols_tau_est)
)

# Groupings for structured comparison
by_metalearner = list(
  R = c("rlasso", "rboost", "rkern"),
  S = c("slasso", "sboost", "skern"),
  T = c("tlasso", "tboost", "tkern"),
  X = c("xlasso", "xboost", "xkern")
)

by_base_method = list(
  lasso = c("rlasso", "slasso", "tlasso", "xlasso"),
  boost = c("rboost", "sboost", "tboost", "xboost"),
  kern  = c("rkern",  "skern",  "tkern",  "xkern")
)

# Colour scheme — by meta-learner type for visual consistency
metalearner_colours = c(
  rlasso = "blue",    rboost = "blue",    rkern = "blue",
  slasso = "red",     sboost = "red",     skern = "red",
  tlasso = "darkgreen", tboost = "darkgreen", tkern = "darkgreen",
  xlasso = "purple",  xboost = "purple",  xkern = "purple",
  const_pred = "black", zero_pred = "grey50", ols_inter = "orange"
)

base_lty = c(
  rlasso = 1, slasso = 1, tlasso = 1, xlasso = 1,   # lasso = solid
  rboost = 2, sboost = 2, tboost = 2, xboost = 2,   # boost = dashed
  rkern  = 3, skern  = 3, tkern  = 3, xkern  = 3,   # kern  = dotted
  const_pred = 1, zero_pred = 5, ols_inter = 4
)

# ============================================================
# SECTION 9: EVALUATION
#
# Primary: spread of tau_hat — lower is better for null study
# Secondary: bias, subgroup false positives
# ============================================================

pass_threshold    = 0.20   # IQR threshold for calibration pass at n=200
false_pos_threshold = 0.10

# --- Build full metrics table ---
compute_metrics = function(est, name) {
  raw_mse  = mean((est - tau_x)^2)
  bias     = mean(est) - true_tau
  iqr_est  = IQR(est)
  range_90 = as.numeric(quantile(est, 0.95) - quantile(est, 0.05))
  sd_est   = sd(est)
  passes   = iqr_est < pass_threshold
  
  est_sg1  = mean(est[sg1])
  est_sg2  = mean(est[sg2])
  est_sg3  = mean(est[sg3])
  
  fp_sg1   = abs(est_sg1 - true_tau) > false_pos_threshold
  fp_sg2   = abs(est_sg2 - true_tau) > false_pos_threshold
  fp_sg3   = abs(est_sg3 - true_tau) > false_pos_threshold
  
  data.frame(
    Learner   = name,
    Bias      = round(bias,    4),
    IQR       = round(iqr_est, 4),
    Range_90  = round(range_90,4),
    SD        = round(sd_est,  4),
    Pass_Cal  = passes,
    FP_SG1    = fp_sg1,
    FP_SG2    = fp_sg2,
    FP_SG3    = fp_sg3,
    stringsAsFactors = FALSE
  )
}

metrics_list = mapply(compute_metrics,
                      learners_all,
                      names(learners_all),
                      SIMPLIFY = FALSE)
metrics_all  = do.call(rbind, metrics_list)

# ============================================================
# PRINT COMPARISON 1: BY META-LEARNER TYPE
# For a given base method, does R outperform S, T, X
# on calibration?
# ============================================================

cat("\n===========================================\n")
cat("COMPARISON 1: BY META-LEARNER TYPE\n")
cat("Within each base method — R vs S vs T vs X\n")
cat("Lower IQR and Range_90 = better calibration\n")
cat("===========================================\n\n")

for (base in c("lasso", "boost", "kern")) {
  cat("--- Base method:", toupper(base), "---\n")
  learner_names = c(paste0("r", base), paste0("s", base),
                    paste0("t", base), paste0("x", base))
  subset_metrics = metrics_all[metrics_all$Learner %in% learner_names, ]
  print(subset_metrics[, c("Learner","Bias","IQR","Range_90","Pass_Cal",
                           "FP_SG1","FP_SG2","FP_SG3")],
        row.names = FALSE)
  cat("\n")
}

# ============================================================
# PRINT COMPARISON 2: BY BASE METHOD
# For a given meta-learner, does lasso, boost or kern
# produce less spurious heterogeneity?
# ============================================================

cat("===========================================\n")
cat("COMPARISON 2: BY BASE METHOD\n")
cat("Within each meta-learner — lasso vs boost vs kern\n")
cat("Lower IQR and Range_90 = better calibration\n")
cat("===========================================\n\n")

for (ml in c("r", "s", "t", "x")) {
  ml_name = switch(ml, r="R-LEARNER", s="S-LEARNER",
                   t="T-LEARNER", x="X-LEARNER")
  cat("---", ml_name, "---\n")
  learner_names = paste0(ml, c("lasso", "boost", "kern"))
  subset_metrics = metrics_all[metrics_all$Learner %in% learner_names, ]
  print(subset_metrics[, c("Learner","Bias","IQR","Range_90","Pass_Cal",
                           "FP_SG1","FP_SG2","FP_SG3")],
        row.names = FALSE)
  cat("\n")
}

# ============================================================
# PRINT COMPARISON 3: BASELINES
# ============================================================

cat("===========================================\n")
cat("COMPARISON 3: BASELINES\n")
cat("===========================================\n\n")
baseline_names = c("const_pred", "zero_pred", "ols_inter")
baseline_metrics = metrics_all[metrics_all$Learner %in% baseline_names, ]
print(baseline_metrics[, c("Learner","Bias","IQR","Range_90","Pass_Cal",
                           "FP_SG1","FP_SG2","FP_SG3")],
      row.names = FALSE)

# ============================================================
# SECTION 10: VISUALISATIONS
#
# Plot A — KDE by meta-learner type (4 panels, one per type)
# Plot B — KDE by base method (3 panels, one per method)
# Plot C — PDP plots (5 covariates, all learners)
# Plot D — Scatter: true vs estimated tau
# ============================================================

cat("\n=== GENERATING VISUALISATIONS ===\n")

# Helper: compute KDE safely
safe_kde = function(est) {
  if (sd(est) < 1e-10) return(NULL)
  density(est, n = 500)
}

# ---- PLOT A: KDE by meta-learner type ----
# Each panel shows lasso/boost/kern variants of one meta-learner
# True tau = 0.01 shown as vertical line
# Ideal: all lines are tight spikes at 0.01

par(mfrow = c(2, 2))

for (ml in c("R", "S", "T", "X")) {
  learner_names = by_metalearner[[ml]]
  
  # Gather all estimates for x-range
  all_in_group = unlist(learners_all[learner_names])
  xmin = min(all_in_group) - 0.05
  xmax = max(all_in_group) + 0.05
  
  kdes = lapply(learners_all[learner_names], safe_kde)
  
  ymax = max(sapply(kdes[!sapply(kdes, is.null)],
                    function(d) max(d$y)), na.rm = TRUE) * 1.15
  
  plot(NULL,
       xlim = c(xmin, xmax),
       ylim = c(0, ymax),
       xlab = "Estimated tau(X)",
       ylab = "Density",
       main = paste0(ml, "-Learner: lasso vs boost vs kern\n",
                     "Ideal: spike at 0.01"))
  
  abline(v = 0.01, col = "black",  lwd = 2, lty = 1)
  abline(v = 0,    col = "grey70", lwd = 1, lty = 2)
  
  line_cols = c("blue", "red", "darkgreen")
  line_ltys = c(1, 2, 3)
  labels    = learner_names
  
  for (k in seq_along(learner_names)) {
    nm = learner_names[k]
    if (is.null(kdes[[nm]])) {
      abline(v = mean(learners_all[[nm]]),
             col = line_cols[k], lwd = 2, lty = line_ltys[k])
    } else {
      lines(kdes[[nm]]$x, kdes[[nm]]$y,
            col = line_cols[k], lwd = 2, lty = line_ltys[k])
    }
  }
  
  legend("topright",
         legend = c("True tau=0.01", labels),
         col    = c("black", line_cols),
         lwd    = 2,
         lty    = c(1, line_ltys),
         cex    = 0.7,
         bg     = "white")
}

par(mfrow = c(1, 1))

# ---- PLOT B: KDE by base method ----
# Each panel shows R/S/T/X variants of one base method
# Shows whether the base method matters more than the meta-learner

par(mfrow = c(1, 3))

base_labels = c(lasso = "Lasso", boost = "Boost", kern = "Kernel")

for (base in c("lasso", "boost", "kern")) {
  learner_names = by_base_method[[base]]
  
  all_in_group = unlist(learners_all[learner_names])
  xmin = min(all_in_group) - 0.05
  xmax = max(all_in_group) + 0.05
  
  kdes = lapply(learners_all[learner_names], safe_kde)
  
  ymax = max(sapply(kdes[!sapply(kdes, is.null)],
                    function(d) max(d$y)), na.rm = TRUE) * 1.15
  
  plot(NULL,
       xlim = c(xmin, xmax),
       ylim = c(0, ymax),
       xlab = "Estimated tau(X)",
       ylab = "Density",
       main = paste0(base_labels[base],
                     ": R vs S vs T vs X learner\n",
                     "Ideal: spike at 0.01"))
  
  abline(v = 0.01, col = "black",  lwd = 2, lty = 1)
  abline(v = 0,    col = "grey70", lwd = 1, lty = 2)
  
  ml_cols = c("blue", "red", "darkgreen", "purple")
  ml_ltys = c(1, 2, 3, 4)
  
  for (k in seq_along(learner_names)) {
    nm = learner_names[k]
    if (is.null(kdes[[nm]])) {
      abline(v = mean(learners_all[[nm]]),
             col = ml_cols[k], lwd = 2, lty = ml_ltys[k])
    } else {
      lines(kdes[[nm]]$x, kdes[[nm]]$y,
            col = ml_cols[k], lwd = 2, lty = ml_ltys[k])
    }
  }
  
  legend("topright",
         legend = c("True tau=0.01",
                    paste0("R-", base_labels[base]),
                    paste0("S-", base_labels[base]),
                    paste0("T-", base_labels[base]),
                    paste0("X-", base_labels[base])),
         col    = c("black", ml_cols),
         lwd    = 2,
         lty    = c(1, ml_ltys),
         cex    = 0.65,
         bg     = "white")
}

par(mfrow = c(1, 1))

# ---- PLOT C: PDP plots for R-learner variants ----
# All lines should be flat at 0.01
# Any slope = false covariate-treatment interaction

covariate_labels = c("Age (years)",
                     "Past GP Visits (24m)",
                     "Medication Adherence",
                     "Travel Time (min)",
                     "Deprivation Index")
covariate_vals   = list(x1_age, x2_visits, x3_adherence,
                        x4_travel, x5_deprivation)

# Show R-learner variants — most theoretically motivated
cat("Generating PDP plots for R-learner variants...\n")
par(mfrow = c(2, 3))

for (j in 1:5) {
  cov_vals = covariate_vals[[j]]
  ord      = order(cov_vals)
  
  y_range  = range(c(rlasso_est, rboost_est, rkern_est, 0.01))
  
  plot(NULL,
       xlim = range(cov_vals),
       ylim = y_range,
       xlab = covariate_labels[j],
       ylab = "Treatment effect (visits/year)",
       main = paste("PDP:", covariate_labels[j]))
  
  abline(h = 0.01, col = "black",  lwd = 2, lty = 1)
  abline(h = 0,    col = "grey70", lwd = 1, lty = 2)
  
  lines(cov_vals[ord], rlasso_est[ord], col = "blue",      lwd = 2, lty = 1)
  lines(cov_vals[ord], rboost_est[ord], col = "red",       lwd = 2, lty = 2)
  lines(cov_vals[ord], rkern_est[ord],  col = "darkgreen", lwd = 2, lty = 3)
  
  legend("topleft",
         legend = c("True tau=0.01", "rlasso", "rboost", "rkern"),
         col    = c("black", "blue", "red", "darkgreen"),
         lwd    = 2, lty = c(1,1,2,3), cex = 0.65, bg = "white")
}

# Use 6th panel for S-learner comparison
y_range_s = range(c(slasso_est, sboost_est, skern_est, 0.01))

plot(NULL,
     xlim = range(x1_age),
     ylim = y_range_s,
     xlab = "Age (years)",
     ylab = "Treatment effect (visits/year)",
     main = "PDP Age: S-learner variants\n(for comparison)")

abline(h = 0.01, col = "black",  lwd = 2, lty = 1)
abline(h = 0,    col = "grey70", lwd = 1, lty = 2)

ord_age = order(x1_age)
lines(x1_age[ord_age], slasso_est[ord_age], col = "blue",      lwd = 2, lty = 1)
lines(x1_age[ord_age], sboost_est[ord_age], col = "red",       lwd = 2, lty = 2)
lines(x1_age[ord_age], skern_est[ord_age],  col = "darkgreen", lwd = 2, lty = 3)

legend("topleft",
       legend = c("True tau=0.01", "slasso", "sboost", "skern"),
       col    = c("black", "blue", "red", "darkgreen"),
       lwd    = 2, lty = c(1,1,2,3), cex = 0.65, bg = "white")

par(mfrow = c(1, 1))

# ---- PLOT D: Scatter true vs estimated tau ----
# All points should sit at x = 0.01
# Vertical spread = spurious heterogeneity

# Show all 12 learners, coloured by meta-learner type
all_estimates = c(rlasso_est, rboost_est, rkern_est,
                  slasso_est, sboost_est, skern_est,
                  tlasso_est, tboost_est, tkern_est,
                  xlasso_est, xboost_est, xkern_est)

par(mfrow = c(1, 1))
plot(NULL,
     xlim = c(0, 0.02),
     ylim = range(all_estimates),
     xlab = "True tau(X) = 0.01 for all individuals",
     ylab = "Estimated tau_hat(X)",
     main = "True vs Estimated tau — All Learners\nPoints at x=0.01, vertical spread = false heterogeneity")

abline(h = 0.01, col = "black",  lwd = 2, lty = 1)
abline(h = 0,    col = "grey60", lwd = 1, lty = 2)

point_cols = c("blue", "blue", "blue",
               "red",  "red",  "red",
               "darkgreen", "darkgreen", "darkgreen",
               "purple", "purple", "purple")
point_pchs = c(16, 17, 18,
               16, 17, 18,
               16, 17, 18,
               16, 17, 18)

all_est_list = list(rlasso_est, rboost_est, rkern_est,
                    slasso_est, sboost_est, skern_est,
                    tlasso_est, tboost_est, tkern_est,
                    xlasso_est, xboost_est, xkern_est)

for (k in seq_along(all_est_list)) {
  points(jitter(tau_x, factor = 0.5),
         all_est_list[[k]],
         pch = point_pchs[k],
         col = adjustcolor(point_cols[k], alpha.f = 0.35),
         cex = 0.8)
}

legend("topright",
       legend = c("R-learner", "S-learner", "T-learner", "X-learner",
                  "pch=circle: lasso",
                  "pch=triangle: boost",
                  "pch=diamond: kern"),
       col    = c("blue", "red", "darkgreen", "purple",
                  "grey40", "grey40", "grey40"),
       pch    = c(15, 15, 15, 15, 16, 17, 18),
       cex    = 0.75,
       bg     = "white")

# ============================================================
# SECTION 11: FULL SUMMARY TABLE — ALL LEARNERS
# ============================================================

cat("\n===========================================\n")
cat("FULL SUMMARY TABLE — ALL LEARNERS\n")
cat("Sorted by IQR ascending (best calibration first)\n")
cat("===========================================\n")

full_table = metrics_all[order(metrics_all$IQR), ]
print(full_table, row.names = FALSE)

cat("\nInterpretation guide:\n")
cat("  Bias:     mean tau_hat minus 0.01 — should be near zero\n")
cat("  IQR:      spread of individual estimates — PRIMARY metric, lower = better\n")
cat("  Range_90: P95 minus P05 — captures tail spread\n")
cat("  Pass_Cal: TRUE if IQR < 0.20 (calibration pass at n=200)\n")
cat("  FP_SG:    TRUE if subgroup mean deviates > 0.10 from true tau\n\n")

cat("Key questions from the two comparison dimensions:\n")
cat("  1. Within each base method, does R produce less\n")
cat("     spurious heterogeneity than S, T, or X?\n")
cat("     (R-learner should have lower IQR due to residualisation)\n\n")
cat("  2. Within each meta-learner, does the base method matter?\n")
cat("     (Lasso may collapse to constant — low IQR but uninformative)\n")
cat("     (Boost may overfit noise — high IQR, fails calibration)\n")
cat("     (Kern sits between — moderate flexibility)\n\n")
cat("  3. Compare with telemedicine study:\n")
cat("     Telemedicine = POWER test (find real heterogeneity)\n")
cat("     This study    = CALIBRATION test (stay quiet when nothing there)\n")
cat("     A learner passing both is trustworthy in practice\n")