# ============================================================
# TELEMEDICINE SETUP — SINGLE ITERATION
# Outcome: Raw systolic BP at 12 months (lower = better)
# Treatment: Telemedicine enrolment (W = 1) vs in-person (W = 0)
# Key feature: tau(X) has sign change across patient subgroups
# Population average treatment effect is close to zero
# ============================================================

library(MASS)       # for mvrnorm — correlated covariate generation
library(rlearner)   # for rlasso and rboost
library(KRLS2)

set.seed(42)
n = 200

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: baseline systolic BP (mmHg)         — range 110 to 190
# X2: travel time to clinic (minutes)     — range 5 to 120
# X3: prior digital health interactions   — range 0 to 20 (count)
# X4: number of comorbidities             — range 0 to 5
# X5: age (years)                         — range 40 to 80
#
# Correlation structure (realistic for a mixed community cohort):
#   age <-> comorbidities:       +0.45  older patients accumulate conditions
#   age <-> digital engagement:  -0.40  older patients use digital services less
#   baseline BP <-> comorbidities: +0.35 sicker patients have worse BP control
#   baseline BP <-> age:          +0.25 BP tends to rise with age
#   travel time is largely independent of clinical variables
#   comorbidities <-> digital:   -0.20  complex patients slightly less digitally engaged
# ============================================================

cor_matrix = matrix(c(
  # X1_bp  X2_travel  X3_digital  X4_comorbid  X5_age
  1.00,    0.05,      -0.10,       0.35,        0.25,   # X1 baseline BP
  0.05,    1.00,       0.05,       0.00,       -0.05,   # X2 travel time
  -0.10,    0.05,       1.00,      -0.20,       -0.40,   # X3 digital engagement
  0.35,    0.00,      -0.20,       1.00,        0.45,   # X4 comorbidities
  0.25,   -0.05,      -0.40,       0.45,        1.00    # X5 age
), nrow = 5, byrow = TRUE)

# Generate correlated standard normal data
z = mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

# Transform each variable from standard normal to its realistic scale
# pnorm maps z-scores to (0,1) probabilities first,
# then we scale to the target range

# X1: baseline systolic BP — 110 to 190 mmHg
# Mean around 145 is realistic for a hypertension cohort
x1_bp       = 110 + 80 * pnorm(z[,1])

# X2: travel time — 5 to 120 minutes
# Right-skewed distribution reflects most patients living fairly close
# with a tail of genuinely remote patients
x2_travel = 20 + 60 * pnorm(z[,2])

# X3: prior digital health interactions — 0 to 20
# Rounded to integer as this is a count
x3_digital  = round(20 * pnorm(z[,3]))

# X4: number of comorbidities — 0 to 5
# Rounded to integer, most patients have 1 or 2
x4_comorbid = round(5 * pnorm(z[,4]))

# X5: age — 40 to 80 years
# Rounded to integer
x5_age      = round(40 + 40 * pnorm(z[,5]))

# Combine into matrix for rlearner
# rlearner expects a plain numeric matrix
x = cbind(x1_bp, x2_travel, x3_digital, x4_comorbid, x5_age)
colnames(x) = c("bp_baseline", "travel_time", "digital_prior",
                "comorbidities", "age")

# Quick sanity check on covariate distributions
cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))

cat("\n=== COVARIATE CORRELATION (should reflect design above) ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
#
# Clinical logic:
#   Travel time (X2) is the dominant driver — programme targets remote patients
#   Baseline BP (X1) has moderate positive effect — system wants to reach poorly
#     controlled patients at risk of dropping out of in-person care
#   Age (X5) has negative effect — elderly patients kept on in-person pathways
#     by GPs and families
#
# Expected enrolment rate: roughly 40-50% reflecting a broad rollout
# to remote and moderately unwell patients
# ============================================================

# Scale covariates before entering propensity model
# This ensures coefficients are comparable across variables
x1_s = scale(x1_bp)
x2_s = scale(x2_travel)
x3_s = scale(x3_digital)
x4_s = scale(x4_comorbid)
x5_s = scale(x5_age)

log_odds_w = -0.5 +
  1.5 * x2_s +    # travel time dominant positive driver
  0.5 * x1_s +    # poorly controlled BP moderate positive
  -0.4 * x5_s      # older patients slightly less likely enrolled

propensity = plogis(log_odds_w)

# Clip to avoid exact 0 or 1 which causes numerical problems in learners
propensity = pmax(0.05, pmin(propensity, 0.95))

# Generate treatment assignment
w = rbinom(n, 1, propensity)

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion enrolled in telemedicine:", round(mean(w), 3), "\n")
cat("Propensity score range:",
    round(min(propensity), 3), "to", round(max(propensity), 3), "\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")

# Check confounding — do treated and control differ on key covariates?
cat("\nMean baseline BP  — treated:", round(mean(x1_bp[w==1]), 1),
    "| control:", round(mean(x1_bp[w==0]), 1), "\n")
cat("Mean travel time  — treated:", round(mean(x2_travel[w==1]), 1),
    "| control:", round(mean(x2_travel[w==0]), 1), "\n")
cat("Mean age          — treated:", round(mean(x5_age[w==1]), 1),
    "| control:", round(mean(x5_age[w==0]), 1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)
#
# Clinical logic for sign change:
#   X2 travel time:    negative coefficient — remote patients benefit (BP reduced)
#   X1 baseline BP:    positive coefficient — severely uncontrolled patients harmed
#   X4 comorbidities:  positive coefficient — complex patients harmed
#   X3 digital:        negative coefficient — digitally engaged patients benefit
#   X5 age:            positive coefficient — older patients slightly harmed
#   Constant:          calibrated so population average tau is close to zero
#
# All covariates scaled so coefficients are directly comparable
# Negative tau = telemedicine reduces BP = beneficial
# Positive tau = telemedicine increases BP = harmful
# ============================================================

tau_x = -2.0 * x2_s +    # remote patients benefit
  2.0 * x1_s +    # poorly controlled patients harmed
  1.5 * x4_s +    # complex multimorbid patients harmed
  -0.8 * x3_s +    # digitally engaged patients benefit
  0.5 * x5_s +    # older patients slightly harmed
  -0.7             # calibration constant — pulls average towards zero

cat("\n=== TRUE TREATMENT EFFECT SUMMARY ===\n")
cat("Mean tau(X):", round(mean(tau_x), 3),
    "(should be close to zero)\n")
cat("SD of tau(X):", round(sd(tau_x), 3), "\n")
cat("Range of tau(X):", round(min(tau_x), 3),
    "to", round(max(tau_x), 3), "\n")
cat("Proportion with negative tau (telemedicine beneficial):",
    round(mean(tau_x < 0), 3), "\n")
cat("Proportion with positive tau (telemedicine harmful):",
    round(mean(tau_x > 0), 3), "\n")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
#
# b(X) is what a patient's systolic BP would be at 12 months
# under continued standard in-person care — before any
# telemedicine effect is applied
#
# Clinical logic:
#   Higher baseline BP persists — strong positive effect
#   More comorbidities worsen BP trajectory
#   Older patients have less responsive BP to standard management
#   Digital engagement reflects general health engagement
#     and is associated with better adherence and lower follow-up BP
# ============================================================

b_x = 0.6 * x1_bp +         # baseline BP strongly predicts follow-up BP
  2.5 * x4_comorbid +    # each additional comorbidity adds ~2.5 mmHg
  0.3 * x5_age +         # each decade adds modest upward pressure
  -1.2 * x3_digital +     # each digital interaction associated with -1.2 mmHg
  20                     # intercept — sets plausible absolute level

cat("\n=== BASELINE OUTCOME b(X) SUMMARY ===\n")
cat("Mean b(X):", round(mean(b_x), 2), "mmHg\n")
cat("Range b(X):", round(min(b_x), 2),
    "to", round(max(b_x), 2), "mmHg\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
#
# Y = b(X) + (W - propensity) * tau(X) + epsilon
#
# Using actual propensity rather than 0.5 because treatment
# is not randomised here — propensity varies across patients
# This means the treatment component is centred around zero
# for each patient given their actual probability of treatment
#
# Noise: SD of 5 mmHg reflects realistic measurement variability
# and unmeasured factors in BP at 12 months
# ============================================================

sigma   = 5
epsilon = rnorm(n, mean = 0, sd = sigma)

y = b_x + (w - propensity) * tau_x + epsilon

cat("\n=== OBSERVED OUTCOME Y SUMMARY ===\n")
cat("Mean Y (all patients):",   round(mean(y), 2), "mmHg\n")
cat("Mean Y (treated, W=1):",   round(mean(y[w==1]), 2), "mmHg\n")
cat("Mean Y (control, W=0):",   round(mean(y[w==0]), 2), "mmHg\n")
cat("Naive ATE (treated - control):",
    round(mean(y[w==1]) - mean(y[w==0]), 3),
    "mmHg (confounded — not the true ATE)\n")
cat("True ATE (mean tau):", round(mean(tau_x), 3), "mmHg\n")

# ============================================================
# SECTION 6: DEFINE SUBGROUPS BEFORE FITTING
#
# Subgroup 1 — Young, remote, digitally engaged (should benefit)
#   Below median age AND above median travel time
#   AND above median digital interactions
#
# Subgroup 2 — Severely ill, complex (should be harmed)
#   Above 75th percentile baseline BP
#   AND above 75th percentile comorbidities
#
# Subgroup 3 — Elderly, low digital engagement (near-zero / mildly harmful)
#   Above 75th percentile age
#   AND below 25th percentile digital interactions
# ============================================================

sg1 = x5_age    <  median(x5_age)    &
  x2_travel >  median(x2_travel) &
  x3_digital > median(x3_digital)

sg2 = x1_bp     > quantile(x1_bp,     0.75) &
  x4_comorbid > quantile(x4_comorbid, 0.75)

sg3 = x5_age    >  quantile(x5_age,    0.75) &
  x3_digital < quantile(x3_digital, 0.25)

cat("\n=== SUBGROUP SIZES ===\n")
cat("Subgroup 1 (young, remote, digital):", sum(sg1), "patients\n")
cat("Subgroup 2 (severe, complex):",        sum(sg2), "patients\n")
cat("Subgroup 3 (elderly, low digital):",   sum(sg3), "patients\n")

cat("\n=== TRUE TAU BY SUBGROUP ===\n")
cat("SG1 mean true tau:", round(mean(tau_x[sg1]), 3),
    "(expect negative — beneficial)\n")
cat("SG2 mean true tau:", round(mean(tau_x[sg2]), 3),
    "(expect positive — harmful)\n")
cat("SG3 mean true tau:", round(mean(tau_x[sg3]), 3),
    "(expect near zero or mildly positive)\n")

# ============================================================
# SECTION 7: FIT LEARNERS
# ============================================================

start_time = Sys.time()
cat("\n=== FITTING LEARNERS ===\n")
cat("Fitting rlasso...\n")
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

cat("Fitting rboost...\n")
rboost_fit = rboost(x, w, y, num_search_rounds = 5, k_folds = 5, ntrees_max = 300, early_stopping_rounds = 5, verbose = T)
rboost_est = predict(rboost_fit, x)

cat("Fitting rkern...\n")
# Reduced grid for exploratory use
rkern_fit = rkern(x, w, y)
rkern_est = predict(rkern_fit, x)

cat("Fitting slasso...\n")
slasso_fit = slasso(x, w, y)
slasso_est = predict(slasso_fit, x)

cat("Fitting sboost...\n")
sboost_fit = sboost(x, w, y, num_search_rounds = 5, k_folds = 5, ntrees_max = 300, early_stopping_rounds = 5, verbose = T)
sboost_est = predict(sboost_fit, x)

cat("Fitting skern...\n")
# Reduced grid for exploratory use
skern_fit = skern(x, w, y)
skern_est = predict(skern_fit, x)

cat("Fitting tlasso...\n")
tlasso_fit = tlasso(x, w, y)
tlasso_est = predict(tlasso_fit, x)

cat("Fitting tboost...\n")
tboost_fit = tboost(x, w, y, ntrees_max = 300, early_stopping_rounds = 5, verbose = T)
tboost_est = predict(tboost_fit, x)

cat("Fitting tkern...\n")
# Reduced grid for exploratory use
tkern_fit = tkern(x, w, y)
tkern_est = predict(tkern_fit, x)

cat("Fitting xlasso...\n")
xlasso_fit = xlasso(x, w, y)
xlasso_est = predict(xlasso_fit, x)

cat("Fitting xboost...\n")
xboost_fit = xboost(x, w, y, ntrees_max = 300, early_stopping_rounds = 5, verbose = T)
xboost_est = predict(xboost_fit, x)

cat("Fitting xkern...\n")
# Reduced grid for exploratory use
xkern_fit = xkern(x, w, y)
xkern_est = predict(xkern_fit, x)

learners = list(rlasso = rlasso_est,
                rboost = rboost_est,
                rkern = rkern_est,
                slasso = slasso_est,
                sboost = sboost_est,
                skern = skern_est,
                tlasso = tlasso_est,
                tboost = tboost_est,
                tkern = tkern_est,
                xlasso = xlasso_est,
                xboost = xboost_est,
                xkern = xkern_est)

# ============================================================
# SECTION 8: EVALUATION
# Three complementary measures as discussed previously
# ============================================================

tau_variance = var(tau_x)

cat("\n=== MEASURE 1: NORMALISED MSE ===\n")
cat("Variance of true tau(X):", round(tau_variance, 4), "\n\n")

for (name in names(learners)) {
  est      = learners[[name]]
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  
  quality  = ifelse(norm_mse < 0.25, "EXCELLENT",
                    ifelse(norm_mse < 0.75, "ACCEPTABLE",
                           ifelse(norm_mse < 1.00, "POOR",
                                  "WORSE THAN MEAN")))
  
  cat("Learner:", name, "\n")
  cat("  Raw MSE:        ", round(raw_mse,  4), "\n")
  cat("  Normalised MSE: ", round(norm_mse, 4), "\n")
  cat("  Performance:    ", quality, "\n\n")
}

cat("Baseline (constant mean prediction) MSE:",
    round(var(tau_x), 4), "\n")

cat("\n=== MEASURE 2: RANK CORRELATION ===\n")
for (name in names(learners)) {
  est       = learners[[name]]
  rank_corr = cor(est, tau_x, method = "spearman")
  
  quality   = ifelse(rank_corr > 0.80, "RELIABLE RANKING",
                     ifelse(rank_corr > 0.50, "MODERATE SIGNAL",
                            ifelse(rank_corr > 0.30, "WEAK SIGNAL",
                                   "ESSENTIALLY RANDOM")))
  
  cat("Learner:", name, "\n")
  cat("  Spearman correlation:", round(rank_corr, 4), "\n")
  cat("  Ranking quality:     ", quality, "\n\n")
}
end_time = Sys.time()
end_time-start_time

cat("\n=== MEASURE 3: SUBGROUP RECOVERY ===\n")

true_sg1 = mean(tau_x[sg1])
true_sg2 = mean(tau_x[sg2])
true_sg3 = mean(tau_x[sg3])

cat("True mean tau by subgroup:\n")
cat("  SG1 (young/remote/digital):", round(true_sg1, 3), "\n")
cat("  SG2 (severe/complex):      ", round(true_sg2, 3), "\n")
cat("  SG3 (elderly/low digital): ", round(true_sg3, 3), "\n\n")

for (name in names(learners)) {
  est = learners[[name]]
  
  est_sg1 = mean(est[sg1])
  est_sg2 = mean(est[sg2])
  est_sg3 = mean(est[sg3])
  
  # Recovery ratio: estimated subgroup mean / true subgroup mean
  # Values close to 1 mean the learner correctly estimates the magnitude
  # Negative recovery means the learner got the sign wrong
  rec_sg1 = est_sg1 / true_sg1
  rec_sg2 = est_sg2 / true_sg2
  rec_sg3 = ifelse(abs(true_sg3) < 0.1, NA,   # avoid dividing near zero
                   est_sg3 / true_sg3)
  
  # Sign recovery — most important check for this setup
  correct_sign_sg1 = sign(est_sg1) == sign(true_sg1)
  correct_sign_sg2 = sign(est_sg2) == sign(true_sg2)
  
  cat("Learner:", name, "\n")
  cat("  SG1 estimated:", round(est_sg1, 3),
      "| true:", round(true_sg1, 3),
      "| recovery:", round(rec_sg1, 3),
      "| correct sign:", correct_sign_sg1, "\n")
  cat("  SG2 estimated:", round(est_sg2, 3),
      "| true:", round(true_sg2, 3),
      "| recovery:", round(rec_sg2, 3),
      "| correct sign:", correct_sign_sg2, "\n")
  cat("  SG3 estimated:", round(est_sg3, 3),
      "| true:", round(true_sg3, 3),
      "| recovery:", ifelse(is.na(rec_sg3), "NA (near zero)", 
                            as.character(round(rec_sg3, 3))), "\n\n")
}

# ============================================================
# SECTION 9: PARTIAL RELATIONSHIP PLOTS
# One plot per covariate showing true vs estimated tau
# as that covariate varies, all others held at their mean
# ============================================================

par(mfrow = c(2, 3))

covariate_names = c("bp_baseline", "travel_time", "digital_prior",
                    "comorbidities", "age")
covariate_vals  = list(x1_bp, x2_travel, x3_digital, x4_comorbid, x5_age)

for (j in 1:5) {
  
  cov_vals = covariate_vals[[j]]
  ord      = order(cov_vals)
  
  plot(cov_vals[ord], tau_x[ord],
       type = "l", col = "black", lwd = 2,
       xlab = covariate_names[j],
       ylab = "Treatment effect tau (mmHg)",
       main = paste("tau vs", covariate_names[j]),
       ylim = range(c(tau_x,
                      rlasso_est,
                      rboost_est)))
  
  lines(cov_vals[ord], rlasso_est[ord],
        col = "blue", lwd = 2, lty = 2)
  lines(cov_vals[ord], rboost_est[ord],
        col = "red",  lwd = 2, lty = 3)
  
  abline(h = 0, col = "grey50", lty = 1)
  
  legend("topleft",
         legend = c("True tau", "rlasso", "rboost"),
         col    = c("black", "blue", "red"),
         lwd    = 2,
         lty    = c(1, 2, 3),
         cex    = 0.7)
}

# Scatter plot: true tau vs estimated tau for each learner
plot(tau_x, rlasso_est,
     pch = 16, col = rgb(0, 0, 1, 0.4),
     xlab = "True tau(X)",
     ylab = "Estimated tau(X)",
     main = "True vs Estimated tau")
points(tau_x, rboost_est,
       pch = 17, col = rgb(1, 0, 0, 0.4))
abline(0, 1, col = "black", lwd = 2)
abline(h = 0, col = "grey50", lty = 2)
abline(v = 0, col = "grey50", lty = 2)
legend("topleft",
       legend = c("rlasso", "rboost", "45-degree line"),
       col    = c("blue", "red", "black"),
       pch    = c(16, 17, NA),
       lty    = c(NA, NA, 1),
       cex    = 0.5)

par(mfrow = c(1, 1))

# ============================================================
# ADDITIONAL BASELINES
# Baseline 1: Constant zero predictor
# Baseline 4: OLS with treatment-covariate interactions
# ============================================================

# --- Baseline 1: Constant zero predictor ---
# Predicts tau = 0 for everyone
# Represents a researcher who concludes no treatment effect exists
# MSE of this predictor equals the variance of tau by definition
zero_pred = rep(0, n)

# --- Baseline 4: OLS with interactions ---
# Regress Y on all covariates, W, and W x covariate interactions
# This produces a heterogeneous treatment effect estimate
# that is linear in each covariate separately
# tau estimate for each individual is extracted as:
# tau_hat(X) = coef(W) + coef(W:X1)*X1 + coef(W:X2)*X2 + ...

# Build interaction design matrix
x_df = as.data.frame(x)
colnames(x_df) = c("bp", "travel", "digital", "comorbid", "age")

# Create interaction terms between W and each covariate
ols_data = data.frame(
  y       = y,
  w       = w,
  x_df,
  w_bp      = w * x_df$bp,
  w_travel  = w * x_df$travel,
  w_digital = w * x_df$digital,
  w_comorbid= w * x_df$comorbid,
  w_age     = w * x_df$age
)

ols_fit = lm(y ~ bp + travel + digital + comorbid + age +
               w + w_bp + w_travel + w_digital + 
               w_comorbid + w_age,
             data = ols_data)

# Extract implied treatment effect for each individual
# tau_hat(Xi) = coef[W] + coef[W:bp]*bp_i + coef[W:travel]*travel_i + ...
coef_w          = coef(ols_fit)["w"]
coef_w_bp       = coef(ols_fit)["w_bp"]
coef_w_travel   = coef(ols_fit)["w_travel"]
coef_w_digital  = coef(ols_fit)["w_digital"]
coef_w_comorbid = coef(ols_fit)["w_comorbid"]
coef_w_age      = coef(ols_fit)["w_age"]

ols_tau_est = coef_w +
  coef_w_bp       * x_df$bp +
  coef_w_travel   * x_df$travel +
  coef_w_digital  * x_df$digital +
  coef_w_comorbid * x_df$comorbid +
  coef_w_age      * x_df$age

# Add baselines to learner list
learners_all = list(
  rlasso     = rlasso_est,
  rboost     = rboost_est,
  zero_pred  = zero_pred,
  ols_inter  = as.numeric(ols_tau_est)
)

# ============================================================
# EVALUATION ACROSS ALL LEARNERS AND BASELINES
# ============================================================

cat("===========================================\n")
cat("MEASURE 1: NORMALISED MSE\n")
cat("===========================================\n")
cat("Variance of true tau(X):", round(tau_variance, 4), "\n")
cat("Note: normalised MSE = 1.0 means no better than zero predictor\n\n")

for (name in names(learners_all)) {
  est      = learners_all[[name]]
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  
  quality  = ifelse(norm_mse < 0.25, "EXCELLENT",
                    ifelse(norm_mse < 0.75, "ACCEPTABLE",
                           ifelse(norm_mse < 1.00, "POOR",
                                  "WORSE THAN MEAN")))
  
  cat("Learner:", name, "\n")
  cat("  Raw MSE:        ", round(raw_mse,  4), "\n")
  cat("  Normalised MSE: ", round(norm_mse, 4), "\n")
  cat("  Performance:    ", quality, "\n\n")
}

# Sanity check: zero predictor normalised MSE should equal exactly 1.0
# because MSE of zero predictor = E[tau^2] = var(tau) when mean(tau) ~ 0
cat("Sanity check — zero predictor normalised MSE should be ~1.0:",
    round(mean((zero_pred - tau_x)^2) / tau_variance, 4), "\n\n")

cat("===========================================\n")
cat("MEASURE 2: RANK CORRELATION\n")
cat("===========================================\n")
cat("Note: zero predictor has undefined rank correlation\n")
cat("      (all predictions identical — no ranking information)\n\n")

for (name in names(learners_all)) {
  est = learners_all[[name]]
  
  # Zero predictor has no variation so rank correlation is undefined
  if (sd(est) < 1e-10) {
    cat("Learner:", name, "\n")
    cat("  Spearman correlation: NA (all predictions identical)\n")
    cat("  Ranking quality:      UNDEFINED — no heterogeneity captured\n\n")
    next
  }
  
  rank_corr = cor(est, tau_x, method = "spearman")
  quality   = ifelse(rank_corr > 0.80, "RELIABLE RANKING",
                     ifelse(rank_corr > 0.50, "MODERATE SIGNAL",
                            ifelse(rank_corr > 0.30, "WEAK SIGNAL",
                                   "ESSENTIALLY RANDOM")))
  
  cat("Learner:", name, "\n")
  cat("  Spearman correlation:", round(rank_corr, 4), "\n")
  cat("  Ranking quality:     ", quality, "\n\n")
}

cat("===========================================\n")
cat("MEASURE 3: SUBGROUP RECOVERY\n")
cat("===========================================\n")

true_sg1 = mean(tau_x[sg1])
true_sg2 = mean(tau_x[sg2])
true_sg3 = mean(tau_x[sg3])

cat("True mean tau by subgroup:\n")
cat("  SG1 (young/remote/digital):", round(true_sg1, 3), "\n")
cat("  SG2 (severe/complex):      ", round(true_sg2, 3), "\n")
cat("  SG3 (elderly/low digital): ", round(true_sg3, 3), "\n\n")

for (name in names(learners_all)) {
  est = learners_all[[name]]
  
  est_sg1 = mean(est[sg1])
  est_sg2 = mean(est[sg2])
  est_sg3 = mean(est[sg3])
  
  rec_sg1 = est_sg1 / true_sg1
  rec_sg2 = est_sg2 / true_sg2
  rec_sg3 = ifelse(abs(true_sg3) < 0.1, NA, est_sg3 / true_sg3)
  
  correct_sign_sg1 = sign(est_sg1) == sign(true_sg1)
  correct_sign_sg2 = sign(est_sg2) == sign(true_sg2)
  
  cat("Learner:", name, "\n")
  cat("  SG1 estimated:", round(est_sg1, 3),
      "| true:", round(true_sg1, 3),
      "| recovery:", round(rec_sg1, 3),
      "| correct sign:", correct_sign_sg1, "\n")
  cat("  SG2 estimated:", round(est_sg2, 3),
      "| true:", round(true_sg2, 3),
      "| recovery:", round(rec_sg2, 3),
      "| correct sign:", correct_sign_sg2, "\n")
  cat("  SG3 estimated:", round(est_sg3, 3),
      "| true:", round(true_sg3, 3),
      "| recovery:", ifelse(is.na(rec_sg3), "NA (near zero)",
                            as.character(round(rec_sg3, 3))), "\n\n")
}

# ============================================================
# SUMMARY TABLE — all learners and baselines side by side
# ============================================================

cat("===========================================\n")
cat("FULL SUMMARY TABLE\n")
cat("===========================================\n")

summary_rows = list()

for (name in names(learners_all)) {
  est      = learners_all[[name]]
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  
  rank_corr = ifelse(sd(est) < 1e-10, NA,
                     cor(est, tau_x, method = "spearman"))
  
  est_sg1  = mean(est[sg1])
  est_sg2  = mean(est[sg2])
  rec_sg1  = round(est_sg1 / true_sg1, 3)
  rec_sg2  = round(est_sg2 / true_sg2, 3)
  
  sign_sg1 = sign(est_sg1) == sign(true_sg1)
  sign_sg2 = sign(est_sg2) == sign(true_sg2)
  
  summary_rows[[name]] = data.frame(
    Learner    = name,
    Norm_MSE   = round(norm_mse, 3),
    Rank_Corr  = ifelse(is.na(rank_corr), "NA", round(rank_corr, 3)),
    Rec_SG1    = rec_sg1,
    Rec_SG2    = rec_sg2,
    Sign_SG1   = sign_sg1,
    Sign_SG2   = sign_sg2
  )
}

summary_table = do.call(rbind, summary_rows)
print(summary_table, row.names = FALSE)

cat("\nInterpretation guide:\n")
cat("  Norm_MSE  < 1.0 means better than zero predictor\n")
cat("  Rank_Corr > 0.5 means meaningful individual ranking\n")
cat("  Rec_SG1/2 close to 1.0 means correct subgroup magnitude\n")
cat("  Sign_SG1/2 = TRUE means correct direction of effect\n")
