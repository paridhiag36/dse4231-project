# ============================================================
# SMOKING SETUP — SINGLE ITERATION
# Outcome: 10-year lung function decline (higher = worse)
# Treatment: Heavy long-term smoking (W = 1) vs not heavy smoker (W = 0)
# Key feature: treatment is rare (target 5-10%) and always harmful
# Confounding: low income and high stress increase both smoking and worse outcomes
# ============================================================
# install.packages("remotes")
# install.packages(c("xgboost", "stringr", "glmnet", "caret"))
# remotes::install_github("xnie/rlearner")
library(rlearner)


library(MASS)       # for mvrnorm — correlated covariate generation
library(rlearner)   # for rlasso and rboost

set.seed(42)
n = 200

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: age (years)                       — range 40 to 85
# X2: income / socioeconomic status     — range 20 to 150 (thousand units)
# X3: baseline lung function score      — range 40 to 110
# X4: health literacy / awareness       — range 0 to 20
# X5: stress index                      — range 0 to 10
#
# Correlation structure:
#   age <-> baseline lung:      -0.35  older people tend to have poorer lungs
#   income <-> literacy:        +0.50  higher SES linked to higher health literacy
#   income <-> stress:          -0.30  lower SES linked to higher stress
#   literacy <-> stress:        -0.35  low literacy tends to coexist with stress
#   age <-> literacy:           -0.20  older cohorts slightly less health-literate
#   age <-> stress:             +0.15  modest positive relation
# ============================================================

cor_matrix = matrix(c(
  # X1_age  X2_income  X3_lung  X4_literacy  X5_stress
  1.00,    -0.10,     -0.35,    -0.20,       0.15,   # X1 age
  -0.10,     1.00,      0.15,     0.50,      -0.30,   # X2 income
  -0.35,     0.15,      1.00,     0.20,      -0.25,   # X3 baseline lung
  -0.20,     0.50,      0.20,     1.00,      -0.35,   # X4 literacy
  0.15,    -0.30,     -0.25,    -0.35,       1.00    # X5 stress
), nrow = 5, byrow = TRUE)

# Generate correlated standard normal data
z = mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

# Transform to realistic scales

# X1: age — 40 to 85 years
x1_age = round(40 + 45 * pnorm(z[,1]))

# X2: income / SES — 20 to 150
# Think of this as annual income in thousands
x2_income = 20 + 130 * pnorm(z[,2])

# X3: baseline lung function score — 40 to 110
# Higher means better baseline lung function
x3_lung = 40 + 70 * pnorm(z[,3])

# X4: health literacy — 0 to 20
x4_literacy = round(20 * pnorm(z[,4]))

# X5: stress index — 0 to 10
x5_stress = 10 * pnorm(z[,5])

# Combine into matrix for rlearner
x = cbind(x1_age, x2_income, x3_lung, x4_literacy, x5_stress)
colnames(x) = c("age", "income", "lung_baseline", "health_literacy", "stress")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))

cat("\n=== COVARIATE CORRELATION (should reflect design above) ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
#
# Clinical / social logic:
#   Lower income strongly raises heavy-smoking probability
#   Higher stress strongly raises heavy-smoking probability
#   Lower health literacy raises smoking probability
#   Older people are slightly more likely to be heavy smokers historically
#   Poorer baseline lungs may also be associated with smoking selection
#
# Target treated proportion: around 5-10%
# ============================================================

# Standardise risk-oriented versions of the covariates
# Positive values should mean "more smoking risk"
x1_s = as.numeric(scale(x1_age))          # older = more risk
x2_lowinc_s = as.numeric(scale(-x2_income))   # lower income = more risk
x3_poorlung_s = as.numeric(scale(-x3_lung))   # poorer lungs = more risk
x4_lowlit_s = as.numeric(scale(-x4_literacy)) # lower literacy = more risk
x5_s = as.numeric(scale(x5_stress))       # higher stress = more risk

# Build propensity score without intercept first
linpred_no_intercept =
  0.7 * x1_s +
  1.5 * x2_lowinc_s +
  0.4 * x3_poorlung_s +
  0.9 * x4_lowlit_s +
  1.3 * x5_s

# Calibrate intercept so average propensity is around 7.5%
target_prev = 0.075

intercept_fn = function(a) {
  mean(plogis(a + linpred_no_intercept)) - target_prev
}

alpha = uniroot(intercept_fn, interval = c(-10, 0))$root

log_odds_w = alpha + linpred_no_intercept
propensity = plogis(log_odds_w)

# Clip lightly for numerical stability but keep treatment rare
propensity = pmax(0.01, pmin(propensity, 0.20))

# Draw treatment until realised sample share is within 5-10%
repeat {
  w = rbinom(n, 1, propensity)
  treated_share = mean(w)
  if (treated_share >= 0.05 && treated_share <= 0.10) break
}

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion heavy smokers:", round(mean(w), 3), "\n")
cat("Propensity score range:",
    round(min(propensity), 3), "to", round(max(propensity), 3), "\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")

cat("\nMean age           — treated:", round(mean(x1_age[w==1]), 1),
    "| control:", round(mean(x1_age[w==0]), 1), "\n")
cat("Mean income        — treated:", round(mean(x2_income[w==1]), 1),
    "| control:", round(mean(x2_income[w==0]), 1), "\n")
cat("Mean stress        — treated:", round(mean(x5_stress[w==1]), 2),
    "| control:", round(mean(x5_stress[w==0]), 2), "\n")
cat("Mean literacy      — treated:", round(mean(x4_literacy[w==1]), 1),
    "| control:", round(mean(x4_literacy[w==0]), 1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)
#
# Interpretation:
#   tau(X) is the extra 10-year lung-function decline caused by heavy smoking
#   Higher tau = more harmful
#
# Clinical logic:
#   Older people are harmed more
#   People with poorer baseline lungs are harmed more
#   Higher stress worsens smoking damage
#   Lower literacy is associated with worse smoking management / prevention
#
# Unlike the telemedicine file, there is NO sign change here:
# smoking is always harmful, though the amount of harm varies by person
# ============================================================

tau_x = 2.0 +
  1.1 * x1_s +
  1.4 * x3_poorlung_s +
  0.8 * x5_s +
  0.5 * x4_lowlit_s

# Enforce strictly harmful treatment effect
tau_x = pmax(0.5, tau_x)

cat("\n=== TRUE TREATMENT EFFECT SUMMARY ===\n")
cat("Mean tau(X):", round(mean(tau_x), 3), "\n")
cat("SD of tau(X):", round(sd(tau_x), 3), "\n")
cat("Range of tau(X):", round(min(tau_x), 3),
    "to", round(max(tau_x), 3), "\n")
cat("Proportion with tau > 0:", round(mean(tau_x > 0), 3),
    "(should be 1.000 or very close)\n")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
#
# b(X) is untreated 10-year lung function decline
# before adding the smoking effect
#
# Higher b(X) = worse decline even without heavy smoking
#
# Logic:
#   Older age worsens decline
#   Lower income worsens decline
#   Poorer baseline lungs worsen decline
#   Lower literacy worsens disease management
#   Higher stress worsens long-run outcomes
#   An age x poor-lung interaction adds extra risk for vulnerable patients
# ============================================================

b_x = 18 +
  3.5 * x1_s +
  3.0 * x2_lowinc_s +
  5.0 * x3_poorlung_s +
  2.0 * x4_lowlit_s +
  4.0 * x5_s +
  1.5 * x1_s * x3_poorlung_s

cat("\n=== BASELINE OUTCOME b(X) SUMMARY ===\n")
cat("Mean b(X):", round(mean(b_x), 2), "\n")
cat("Range b(X):", round(min(b_x), 2),
    "to", round(max(b_x), 2), "\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
#
# Y = b(X) + W * tau(X) + epsilon
#
# Important:
# Here b(X) is interpreted as the untreated baseline outcome.
# So we use W * tau(X), not (W - propensity) * tau(X).
#
# Noise: SD = 3 reflects measurement error and unmeasured factors
# ============================================================

sigma   = 3
epsilon = rnorm(n, mean = 0, sd = sigma)

y = b_x + w * tau_x + epsilon

for (name in names(learners)) {
  est = learners[[name]]
  
  est_sg1 = mean(est[sg1])
  est_sg2 = mean(est[sg2])
  est_sg3 = mean(est[sg3])
  
  rec_sg1 = est_sg1 / true_sg1
  rec_sg2 = est_sg2 / true_sg2
  rec_sg3 = est_sg3 / true_sg3
  
  correct_order = (est_sg2 > est_sg3) & (est_sg3 > est_sg1)
  
  cat("Learner:", name, "\n")
  cat("  SG1 estimated:", round(est_sg1, 3),
      "| true:", round(true_sg1, 3),
      "| recovery:", round(rec_sg1, 3), "\n")
  cat("  SG2 estimated:", round(est_sg2, 3),
      "| true:", round(true_sg2, 3),
      "| recovery:", round(rec_sg2, 3), "\n")
  cat("  SG3 estimated:", round(est_sg3, 3),
      "| true:", round(true_sg3, 3),
      "| recovery:", round(rec_sg3, 3), "\n")
  cat("  Correct subgroup severity ordering (SG2 > SG3 > SG1):",
      correct_order, "\n\n")
}

# ============================================================
# SECTION 9: PARTIAL RELATIONSHIP PLOTS
# One plot per covariate showing true vs estimated tau
# ============================================================

par(mfrow = c(2, 3))

covariate_names = c("age", "income", "lung_baseline",
                    "health_literacy", "stress")
covariate_vals  = list(x1_age, x2_income, x3_lung, x4_literacy, x5_stress)

for (j in 1:5) {
  
  cov_vals = covariate_vals[[j]]
  ord      = order(cov_vals)
  
  plot(cov_vals[ord], tau_x[ord],
       type = "l", col = "black", lwd = 2,
       xlab = covariate_names[j],
       ylab = "Treatment effect tau (decline points)",
       main = paste("tau vs", covariate_names[j]),
       ylim = range(c(tau_x, rlasso_est, rboost_est)))
  
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
