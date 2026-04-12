# ============================================================
# SMOKING AND LONGEVITY — SINGLE ITERATION
# Outcome: 10-year lung function decline score (higher = worse decline)
# Treatment: Heavy long-term smoking (W = 1) vs non-heavy smoker (W = 0)

# Core challenge being tested:
#   Severe treatment imbalance — only ~7.5% of the cohort are heavy smokers.
#   With so few treated units, learners that rely on direct comparison between treated and control patients struggle because there are very few treated individuals to learn from. 
#   The R-learner's propensity residualisation should give it a structural advantage here — it explicitly accounts for the rarity of treatment in its loss function, de-weighting observations where treatment probability is very low.

# Confounding structure:
#   Low income and high stress both increase smoking probability AND independently worsen long-run health outcomes. 
#   A naive learner will see that low-income patients have worse outcomes and may incorrectly attribute some of that to the smoking — overstating the treatment effect.
#   A well-calibrated learner separates the two channels.
# ============================================================

# install.packages("devtools")
# install_github("xnie/rlearner")
library(devtools) 
library(MASS)       # mvrnorm — generates correlated multivariate normal data
library(rlearner)   # rlasso, rboost and other meta-learners

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
# After transforming to realistic scales (via pnorm and rounding), the observed sample correlations will be approximately but not exactly equal to the designed values.
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

# Clinical / social logic:
#   Lower income strongly raises heavy-smoking probability
#   Higher stress strongly raises heavy-smoking probability
#   Lower health literacy raises smoking probability
#   Older people are slightly more likely to be heavy smokers historically
#   Poorer baseline lungs may also be associated with smoking selection

# Target treated proportion: around 5-10%
# This is the core challenge of this setup: with n=200, we expect roughly 14-16 treated units. Every learner has very few treated observations to learn from.
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
  0.7  * x1_s         +   # age: older = marginally more likely to be heavy smoker
  1.5  * x2_lowinc_s  +   # income: strongest driver — low SES strongly predicts smoking
  0.4  * x3_poorlung_s+   # poor lungs: some selection effect (sick people smoke more)
  0.9  * x4_lowlit_s  +   # literacy: low awareness = higher smoking uptake
  1.3  * x5_s             # stress: high stress = strong predictor of heavy smoking

# Find the intercept that makes average propensity = 7.5%
# uniroot finds the value of 'a' such that mean(plogis(a + linear_pred)) = 0.075
# This is more reliable than guessing an intercept value manually
target_prev = 0.075

intercept_fn = function(a) {
  mean(plogis(a + linpred_no_intercept)) - target_prev
}

alpha = uniroot(intercept_fn, interval = c(-10, 0))$root

log_odds_w = alpha + linpred_no_intercept
propensity = plogis(log_odds_w)

# Clip lightly for numerical stability but keep treatment rare
propensity = pmax(0.01, pmin(propensity, 0.20))
# Note for above: Even the highest-risk individuals (old, low-income, high-stress, low-literacy all at once) are capped at 20% probability of being a heavy smoker. 
# This keeps the treatment genuinely rare across the entire covariate space and is clinically realistic — even in the most vulnerable groups, heavy smoking remains a minority.
# Learners will not know about this cap. Their estimated propensity model may predict values above 0.20 for some patients, which is a mild structural misspecification inherent to this setup.

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


# Confounding check — do treated and control groups differ on key covariates?
# We expect treated to be older, lower income, more stressed, less literate
cat("\nCovariate means by treatment arm (confounding check):\n")
cat("  Age        — treated:", round(mean(x1_age[w==1]),     1),
    "| control:", round(mean(x1_age[w==0]),     1), "\n")
cat("  Income     — treated:", round(mean(x2_income[w==1]),  1),
    "| control:", round(mean(x2_income[w==0]),  1), "\n")
cat("  Stress     — treated:", round(mean(x5_stress[w==1]),  2),
    "| control:", round(mean(x5_stress[w==0]),  2), "\n")
cat("  Literacy   — treated:", round(mean(x4_literacy[w==1]),1),
    "| control:", round(mean(x4_literacy[w==0]),1), "\n")
cat("  Lung score — treated:", round(mean(x3_lung[w==1]),    1),
    "| control:", round(mean(x3_lung[w==0]),    1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)

# Interpretation:
#   tau(X) is the extra 10-year lung-function decline caused by heavy smoking, above and beyond what would have happened anyway (which is captured by b(X) below)
#   Higher tau = more harmful = more decline caused by smoking

# Clinical logic:
#   Older people are harmed more
#   People with poorer baseline lungs are harmed more
#   Higher stress worsens smoking damage
#   Lower literacy is associated with worse smoking management / prevention

#   Income (X2) does NOT appear in tau because it only shapes who smokes (propensity) and baseline health (b_x) but not the causal mechanism of how smoking damages lungs.
#   This creates a confounding trap for naive learners: they see that low-income people have worse outcomes overall and may mistakenly inflate their estimated treatment effect for low-income patients.

# Unlike the telemedicine file, there is NO sign change here: smoking is always harmful, though the amount of harm varies by person
# ============================================================

tau_x = 2.0 +
  1.1 * x1_s          +   # older = harmed more (lung plasticity declines with age)
  1.4 * x3_poorlung_s +   # poorer baseline lungs = harmed most (largest coefficient)
  0.8 * x5_s          +   # higher stress = amplifies smoking damage
  0.5 * x4_lowlit_s       # lower literacy = less able to mitigate harm
# no income as mentioned above

# Enforce strictly harmful treatment effect
# Without this floor, the formula produces a few very low values (close to 0 or small positive) for the healthiest young patients.
# We floor at 0.5 to enforce "smoking always causes at least some measurable harm" — no patient has a zero or negative treatment effect.
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

# b(X) is untreated 10-year lung function decline before adding the smoking effect. It captures natural ageing, socioeconomic disadvantage, and chronic stress effects on lung health.
# Higher b(X) = worse decline even without heavy smoking

# Logic:
#   Older age worsens decline
#   Lower income worsens decline
#   Poorer baseline lungs worsen decline
#   Lower literacy worsens disease management
#   Higher stress worsens long-run outcomes
#   An age x poor-lung interaction adds extra risk for vulnerable patients

# A naive learner looking at raw outcomes will see smokers are worse off and correctly attribute some of this to smoking, but will incorrectly attribute some of the b(X) gap to smoking as well — overestimating tau.
# ============================================================

b_x = 18 +
  3.5 * x1_s          +      # older age worsens natural lung decline
  3.0 * x2_lowinc_s   +      # lower income = worse long-run outcomes (confounding)
  5.0 * x3_poorlung_s +      # poor baseline function = faster decline regardless
  2.0 * x4_lowlit_s   +      # low literacy = worse disease management
  4.0 * x5_s          +      # high stress = independently worsens lung outcomes
  1.5 * x1_s * x3_poorlung_s # interaction: old + bad lungs = extra steep decline
# this nonlinearity makes nuisance estimation harder

cat("\n=== BASELINE OUTCOME b(X) SUMMARY ===\n")
cat("Mean b(X):", round(mean(b_x), 2),
    "(average decline without heavy smoking)\n")
cat("Range b(X):", round(min(b_x), 2),
    "to", round(max(b_x), 2), "\n")
cat("SD b(X):", round(sd(b_x), 2),
    "— should be larger than SD of tau to make confounding hard\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y

# Y = b(X) + W * tau(X) + epsilon

# Important:
# Here b(X) is interpreted as the untreated baseline outcome.
# So we use W * tau(X), not (W - propensity) * tau(X).
# The propensity affects who ends up in each group but NOT how much harm smoking causes to a treated patient.

# Noise: SD = 3 reflects measurement error and unmeasured factors (diet, genetics, air quality, etc.).
# SD of 3 is modest relative to the signal (mean tau ≈ 2, b_x SD ≈ 5+) making the problem learnable but not trivially easy.
# ============================================================

sigma   = 3
epsilon = rnorm(n, mean = 0, sd = sigma)

y = b_x + w * tau_x + epsilon

cat("\n=== OBSERVED OUTCOME Y SUMMARY ===\n")
cat("Mean Y (all):", round(mean(y), 2), "\n")
cat("Mean Y (treated, W=1):", round(mean(y[w==1]), 2), "\n")
cat("Mean Y (control, W=0):", round(mean(y[w==0]), 2), "\n")
cat("Naive treated-control difference:",
    round(mean(y[w==1]) - mean(y[w==0]), 3),
    " (confounded, not the true ATE)\n")
cat("True ATE (mean tau):", round(mean(tau_x), 3), "\n")
cat("\nThe naive gap overstates true ATE because smokers have worse b(X) on average (lower income, higher stress) — this is the confounding that learners must correct for.\n")

# ============================================================
# SECTION 6: DEFINE SUBGROUPS BEFORE FITTING

# SG1 — lower-risk subgroup, should still be harmed but less so younger, high lung function, low stress, high literacy (removed income from before here since not a driver in tau)

# SG3 — socially disadvantaged subgroup because have low income and low literacy (confounding trap). This is the subgroup that tests whether learners separate confounding from causal effects.

# Desired ordering: SG2 > SG3 > SG1
#   SG2 has highest true tau (biologically most vulnerable)
#   SG3 has moderate true tau but will tempt naive learners to overestimate
#   SG1 has lowest true tau (biologically most resilient)
# ============================================================

# Fixed clinical thresholds — same every run regardless of sample
sg1 = x1_age    <  55 &    # younger working-age adult
  x3_lung    >  75 &    # good baseline lung function
  x5_stress  <   4 &    # low stress
  x4_literacy > 14      # health-literate

sg2 = x1_age    >  70 &    # elderly
  x3_lung    <  55 &    # clinically impaired lung function
  x5_stress  >   7      # high chronic stress

sg3 = x2_income <  40 &    # low income (bottom of distribution)
  x4_literacy <   5      # low health literacy

cat("\n=== SUBGROUP SIZES ===\n")
cat("Subgroup 1 (younger, good-lung, low-stress, literate):", sum(sg1), "\n")
cat("Subgroup 2 (older, poor-lung, high-stress):", sum(sg2), "\n")
cat("Subgroup 3 (low-income, low-literacy):", sum(sg3), "\n")

cat("\n=== TRUE TAU BY SUBGROUP ===\n")
cat("SG1 mean true tau:", round(mean(tau_x[sg1]), 3),
    "(expect lowest — biologically resilient)\n")
cat("SG2 mean true tau:", round(mean(tau_x[sg2]), 3),
    "(expect highest — biologically vulnerable)\n")
cat("SG3 mean true tau:", round(mean(tau_x[sg3]), 3),
    "(expect moderate — income/literacy don't drive tau)\n")
cat("\nTarget ordering: SG2 > SG3 > SG1\n")

# ============================================================
# SECTION 7: FIT LEARNERS
# Important: only rlasso + rboost in this version
# ============================================================

cat("\n=== FITTING LEARNERS ===\n")
cat("Fitting rlasso...\n")
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

cat("Fitting rboost...\n")
rboost_fit = rboost(x, w, y, verbose = TRUE)
rboost_est = predict(rboost_fit, x)

learners = list(
  rlasso = rlasso_est,
  rboost = rboost_est
)

# ============================================================
# SECTION 8: EVALUATION

# Metric 1: Normalised MSE
#   Raw MSE divided by variance of true tau.
#   A value of 1.0 means the learner is no better than predicting the mean tau for everyone (zero predictor benchmark).
#   Values below 1.0 are better than the naive baseline.
#   Values below 0.25 are excellent; 0.25-0.75 acceptable; above 1.0 is failure.

# Metric 2: Rank correlation (Spearman)
#   Does the learner correctly rank patients by how harmed they are?
#   Even if absolute values are off (common under treatment imbalance), a learner may still correctly identify that Patient A benefits less than Patient B. Rank correlation above 0.5 means meaningful signal.

# Metric 3: Subgroup recovery
#   Does the learner correctly recover tau within each subgroup?
#   Recovery ratio = estimated subgroup mean / true subgroup mean.
#   Close to 1.0 means correct magnitude. Below 1.0 means attenuation.
#   Key check for SG3: does the learner overestimate due to confounding?
# ============================================================

tau_variance = var(tau_x)

cat("\n=== MEASURE 1: NORMALISED MSE ===\n")
cat("Variance of true tau(X):", round(tau_variance, 4), "\n")
cat("(Normalised MSE = 1.0 means no better than predicting mean tau)\n\n")

for (name in names(learners)) {
  est      = learners[[name]]
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  
  quality  = ifelse(norm_mse < 0.25, "EXCELLENT",
                    ifelse(norm_mse < 0.75, "ACCEPTABLE",
                           ifelse(norm_mse < 1.00, "POOR",
                                  "WORSE THAN MEAN")))
  
  cat("Learner:", name, "\n")
  cat("  Raw MSE:        ", round(raw_mse, 4), "\n")
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

cat("\n=== MEASURE 3: SUBGROUP RECOVERY ===\n")

true_sg1 = mean(tau_x[sg1])
true_sg2 = mean(tau_x[sg2])
true_sg3 = mean(tau_x[sg3])

cat("True mean tau by subgroup:\n")
cat("  SG1 (lower risk):            ", round(true_sg1, 3), "\n")
cat("  SG2 (older/poor-lung/stress):", round(true_sg2, 3), "\n")
cat("  SG3 (low-income/low-literacy):", round(true_sg3, 3), "\n\n")

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

# Plot A: Propensity score distribution by treatment arm
#   This is unique to this setup — shows visually how extreme the imbalance is. Treated patients have higher propensity but even they are mostly below 0.20.

# Plot B: True tau vs covariate plots predicted by rboost and rlasso 
# ============================================================

par(mfrow = c(1, 1))
# --- Plot A: Propensity score histogram ---
hist(propensity[w == 0], breaks = 20,
     col  = rgb(0.2, 0.4, 0.8, 0.5),
     xlim = c(0, 0.22),
     xlab = "Propensity score",
     ylab = "Count",
     main = "Propensity scores by treatment arm\n(note: max capped at 0.20)")
hist(propensity[w == 1], breaks = 10,
     col  = rgb(0.9, 0.2, 0.2, 0.5),
     add  = TRUE)
legend("topright",
       legend = c("Control (W=0)", "Treated (W=1)"),
       fill   = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.9, 0.2, 0.2, 0.5)),
       cex    = 0.8)
abline(v = mean(propensity), col = "black", lty = 2)



# --- Plot B: True tau vs covariate predicted by rboost and rlasso ---
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