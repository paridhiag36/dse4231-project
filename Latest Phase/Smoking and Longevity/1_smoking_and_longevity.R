# ============================================================
# SMOKING AND LONGEVITY — SINGLE ITERATION (EXPLORATORY)
# Outcome: 10-year lung function decline (higher = worse)
# Treatment: Heavy long-term smoking (W=1) vs non-heavy smoker (W=0)

# Core challenge: severe treatment imbalance (~7.5% treated)
#   Only ~7.5% of the cohort are heavy smokers. With so few treated units, any learner that relies on direct treated/control comparison struggles. The R-learner's propensity residualisation should give it a structural advantage — it explicitly de-weights observations where treatment probability is very low.

# Confounding structure:
#   Low income and high stress both increase smoking probability AND independently worsen long-run health outcomes. A naive learner will conflate these channels and overstate the treatment effect.

# Learners in this file: rlasso and rboost only (exploratory version)
# Full learner comparison is in 2_smoking_and_longevity_all.R
# ============================================================

library(MASS)
library(rlearner)

set.seed(42)
n = 500   # change this single number to run for n=300, 500, or 1000 if needed (we didnt)

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: age (years)                       — range 40 to 85
# X2: income / SES                      — range 20 to 150 (thousands)
# X3: baseline lung function score      — range 40 to 110
# X4: health literacy / awareness       — range 0 to 20
# X5: stress index                      — range 0 to 10
#
# Correlation structure:
#   age    <-> lung:     -0.35  older patients have lower baseline function
#   income <-> literacy: +0.50  higher SES strongly linked to health literacy
#   income <-> stress:   -0.30  lower SES linked to higher chronic stress
#   literacy <-> stress: -0.35  low literacy coexists with high stress
#   age    <-> literacy: -0.20  older cohorts slightly less health-literate
#   age    <-> stress:   +0.15  modest positive — older adults carry more stress
# ============================================================

cor_matrix = matrix(c(
  1.00, -0.10, -0.35, -0.20,  0.15,
  -0.10,  1.00,  0.15,  0.50, -0.30,
  -0.35,  0.15,  1.00,  0.20, -0.25,
  -0.20,  0.50,  0.20,  1.00, -0.35,
  0.15, -0.30, -0.25, -0.35,  1.00
), nrow = 5, byrow = TRUE)

z = MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

x1_age      = round(40 + 45 * pnorm(z[,1]))
x2_income   = 20  + 130 * pnorm(z[,2])
x3_lung     = 40  +  70 * pnorm(z[,3])
x4_literacy = round(20  * pnorm(z[,4]))
x5_stress   =  10 * pnorm(z[,5])

x = cbind(x1_age, x2_income, x3_lung, x4_literacy, x5_stress)
colnames(x) = c("age", "income", "lung_baseline", "health_literacy", "stress")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))
cat("\n=== COVARIATE CORRELATION ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT

# Smoking uptake is driven purely by socioeconomic and behavioural factors — income, stress, literacy, age. Baseline lung function is deliberately excluded from propensity: it is a physiological state that determines how much smoking damages you (tau), but is not a meaningful driver of whether you start smoking.
#
# Propensity clipped at 0.20: even the highest-risk patients are capped at 20% — heavy smoking is a minority behaviour even among the most deprived.
# Learners do not know about this cap, introducing mild propensity misspecification inherent to this setup.
# ============================================================

x1_s        = as.numeric(scale(x1_age))
x2_lowinc_s = as.numeric(scale(-x2_income))   # flipped: lower income = higher risk
x4_lowlit_s = as.numeric(scale(-x4_literacy)) # flipped: lower literacy = higher risk
x5_s        = as.numeric(scale(x5_stress))

# Lung function excluded from propensity 
linpred_no_intercept =
  0.7 * x1_s        +   # age: older = marginally more likely to be heavy smoker
  1.5 * x2_lowinc_s +   # income: strongest driver — low SES predicts smoking
  0.9 * x4_lowlit_s +   # literacy: low awareness raises smoking uptake
  1.3 * x5_s             # stress: strong predictor of heavy smoking

target_prev  = 0.075
intercept_fn = function(a) mean(plogis(a + linpred_no_intercept)) - target_prev
alpha        = uniroot(intercept_fn, interval = c(-10, 0))$root

propensity = pmax(0.01, pmin(plogis(alpha + linpred_no_intercept), 0.20))

repeat {
  w = rbinom(n, 1, propensity)
  if (mean(w) >= 0.05 && mean(w) <= 0.10) break
}

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion heavy smokers (target 5-10%):", round(mean(w), 3), "\n")
cat("Number treated:", sum(w), "| Number control:", sum(1-w), "\n")
cat("Propensity range:", round(min(propensity), 3),
    "to", round(max(propensity), 3), "(capped at 0.20)\n")

cat("\nConfounding check:\n")
cat("  Age      — treated:", round(mean(x1_age[w==1]),    1),
    "| control:", round(mean(x1_age[w==0]),    1), "\n")
cat("  Income   — treated:", round(mean(x2_income[w==1]), 1),
    "| control:", round(mean(x2_income[w==0]), 1), "\n")
cat("  Stress   — treated:", round(mean(x5_stress[w==1]), 2),
    "| control:", round(mean(x5_stress[w==0]), 2), "\n")
cat("  Literacy — treated:", round(mean(x4_literacy[w==1]),1),
    "| control:", round(mean(x4_literacy[w==0]),1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)

# tau(X) = extra lung function decline CAUSED BY heavy smoking, above and beyond natural decline captured by b(X).
# Higher tau = more harmful. Always positive (floor rarely binds).

# Biological drivers of harm:
#   Poor baseline lung (1.4): less reserve capacity, faster acceleration
#   Older age (1.1):           less plastic lungs, slower recovery
#   High stress (0.8):         amplifies inflammatory damage
#   Low literacy (0.5):        less able to mitigate through behaviour change

# Income excluded from tau: income affects BASELINE health trajectory (via healthcare access, nutrition) captured in b(X), but conditional on lung function and age — the direct physiological determinants — income does not add independent causal pathways for smoking damage.
# This deliberate exclusion creates the confounding trap for SG3.

# Constant raised to 3.0 (from 2.0) so the pmax(0.5) floor rarely binds — even the healthiest patient has tau ~1.5 before the floor.
# ============================================================

x3_poorlung_s = as.numeric(scale(-x3_lung))  # flipped: poorer = higher

tau_x = 3.0 +
  1.1 * x1_s          +   # older = harmed more
  1.4 * x3_poorlung_s +   # poorer lungs = harmed most
  0.8 * x5_s          +   # higher stress = amplifies damage
  0.5 * x4_lowlit_s       # lower literacy = less able to mitigate

tau_x = pmax(0.5, tau_x)   # floor: smoking always causes at least some harm

cat("\n=== TRUE TREATMENT EFFECT SUMMARY ===\n")
cat("Mean tau(X):", round(mean(tau_x), 3), "\n")
cat("SD tau(X):  ", round(sd(tau_x),   3), "\n")
cat("Range:      ", round(min(tau_x),  3), "to", round(max(tau_x), 3), "\n")
cat("% hitting floor (tau = 0.5):", round(mean(tau_x == 0.5)*100, 1), "%\n")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)

# b(X) = lung function decline WITHOUT heavy smoking.
# Includes a nonlinear age x lung interaction to make nuisance estimation harder — tests learners' ability to separate b(X) from tau.

# Confounding lives here: low income and high stress raise b(X) AND raise smoking probability. Smokers therefore have worse b(X) on average.
# A naive learner conflates bad b(X) with large tau.
# ============================================================

x2_lowinc_s_b = as.numeric(scale(-x2_income))  # same flip as propensity

b_x = 18 +
  3.5 * x1_s              +   # age worsens natural decline
  3.0 * x2_lowinc_s_b     +   # lower income = worse trajectory (confounding)
  5.0 * x3_poorlung_s     +   # poor baseline = faster natural decline
  2.0 * x4_lowlit_s       +   # low literacy = worse disease management
  4.0 * x5_s              +   # high stress = independently worsens outcomes
  1.5 * x1_s * x3_poorlung_s  # interaction: old + bad lungs = extra steep decline

cat("\n=== BASELINE OUTCOME b(X) ===\n")
cat("Mean b(X):", round(mean(b_x), 2), "\n")
cat("SD b(X):  ", round(sd(b_x),   2), "\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
# Y = b(X) + W * tau(X) + epsilon
# ============================================================

sigma   = 3
epsilon = rnorm(n, mean = 0, sd = sigma)
y       = b_x + w * tau_x + epsilon

cat("\n=== OBSERVED OUTCOME Y ===\n")
cat("Mean Y (all):   ", round(mean(y),       2), "\n") # 18.61
cat("Mean Y treated: ", round(mean(y[w==1]), 2), "\n") # 34.69
cat("Mean Y control: ", round(mean(y[w==0]), 2), "\n") # 17.62
cat("Naive ATE:      ", round(mean(y[w==1]) - mean(y[w==0]), 3),  # 17.066 (has confounding)
    "(confounded)\n")
cat("True ATE:       ", round(mean(tau_x), 3), "\n") # 3.203

# ============================================================
# SECTION 6: SUBGROUPS

# Three subgroups designed to test different aspects of the setup:

# SG1 — Biologically resilient (expect LOWEST tau)
#   age < 60:   working-age adult, lungs still plastic
#   lung > 75:  good baseline function (upper third of 40-110)
#   stress < 4: low chronic stress
#   Clinical story: 48-year-old with healthy lungs and low stress —smoking still harms them but their reserve limits the damage.

# SG2 — Biologically vulnerable (expect HIGHEST tau)
#   age > 68:   elderly
#   lung < 58:  clinically impaired baseline function
#   stress > 6: chronically stressed
#   Clinical story: 72-year-old with compromised lungs and high stress —smoking causes maximum damage, little physiological reserve left.

# SG3 — Socially disadvantaged / confounding trap (expect MODERATE tau)
#   income < 45: low income (bottom ~15% of distribution)
#   literacy < 6: low health literacy
#   Key test: income and literacy are NOT in tau(X). These patients have bad b(X) AND high propensity, tempting naive learners to overestimate their tau. A well-calibrated learner should estimate moderate tau here, not inflate it due to confounding.

# Desired ordering: SG2 > SG3 > SG1
# ============================================================

sg1 = x1_age    <  60 &
  x3_lung   >  75 &
  x5_stress <   4

sg2 = x1_age    >  68 &
  x3_lung   <  58 &
  x5_stress >   6

sg3 = x2_income <  45 &
  x4_literacy < 6

cat("\n=== SUBGROUP SIZES ===\n")
cat("SG1 (biologically resilient — low harm):      ", sum(sg1), "\n") # 70 people
cat("SG2 (biologically vulnerable — high harm):    ", sum(sg2), "\n") # 36 people
cat("SG3 (socially disadvantaged — confounding):   ", sum(sg3), "\n") # 61 people

cat("\n=== TRUE TAU BY SUBGROUP ===\n")
cat("SG1:", round(mean(tau_x[sg1]), 3), "(expect lowest)\n") # SG1: 0.599 (expect lowest)
cat("SG2:", round(mean(tau_x[sg2]), 3), "(expect highest)\n") # SG2: 7.413 (expect highest)
cat("SG3:", round(mean(tau_x[sg3]), 3), "(expect moderate)\n") # SG3: 4.655 (expect moderate)
cat("Ordering SG2 > SG3 > SG1:",
    mean(tau_x[sg2]) > mean(tau_x[sg3]) &
      mean(tau_x[sg3]) > mean(tau_x[sg1]), "\n")

# ============================================================
# SECTION 7: FIT LEARNERS (rlasso and rboost only)
# ============================================================

cat("\n=== FITTING LEARNERS ===\n")

cat("Fitting rlasso...\n")
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

cat("Fitting rboost...\n")
rboost_fit = rboost(x, w, y, verbose = FALSE)
rboost_est = predict(rboost_fit, x)

# ============================================================
# SECTION 8: EVALUATION
# ============================================================

tau_variance = var(tau_x)

zero_pred  = rep(0, n)                                        # floor: assumes no treatment effect
const_pred = rep(mean(y[w==1]) - mean(y[w==0]), n)           # naive confounded ATE (for full sample)

cat("\n=== NORMALISED MSE ===\n")
cat("(< 1.0 means better than predicting mean tau for everyone)\n\n")

for (name in c("rlasso", "rboost", "zero_pred", "const_pred")) {
  est      = switch(name, rlasso=rlasso_est, rboost=rboost_est, zero_pred=zero_pred, const_pred=const_pred)
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  quality  = ifelse(norm_mse < 0.25, "EXCELLENT",
                    ifelse(norm_mse < 0.75, "ACCEPTABLE",
                           ifelse(norm_mse < 1.00, "POOR", "WORSE THAN MEAN")))
  cat(sprintf("%-12s  NormMSE: %.4f  (%s)\n", name, norm_mse, quality))
}

cat("\n=== RANK CORRELATION ===\n")
for (name in c("rlasso", "rboost")) {
  est       = switch(name, rlasso=rlasso_est, rboost=rboost_est)
  if (sd(est) < 1e-10) {
    cat(sprintf("%-12s  RankCorr: NA (constant prediction)\n", name))
  } else {
    rc = cor(est, tau_x, method = "spearman")
    quality = ifelse(rc > 0.80, "RELIABLE",
                     ifelse(rc > 0.50, "MODERATE",
                            ifelse(rc > 0.30, "WEAK", "RANDOM")))
    cat(sprintf("%-12s  RankCorr: %.4f  (%s)\n", name, rc, quality))
  }
}

cat("\n=== SUBGROUP RECOVERY ===\n")
true_sg1 = mean(tau_x[sg1])
true_sg2 = mean(tau_x[sg2])
true_sg3 = mean(tau_x[sg3])
cat(sprintf("True tau:  SG1=%.3f  SG2=%.3f  SG3=%.3f\n\n",
            true_sg1, true_sg2, true_sg3))

for (name in c("rlasso", "rboost")) {
  est     = switch(name, rlasso=rlasso_est, rboost=rboost_est)
  e1 = mean(est[sg1]); e2 = mean(est[sg2]); e3 = mean(est[sg3])
  cat(sprintf("%-12s  SG1: est=%.3f rec=%.3f | SG2: est=%.3f rec=%.3f | SG3: est=%.3f rec=%.3f | order=%s | SG3_inflated=%s\n",
              name,
              e1, e1/true_sg1,
              e2, e2/true_sg2,
              e3, e3/true_sg3,
              ifelse((e2>e3)&(e3>e1), "CORRECT", "WRONG"),
              ifelse(e3/true_sg3 > 1.1, "YES", "no")))
}

# ============================================================
# SECTION 9: SAVE RESULTS TO CSV
# ============================================================

# Make sure working directory set to Rlearner setwd("~/DSE4231/Rlearner")
output_dir = "Latest Phase/Smoking and Longevity"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

results_df = data.frame(
  n            = n,
  seed         = 42,
  n_treated    = sum(w),
  treatment_rate = round(mean(w), 3),
  true_ate     = round(mean(tau_x), 4),
  tau_sd       = round(sd(tau_x), 4),
  true_sg1     = round(true_sg1, 4),
  true_sg2     = round(true_sg2, 4),
  true_sg3     = round(true_sg3, 4),
  sg1_n        = sum(sg1),
  sg2_n        = sum(sg2),
  sg3_n        = sum(sg3)
)

for (name in c("rlasso", "rboost")) {
  est = switch(name, rlasso=rlasso_est, rboost=rboost_est)
  raw_mse  = mean((est - tau_x)^2)
  norm_mse = raw_mse / tau_variance
  rc = if (sd(est) < 1e-10) NA else cor(est, tau_x, method = "spearman")
  e1 = mean(est[sg1]); e2 = mean(est[sg2]); e3 = mean(est[sg3])
  results_df[[paste0(name, "_norm_mse")]]  = round(norm_mse, 4)
  results_df[[paste0(name, "_rank_corr")]] = round(rc, 4)
  results_df[[paste0(name, "_rec_sg1")]]   = round(e1/true_sg1, 4)
  results_df[[paste0(name, "_rec_sg2")]]   = round(e2/true_sg2, 4)
  results_df[[paste0(name, "_rec_sg3")]]   = round(e3/true_sg3, 4)
  results_df[[paste0(name, "_order")]]     = (e2>e3)&(e3>e1)
  results_df[[paste0(name, "_sg3_inflated")]] = e3/true_sg3 > 1.1
}

csv_path = file.path(output_dir, paste0("smoking(only rlasso and rboost)_", n, ".csv"))
write.csv(results_df, csv_path, row.names = FALSE)
cat("\nResults saved to:", csv_path, "\n")

# ============================================================
# SECTION 10: ONE PLOT — Propensity histogram (unique to this setup)
# Shows visually how extreme the imbalance is
# ============================================================

plot_path = file.path(output_dir, paste0("propensity_hist_", n, ".png"))
  png(plot_path, width = 800, height = 500, res = 120)
  
  hist(propensity[w == 0], breaks = 20,
       col  = rgb(0.2, 0.4, 0.8, 0.5),
       xlim = c(0, 0.22),
       xlab = "Propensity score",
       ylab = "Count",
       main = paste0("Propensity by arm — n=", n,
                     " (", sum(w), " treated / ", sum(1-w), " control)"))
  hist(propensity[w == 1], breaks = 10,
       col  = rgb(0.9, 0.2, 0.2, 0.5), add = TRUE)
  legend("topright",
         legend = c(paste0("Control (n=", sum(w==0), ")"),
                    paste0("Treated (n=", sum(w==1), ")")),
         fill = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.9, 0.2, 0.2, 0.5)),
         cex = 0.8)
  abline(v = mean(propensity), col = "black", lty = 2)

dev.off()
cat("Plot saved to:", plot_path, "\n")

# ============================================================
# RESULTS ANALYSIS — n=500, seed=42, only rlasso+rboost
# ============================================================
# Setup context:
#   29 treated out of 500 total (5.8%) — 23 in training, 6 in test
#   True ATE = 3.20 | tau SD = 2.24
#   True tau by subgroup: SG1=0.599 (biologically resilient), SG2=7.413 (biologically vulnerable), SG3=4.655 (confounding trap)
#   SG sizes (full sample): SG1=70, SG2=36, SG3=61

# --- NORMALISED MSE ---
#   rlasso:     1.2300 — worse than mean prediction baseline (> 1.0)
#   rboost:     1.0708 — also above baseline, but closer to 1.0
#   zero_pred:  NormMSE = mean(tau_x)^2 / var(tau_x) >> 1 — far worse, tau is never zero
#   const_pred: NormMSE >> 1 — naive ATE ~17 vs true ATE ~3.2, massively inflated by confounding
#   Neither learner beats the mean predictor. Expected at this sample size: with only ~23 treated observations in training, the o(n^{-1/4}) convergence condition from Nie & Wager (2021) is not met.

# --- RANK CORRELATION ---
#   rlasso: 0.3251 (WEAK) — some signal but unreliable individual ranking
#   rboost: NA            — collapsed to a constant prediction (no variation)
#   rlasso at least produces variation in estimates across patients.
#   rboost predicts the same value for everyone — graceful degradation under imbalance but provides no heterogeneity information at all.

# --- SUBGROUP RECOVERY ---
#   SG1 (resilient — true tau=0.599):
#     rlasso: rec=6.2747 (massively inflated — true tau near pmax(0.5) floor)
#     rboost: rec=6.3600 (same issue)
#     SG1 recovery unreliable because true_sg1=0.599 is very close to the floor: any small estimation error in the numerator gets amplified when divided by ~0.599.
#   SG2 (vulnerable — true tau=7.413):
#     rlasso: rec=0.7392 — underestimates by ~26%, attenuated but correct sign
#     rboost: rec=0.5135 — underestimates by ~49%, more severely attenuated
#     Both learners correctly identify SG2 as high-harm but substantially underestimate magnitude.
#   SG3 (confounding trap — true tau=4.655):
#     rlasso: rec=0.7055 — underestimates, NOT inflated (correct behaviour)
#     rboost: rec=0.8177 — underestimates, NOT inflated (correct behaviour)
#     Neither learner falls into the confounding trap.
#     sg3_inflated = FALSE for both — R-learner's propensity residualisation successfully separates confounding from causal effect (could be a fluke at this sample size).

# --- SUBGROUP ORDERING (SG2 > SG3 > SG1) ---
#   rlasso: FALSE | rboost: FALSE
#   Neither recovers the correct severity ordering. With only ~23 treated training observations, learners cannot reliably rank patients by heterogeneous harm magnitude.

# --- OVERALL INTERPRETATION ---
#   At n=500 with ~5.8% treatment rate, both R-learner variants operate below the finite-sample threshold needed for reliable CATE estimation.
#   Key pattern: rlasso degrades to a weak-but-nonzero rank signal (0.325), while rboost collapses to a constant — rlasso is more informative even under extreme imbalance.
#   Both avoid the confounding trap (SG3 not inflated), confirming the R-loss structure provides robustness to confounding even when heterogeneity recovery fails.
#   These results motivate scaling to n=500 (see file 2) where more treated units in training might allow meaningful signal recovery.
# ============================================================