# ============================================================
# SMOKING AND LONGEVITY — FULL SINGLE ITERATION
# Outcome: 10-year lung function decline (higher = worse)
# Treatment: Heavy long-term smoking (W=1) vs non-heavy smoker (W=0)

# Core challenge: severe treatment imbalance (~7.5% treated)
# Confounding: low income and high stress drive both smoking and worse baseline outcomes independently

# To run for different n: change the single line below marked
# "CHANGE THIS" — everything else (split, subgroups, output) adapts.

# Output saved as CSV to:
#   Rlearner/Latest Phase/Smoking and Longevity/
# ============================================================

library(MASS)
library(rlearner)
library(KRLS2)

# ============================================================
# CONFIGURATION — change n here to run for 300, 500, or 1000
# ============================================================
set.seed(42)
n          = 500        # <<< CHANGE THIS: 300 / 500 / 1000
train_frac = 0.80       # 80/20 train-test split — do not change

## Imp: keep working directory as setwd("~/DSE4231/Rlearner")
output_dir = "Latest Phase/Smoking and Longevity"

n_train = floor(train_frac * n)
n_test  = n - n_train
cat(sprintf("Config: n=%d | n_train=%d | n_test=%d\n", n, n_train, n_test))

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: age (years)                       — range 40 to 85
# X2: income / SES                      — range 20 to 150 (thousands)
# X3: baseline lung function score      — range 40 to 110
# X4: health literacy / awareness       — range 0 to 20
# X5: stress index                      — range 0 to 10
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

cat("\n=== COVARIATE SUMMARY ===\n")
print(summary(x))
cat("\n=== COVARIATE CORRELATION ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
#
# Propensity driven by socioeconomic and behavioural factors only.
# Baseline lung function deliberately excluded — it determines how
# much smoking damages you (tau), not whether you start smoking.
# ============================================================

x1_s        = as.numeric(scale(x1_age))
x2_lowinc_s = as.numeric(scale(-x2_income))
x3_poorlung_s = as.numeric(scale(-x3_lung))  # used in tau and b_x, not propensity
x4_lowlit_s = as.numeric(scale(-x4_literacy))
x5_s        = as.numeric(scale(x5_stress))

linpred_no_intercept =
  0.7 * x1_s        +
  1.5 * x2_lowinc_s +
  0.9 * x4_lowlit_s +
  1.3 * x5_s

target_prev  = 0.075
intercept_fn = function(a) mean(plogis(a + linpred_no_intercept)) - target_prev
alpha        = uniroot(intercept_fn, interval = c(-10, 0))$root

propensity = pmax(0.01, pmin(plogis(alpha + linpred_no_intercept), 0.20))

repeat {
  w = rbinom(n, 1, propensity)
  if (mean(w) >= 0.05 && mean(w) <= 0.10) break
}

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion treated (target 5-10%):", round(mean(w), 3), "\n")
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
cat("  Lung     — treated:", round(mean(x3_lung[w==1]),   1),
    "| control:", round(mean(x3_lung[w==0]),   1),
    "(not a propensity driver — any difference is incidental)\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)
#
# Biological drivers of smoking harm:
#   Poor lung function (1.4): less reserve, faster decline
#   Older age (1.1):          less plastic, slower recovery
#   High stress (0.8):        amplifies inflammatory damage
#   Low literacy (0.5):       less able to mitigate harm
#
# Income excluded: affects baseline trajectory (b_x) via healthcare
# access and nutrition, but not the direct causal mechanism of
# smoking damage conditional on lung function and age.
# This creates the confounding trap for SG3.
#
# Constant = 3.0 ensures tau > 0 for all patients so pmax floor
# rarely binds and SG1 has genuine variation (not all = 0.5).
# ============================================================

tau_x = 3.0 +
  1.1 * x1_s          +
  1.4 * x3_poorlung_s +
  0.8 * x5_s          +
  0.5 * x4_lowlit_s

tau_x = pmax(0.5, tau_x)

cat("\n=== TRUE TREATMENT EFFECT ===\n")
cat("Mean:", round(mean(tau_x), 3),
    "| SD:", round(sd(tau_x), 3),
    "| Range:", round(min(tau_x), 3), "to", round(max(tau_x), 3), "\n")
# Mean: 3.211 | SD: 2.269 | Range: 0.5 to 8.851 
cat("% hitting floor:", round(mean(tau_x == 0.5)*100, 1), "%\n")
# % hitting floor: 17.7 %

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
# ============================================================

b_x = 18 +
  3.5 * x1_s              +
  3.0 * x2_lowinc_s       +
  5.0 * x3_poorlung_s     +
  2.0 * x4_lowlit_s       +
  4.0 * x5_s              +
  1.5 * x1_s * x3_poorlung_s

cat("\n=== BASELINE b(X) ===\n")
cat("Mean:", round(mean(b_x), 2), "| SD:", round(sd(b_x), 2), "\n")
# Mean: 18.46 | SD: 11.25 
cat("Signal/noise ratio SD(tau)/SD(b_x):", round(sd(tau_x)/sd(b_x), 3), "\n")
# Signal/noise ratio SD(tau)/SD(b_x): 0.202 

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
# ============================================================

sigma   = 3
y       = b_x + w * tau_x + rnorm(n, 0, sigma)

cat("\n=== OUTCOME Y ===\n")
cat("Naive ATE:", round(mean(y[w==1]) - mean(y[w==0]), 3), "(confounded) | True ATE:", round(mean(tau_x), 3), "\n")
# Naive ATE: 16.097 (confounded) | True ATE: 3.211 

# ============================================================
# SECTION 6: TRAIN-TEST SPLIT (80/20, no hardcoding)
# ============================================================

set.seed(42)
train_idx = sample(1:n, size = n_train, replace = FALSE)
test_idx  = setdiff(1:n, train_idx)

x_train = x[train_idx, ];  x_test = x[test_idx, ]
w_train = w[train_idx];    w_test = w[test_idx]
y_train = y[train_idx];    y_test = y[test_idx]
tau_test = tau_x[test_idx]

cat("\n=== TRAIN-TEST SPLIT ===\n")
cat("Train:", n_train, "| Treated in train:", sum(w_train), "(", round(mean(w_train)*100, 1), "%)\n")
cat("Test: ", n_test,  "| Treated in test: ", sum(w_test), "(", round(mean(w_test)*100,  1), "%)\n")
# 15 out of 240 (n=300) treated in train and 8 out of 60 treated in test set

# ============================================================
# SECTION 7: PRE-SPECIFIED CLINICAL SUBGROUPS ON TEST SET

# SG1 — Biologically resilient (expect LOWEST tau)
#   age < 55:   working-age, plastic lungs
#   lung > 75:  good baseline function (upper third of 40-110)
#   stress < 4: low chronic stress

# SG2 — Biologically vulnerable (expect HIGHEST tau)
#   age > 68:   elderly
#   lung < 58:  clinically impaired baseline
#   stress > 6: chronically stressed

# SG3 — Socially disadvantaged / confounding trap (expect MODERATE tau)
#   income < 45: low income (bottom ~15%)
#   literacy < 6: low health literacy
#   Income and literacy NOT in tau(X) — any inflation of estimated tau here indicates the learner is confounding b(X) with tau(X).
# ============================================================

x1_t = x_test[, "age"]
x2_t = x_test[, "income"]
x3_t = x_test[, "lung_baseline"]
x4_t = x_test[, "health_literacy"]
x5_t = x_test[, "stress"]

sg1 = x1_t < 55 & x3_t > 75 & x5_t < 4
sg2 = x1_t > 68 & x3_t < 58 & x5_t > 6
sg3 = x2_t < 45 & x4_t < 6

min_sg = 5
cat("\n=== SUBGROUP SIZES (test set) ===\n")
cat("SG1 (resilient — low harm):       ", sum(sg1), "\n") # 8 people
cat("SG2 (vulnerable — high harm):     ", sum(sg2), "\n") # 5 people
cat("SG3 (disadvantaged — confounding):", sum(sg3), "\n") # 9 people
if (sum(sg1) < min_sg) warning("SG1 < 5 patients — consider relaxing: age<60, lung>70, stress<5")
if (sum(sg2) < min_sg) warning("SG2 < 5 patients — consider relaxing: age>65, lung<62, stress>5")
if (sum(sg3) < min_sg) warning("SG3 < 5 patients — consider relaxing: income<55, literacy<8")

true_sg1 = mean(tau_test[sg1])
true_sg2 = mean(tau_test[sg2])
true_sg3 = mean(tau_test[sg3])

cat("True tau:  SG1=", round(true_sg1, 3), " SG2=", round(true_sg2, 3), " SG3=", round(true_sg3, 3), "\n")
# True tau:  SG1= 0.5  SG2= 6.504  SG3= 4.332 
cat("Ordering SG2 > SG3 > SG1:", (true_sg2 > true_sg3) & (true_sg3 > true_sg1), "\n")
# TRUE for ordering

# ============================================================
# SECTION 8: FIT ALL LEARNERS
# Boosting: ntrees_max = 100 — reduced because T/X learners train on arm subsets; with ~22 treated in training, 300 trees overfits.
# All fits wrapped in tryCatch — returns NA on failure.
# ============================================================

boost_args = list(
  num_search_rounds = 5, k_folds = 5,
  ntrees_max = 100, early_stopping_rounds = 5, verbose = FALSE
)
boost_args_tx = list(
  num_search_rounds = 5, k_folds_mu1 = 5, k_folds_mu0 = 5,
  ntrees_max = 100, early_stopping_rounds = 5, verbose = FALSE
)

safe_fit = function(fit_fn, pred_fn, ...) {
  tryCatch({
    fit = fit_fn(...)
    pred_fn(fit, x_test)
  }, error = function(e) {
    cat("  [FAILED:", conditionMessage(e), "]\n")
    rep(NA_real_, n_test)
  })
}

fit_start = Sys.time()
cat("\n=== FITTING LEARNERS ===\n")
cat("n_train =", n_train, "| n_treated_train =", sum(w_train), "\n\n")

cat("R-learner: rlasso...\n")
rlasso_est = safe_fit(rlasso, predict, x_train, w_train, y_train)
cat("R-learner: rboost...\n")
rboost_est = safe_fit(function(...) do.call(rboost, c(list(...), boost_args)),
                      predict, x_train, w_train, y_train)
cat("R-learner: rkern...\n")
rkern_est  = safe_fit(rkern,  predict, x_train, w_train, y_train)

cat("S-learner: slasso...\n")
slasso_est = safe_fit(slasso, predict, x_train, w_train, y_train)
cat("S-learner: sboost...\n")
sboost_est = safe_fit(function(...) do.call(sboost, c(list(...), boost_args)),
                      predict, x_train, w_train, y_train)
cat("S-learner: skern...\n")
skern_est  = safe_fit(skern,  predict, x_train, w_train, y_train)

cat("T-learner: tlasso...\n")
tlasso_est = safe_fit(tlasso, predict, x_train, w_train, y_train)
cat("T-learner: tboost (may be unstable with few treated)...\n")
tboost_est = safe_fit(function(...) do.call(tboost, c(list(...), boost_args_tx)),
                      predict, x_train, w_train, y_train)
cat("T-learner: tkern...\n")
tkern_est  = safe_fit(tkern,  predict, x_train, w_train, y_train)

cat("X-learner: xlasso...\n")
xlasso_est = safe_fit(xlasso, predict, x_train, w_train, y_train)
cat("X-learner: xboost (may be unstable with few treated)...\n")
xboost_est = safe_fit(function(...) do.call(xboost, c(list(...), boost_args_tx)),
                      predict, x_train, w_train, y_train)
cat("X-learner: xkern...\n")
xkern_est  = safe_fit(xkern,  predict, x_train, w_train, y_train)

# OLS with W x covariate interactions as linear baseline
x_tr_df = as.data.frame(x_train)
x_te_df = as.data.frame(x_test)
colnames(x_tr_df) = colnames(x_te_df) = c("age","income","lung","literacy","stress")

ols_fit = tryCatch({
  lm(y_train ~ age + income + lung + literacy + stress + w_train +
       I(w_train*age) + I(w_train*income) + I(w_train*lung) +
       I(w_train*literacy) + I(w_train*stress),
     data = cbind(x_tr_df, y_train, w_train))
}, error = function(e) NULL)

if (!is.null(ols_fit)) {
  cf = coef(ols_fit)
  ols_tau_est = cf["w_train"] +
    cf["I(w_train * age)"]      * x_te_df$age +
    cf["I(w_train * income)"]   * x_te_df$income +
    cf["I(w_train * lung)"]     * x_te_df$lung +
    cf["I(w_train * literacy)"] * x_te_df$literacy +
    cf["I(w_train * stress)"]   * x_te_df$stress
} else {
  ols_tau_est = rep(NA_real_, n_test)
}

cat("\nFitting time:", round(as.numeric(Sys.time()-fit_start, units="mins"), 2), "mins\n")

learners_all = list(
  rlasso = rlasso_est, rboost = rboost_est, rkern  = rkern_est,
  slasso = slasso_est, sboost = sboost_est, skern  = skern_est,
  tlasso = tlasso_est, tboost = tboost_est, tkern  = tkern_est,
  xlasso = xlasso_est, xboost = xboost_est, xkern  = xkern_est,
  zero_pred = rep(mean(tau_test), n_test),   # baseline: predict mean tau
  ols_inter = as.numeric(ols_tau_est)
)

# ============================================================
# SECTION 9: EVALUATION
# ============================================================

tau_variance = var(tau_test)

compute_metrics = function(est, name) {
  if (all(is.na(est))) {
    return(list(learner=name, failed=TRUE,
                norm_mse=NA, rank_corr=NA,
                est_sg1=NA, rec_sg1=NA,
                est_sg2=NA, rec_sg2=NA,
                est_sg3=NA, rec_sg3=NA, dev_sg3=NA,
                correct_order=NA, sg3_inflated=NA))
  }
  raw_mse  = mean((est - tau_test)^2, na.rm=TRUE)
  norm_mse = raw_mse / tau_variance
  rank_corr = ifelse(sd(est, na.rm=TRUE) < 1e-10, NA,
                     cor(est, tau_test, method="spearman", use="complete.obs"))
  e1 = mean(est[sg1], na.rm=TRUE)
  e2 = mean(est[sg2], na.rm=TRUE)
  e3 = mean(est[sg3], na.rm=TRUE)
  list(
    learner       = name,
    failed        = FALSE,
    norm_mse      = round(norm_mse, 4),
    rank_corr     = ifelse(is.na(rank_corr), NA, round(rank_corr, 4)),
    est_sg1       = round(e1, 3), rec_sg1 = round(e1/true_sg1, 3),
    est_sg2       = round(e2, 3), rec_sg2 = round(e2/true_sg2, 3),
    est_sg3       = round(e3, 3), rec_sg3 = round(e3/true_sg3, 3),
    dev_sg3       = round(abs(e3-true_sg3), 3),
    correct_order = (e2>e3)&(e3>e1),
    sg3_inflated  = e3/true_sg3 > 1.1
  )
}

metrics_list = lapply(names(learners_all),
                      function(nm) compute_metrics(learners_all[[nm]], nm))
names(metrics_list) = names(learners_all)

# ============================================================
# SECTION 10: PRINT SUMMARY TABLE
# ============================================================

cat("\n=== FULL RESULTS SUMMARY ===\n")
cat(sprintf("%-12s | %-8s | %-8s | %-7s | %-7s | %-7s | %-5s | %-5s\n",
            "Learner","NormMSE","RankCorr","RecSG1","RecSG2","RecSG3","Order","SG3Inf"))
cat(strrep("-", 80), "\n")

summary_df = do.call(rbind, lapply(metrics_list, function(m) {
  data.frame(
    learner       = m$learner,
    failed        = m$failed,
    norm_mse      = ifelse(m$failed, NA, m$norm_mse),
    rank_corr     = ifelse(m$failed, NA, m$rank_corr),
    est_sg1 = ifelse(m$failed, NA, m$est_sg1),
    true_sg1 = round(true_sg1, 3),
    rec_sg1  = ifelse(m$failed, NA, m$rec_sg1),
    est_sg2 = ifelse(m$failed, NA, m$est_sg2),
    true_sg2 = round(true_sg2, 3),
    rec_sg2  = ifelse(m$failed, NA, m$rec_sg2),
    est_sg3 = ifelse(m$failed, NA, m$est_sg3),
    true_sg3 = round(true_sg3, 3),
    rec_sg3  = ifelse(m$failed, NA, m$rec_sg3),
    dev_sg3  = ifelse(m$failed, NA, m$dev_sg3),
    correct_order = ifelse(m$failed, NA, m$correct_order),
    sg3_inflated  = ifelse(m$failed, NA, m$sg3_inflated),
    stringsAsFactors = FALSE
  )
}))

summary_df = summary_df[order(summary_df$norm_mse, na.last=TRUE), ]

for (i in seq_len(nrow(summary_df))) {
  r = summary_df[i, ]
  if (r$failed) {
    cat(sprintf("%-12s | FAILED\n", r$learner))
  } else {
    cat(sprintf("%-12s | %-8.4f | %-8s | %-7.3f | %-7.3f | %-7.3f | %-5s | %-5s\n",
                r$learner,
                r$norm_mse,
                ifelse(is.na(r$rank_corr), "   NA  ", sprintf("%.4f", r$rank_corr)),
                r$rec_sg1, r$rec_sg2, r$rec_sg3,
                ifelse(r$correct_order, "YES", "NO"),
                ifelse(r$sg3_inflated, "YES", "no")))
  }
}

cat("\nNote: NormMSE < 1.0 = better than predicting mean tau for everyone\n")
cat("      zero_pred (NormMSE = 1.000) is the baseline reference\n")
cat("      SG3 inflated (rec > 1.1) = learner confusing confounding with causal effect\n")

# ============================================================
# SECTION 11: SAVE RESULTS TO CSV
# ============================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Add metadata columns
summary_df$n             = n
summary_df$n_train       = n_train
summary_df$n_test        = n_test
summary_df$n_treated_train = sum(w_train)
summary_df$n_treated_test  = sum(w_test)
summary_df$true_ate      = round(mean(tau_x), 4)
summary_df$tau_sd        = round(sd(tau_x), 4)
summary_df$sg1_n         = sum(sg1)
summary_df$sg2_n         = sum(sg2)
summary_df$sg3_n         = sum(sg3)

csv_path = file.path(output_dir, paste0("smoking_all_", n, ".csv"))
write.csv(summary_df, csv_path, row.names = FALSE)
cat("\nResults saved to:", csv_path, "\n")

# ============================================================
# SECTION 12: ONE PLOT — Propensity histogram
# ============================================================

png(file.path(output_dir, paste0("propensity_hist_all_", n, ".png")), 
    width = 800, height = 500, res = 120)
hist(propensity[w == 0], breaks = 20,
     col  = rgb(0.2, 0.4, 0.8, 0.5),
     xlim = c(0, 0.22),
     xlab = "Propensity score",
     ylab = "Count",
     main = paste0("Propensity by arm — n=", n,
                   " (", sum(w_train), " treated in training)"))
hist(propensity[w == 1], breaks = 10,
     col  = rgb(0.9, 0.2, 0.2, 0.5), add = TRUE)
legend("topright",
       legend = c(paste0("Control (n=", sum(w==0), ")"),
                  paste0("Treated (n=", sum(w==1), ")")),
       fill = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.9, 0.2, 0.2, 0.5)),
       cex = 0.8)
dev.off()
cat("Plot saved to:", file.path(output_dir, paste0("propensity_hist_n", n, ".png")), "\n")

# ============================================================
# RESULTS ANALYSIS n=300, seed=42, all
# ============================================================
# Setup: 15 treated in training (5.8%) — extreme imbalance
# True ATE=3.21 | tau SD=2.27
# True tau: SG1=0.50 (resilient), SG2=6.50 (vulnerable), SG3=4.33 (confounding)
# tkern and xkern FAILED (crashed — insufficient treated observations)

# KEY FINDINGS:
# Best learners: rkern (0.423) and xlasso (0.473) — both below 1.0, meaning they beat mean prediction despite only 15 treated in training. rkern achieves rank_corr=0.809 (reliable), xlasso=0.827 (reliable). Surprising that kern and lasso variants work at this n.
# rlasso (0.732, rank=0.674): functional but weaker than rkern/xlasso. Correct ordering (SG2>SG3>SG1). No SG3 inflation — R-learner's propensity residualisation successfully avoids the confounding trap.
# rboost (1.015): just above baseline, rank=-0.399 (anti-correlated). Collapsed near-constant — no heterogeneity recovered.
# tboost (52.3): catastrophic failure. norm_mse=52, rank=-0.888, SG2 estimated at -14 (sign wrong). T-learner's treated arm model has 15 observations — boosting overfits completely.
# sboost (9.49) and xboost (4.15): also badly degraded. Boosting-based methods universally fail at this imbalance level.
# SG1 recovery: unreliable across all learners (true_sg1=0.50, near the pmax floor — small estimation errors inflate ratios).
# SG3: no learner inflates (sg3_inflated=FALSE for all non-failed). Confounding trap not triggered at n=300 — learners attenuate rather than inflate under extreme imbalance.
# Propensity plot: bimodal distribution — most patients cluster near 0.01-0.03 (very unlikely to smoke), with a second cluster at the 0.20 cap for highest-risk patients. Treated patients (red) concentrated at the cap. Visually confirms extreme imbalance.

# CONCLUSION: At n=300 (15 treated), kern and lasso variants show surprising resilience. Boosting fails consistently. Results motivate n=500 and n=1000 to test whether this pattern holds.
# ============================================================
