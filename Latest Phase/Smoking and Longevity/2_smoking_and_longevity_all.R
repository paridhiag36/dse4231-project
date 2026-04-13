# ============================================================
# SMOKING AND LONGEVITY — FULL SINGLE ITERATION
# Outcome: 10-year lung function decline (higher = worse)
# Treatment: Heavy long-term smoking (W=1) vs non-heavy smoker (W=0)

# Core challenge: severe treatment imbalance (~7.5% treated)
#   With only ~18-22 treated patients in a 240-patient training set, learners that rely on direct treated/control comparison struggle.
#   The R-learner's propensity residualisation should give it a structural advantage — it explicitly de-weights observations where treatment probability is very low.

# Confounding: low income and high stress increase both smoking probability AND worsen baseline health outcomes independently.
#   A naive learner conflates these channels and overestimates tau.

# Chnages from 1_smoking_and_longevity.R (single-iteration exploratory version):
#   - n increased to 300 (240 train / 60 test) — see note below
#   - Learners fitted on training set, evaluated on held-out test set
#   - Full 12-learner comparison (R/S/T/X x lasso/boost/kern)
#   - Boosting-based T and X learners wrapped in tryCatch because they are expected to be unstable with ~18 treated training observations
#   - ntrees_max reduced to 100 for all boosting fits specifically because of the tiny treated arm
#   - Pre-specified clinical subgroup thresholds on test set

# NOTE ON SAMPLE SIZE:
#   n=300 chosen over n=200 because at 7.5% treatment rate, n=200 gives ~15 treated units total — too few for any learner to work with meaningfully. n=300 gives ~22 treated in training, which is still very rare but at least interpretable.
#   This is still far fewer than balanced designs (e.g., telemed had ~120 treated in training) — the imbalance challenge is fully preserved.
# ============================================================

library(MASS)       # mvrnorm — correlated covariate generation
library(rlearner)   # all meta-learners
library(KRLS2)      # kernel ridge regression
library(jsonlite)   # JSON export

set.seed(42)
n = 1000   # increased from 200 to ensure interpretable results

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: age (years)                       — range 40 to 85
# X2: income / SES                      — range 20 to 150 (thousands)
# X3: baseline lung function score      — range 40 to 110
# X4: health literacy                   — range 0 to 20
# X5: stress index                      — range 0 to 10

# Correlation structure:
#   age    <-> lung:     -0.35  older patients have lower baseline function
#   income <-> literacy: +0.50  higher SES strongly linked to health literacy
#   income <-> stress:   -0.30  lower SES linked to higher chronic stress
#   literacy <-> stress: -0.35  low literacy coexists with high stress
#   age    <-> literacy: -0.20  older cohorts slightly less health-literate
#   age    <-> stress:   +0.15  modest positive — older adults carry more stress
# ============================================================

cor_matrix = matrix(c(
  # X1_age  X2_income  X3_lung  X4_literacy  X5_stress
  1.00,    -0.10,     -0.35,    -0.20,       0.15,
  -0.10,     1.00,      0.15,     0.50,      -0.30,
  -0.35,     0.15,      1.00,     0.20,      -0.25,
  -0.20,     0.50,      0.20,     1.00,      -0.35,
  0.15,    -0.30,     -0.25,    -0.35,       1.00
), nrow = 5, byrow = TRUE)

z = MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

x1_age      = round(40 + 45 * pnorm(z[,1]))   # 40-85 years, integer
x2_income   = 20  + 130 * pnorm(z[,2])        # 20-150 thousands, continuous
x3_lung     = 40  +  70 * pnorm(z[,3])        # 40-110 score, continuous
x4_literacy = round(20  * pnorm(z[,4]))        # 0-20 score, integer
x5_stress   =  10 * pnorm(z[,5])              # 0-10 index, continuous

x = cbind(x1_age, x2_income, x3_lung, x4_literacy, x5_stress)
colnames(x) = c("age", "income", "lung_baseline", "health_literacy", "stress")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))
cat("\n=== COVARIATE CORRELATION (approximately reflects design) ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT

# Risk-oriented standardised covariates (positive = more smoking risk): Flipping income, lung, and literacy so higher = riskier for the propensity model — makes coefficients easier to read

# Calibration: uniroot finds the intercept that achieves exactly 7.5% average propensity — more reliable than guessing manually

# Clipping at 0.20: even the highest-risk patients are capped at 20% probability of heavy smoking. Clinically realistic — heavy smoking is a minority behaviour even in the most deprived groups.
#   Learners do not know about this cap, which introduces mild propensity misspecification inherent to this setup.
# ============================================================

x1_s          = as.numeric(scale(x1_age))
x2_lowinc_s   = as.numeric(scale(-x2_income))    # flipped: lower income = higher risk
x3_poorlung_s = as.numeric(scale(-x3_lung))       # flipped: poorer lungs = higher risk
x4_lowlit_s   = as.numeric(scale(-x4_literacy))   # flipped: lower literacy = higher risk
x5_s          = as.numeric(scale(x5_stress))

linpred_no_intercept =
  0.7 * x1_s          +   # age: older = marginally more likely to be heavy smoker
  1.5 * x2_lowinc_s   +   # income: strongest driver — low SES predicts smoking
  0.4 * x3_poorlung_s +   # poor lungs: selection effect (sick people smoke more)
  0.9 * x4_lowlit_s   +   # literacy: low awareness raises smoking uptake
  1.3 * x5_s              # stress: strong predictor of heavy smoking

# Find intercept so mean propensity = 7.5%
target_prev  = 0.075
intercept_fn = function(a) mean(plogis(a + linpred_no_intercept)) - target_prev
alpha        = uniroot(intercept_fn, interval = c(-10, 0))$root

propensity = plogis(alpha + linpred_no_intercept)
propensity = pmax(0.01, pmin(propensity, 0.20))   # cap at 0.20 — see header

# Resample until realised treatment rate is within 5-10%
# Rarely needs more than one draw; prevents unlucky extreme samples
repeat {
  w = rbinom(n, 1, propensity)
  if (mean(w) >= 0.05 && mean(w) <= 0.10) break
}

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion heavy smokers (target 5-10%):", round(mean(w), 3), "\n")
cat("Number treated:", sum(w), "| Number control:", sum(1-w), "\n")
cat("Propensity range:", round(min(propensity), 3),
    "to", round(max(propensity), 3), "(capped at 0.20)\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")

cat("\nConfounding check (treated vs control means):\n")
cat("  Age        — treated:", round(mean(x1_age[w==1]),     1),
    "| control:", round(mean(x1_age[w==0]),     1), "\n")
cat("  Income     — treated:", round(mean(x2_income[w==1]),  1),
    "| control:", round(mean(x2_income[w==0]),  1), "\n")
cat("  Lung score — treated:", round(mean(x3_lung[w==1]),    1),
    "| control:", round(mean(x3_lung[w==0]),    1), "\n")
cat("  Literacy   — treated:", round(mean(x4_literacy[w==1]),1),
    "| control:", round(mean(x4_literacy[w==0]),1), "\n")
cat("  Stress     — treated:", round(mean(x5_stress[w==1]),  2),
    "| control:", round(mean(x5_stress[w==0]),  2), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)

# tau(X) = extra lung function decline CAUSED BY heavy smoking, above and beyond natural decline captured by b(X).
# Higher tau = more harmful. Always positive (floor enforced below).

# Drivers of harm magnitude:
#   Poor baseline lung function (coef 1.4) — largest effect. People with already-impaired lungs have less reserve; smoking accelerates their decline fastest.
#   Older age (coef 1.1) — second largest. Older lungs are less plastic and recover more slowly from damage.
#   High stress (coef 0.8): Stress amplifies inflammatory damage caused by smoking.
#   Low literacy (coef 0.5): Less able to mitigate harm through behaviour change or care-seeking.

# Note: income does NOT appear in tau(X). Income shapes who smokes (propensity) and baseline health (b_x) but NOT the causal mechanism of how smoking damages lung tissue. 
#   This creates the confounding trap for SG3 — naive learners see low-income patients have worse outcomes and may inflate their estimated tau, when in truth tau for low-income patients is close to the population average.
# ============================================================

tau_x = 2.0 +
  1.1 * x1_s          +   # older = harmed more
  1.4 * x3_poorlung_s +   # poorer lungs = harmed most (largest coefficient)
  0.8 * x5_s          +   # higher stress = amplifies damage
  0.5 * x4_lowlit_s       # lower literacy = less able to mitigate

tau_x = pmax(0.5, tau_x)  # enforce always-harmful floor

cat("\n=== TRUE TREATMENT EFFECT SUMMARY ===\n")
cat("Mean tau(X):", round(mean(tau_x), 3), "\n")
cat("SD tau(X):  ", round(sd(tau_x),   3), "\n")
cat("Range:      ", round(min(tau_x),  3), "to", round(max(tau_x), 3), "\n")
cat("All tau > 0:", all(tau_x > 0), "(should be TRUE)\n")

if (abs(mean(tau_x)) > 5)
  warning("Mean tau = ", round(mean(tau_x), 2),
          " — check calibration constant in tau_x formula.")

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)

# b(X) = 10-year lung function decline WITHOUT heavy smoking.
# Deliberately complex (includes nonlinear age x lung interaction) to make nuisance estimation hard — this tests whether learners can separate b(X) from tau(X) under imbalance.

# The confounding lives here: low income and high stress raise b(X) AND raise smoking probability. Smokers therefore have higher b(X) on average. 
# A naive learner sees smokers are worse off overall and over-attributes this to smoking rather than to their worse b(X).
# ============================================================

b_x = 18 +
  3.5 * x1_s          +       # age worsens natural decline
  3.0 * x2_lowinc_s   +       # lower income = worse long-run trajectory (confounding)
  5.0 * x3_poorlung_s +       # poor baseline = faster natural decline
  2.0 * x4_lowlit_s   +       # low literacy = worse disease management
  4.0 * x5_s          +       # high stress = independently worsens outcomes
  1.5 * x1_s * x3_poorlung_s  # nonlinear: old + bad lungs = steeper decline

cat("\n=== BASELINE OUTCOME b(X) ===\n")
cat("Mean b(X):", round(mean(b_x), 2), "\n")
cat("Range:    ", round(min(b_x),  2), "to", round(max(b_x), 2), "\n")
cat("SD b(X):  ", round(sd(b_x),   2),
    "(should exceed SD of tau to reflect dominant nuisance)\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y

# Y = b(X) + W * tau(X) + epsilon

# sigma = 3: measurement noise + unmeasured factors (diet, genetics, air quality etc.).
# ============================================================

sigma   = 3
epsilon = rnorm(n, mean = 0, sd = sigma)
y       = b_x + w * tau_x + epsilon

cat("\n=== OBSERVED OUTCOME Y ===\n")
cat("Mean Y (all):    ", round(mean(y),        2), "\n")
cat("Mean Y treated:  ", round(mean(y[w==1]),  2), "\n")
cat("Mean Y control:  ", round(mean(y[w==0]),  2), "\n")
cat("Naive ATE:       ", round(mean(y[w==1]) - mean(y[w==0]), 3),
    "(confounded — overstates true ATE due to worse b_x in smokers)\n")
cat("True ATE:        ", round(mean(tau_x), 3), "\n")

# ============================================================
# SECTION 6: TRAIN-TEST SPLIT
# 240 train / 60 test

# With 7.5% treatment rate we expect:
#   ~18-22 treated patients in training — very few but workable
#   ~4-6 treated patients in test set — enough for subgroup evaluation (subgroup sizes checked below with warnings)
# ============================================================

set.seed(42)
train_idx = sample(1:n, size = 800, replace = FALSE)
test_idx  = setdiff(1:n, train_idx)

x_train  = x[train_idx, ];  x_test  = x[test_idx, ]
w_train  = w[train_idx];    w_test  = w[test_idx]
y_train  = y[train_idx];    y_test  = y[test_idx]
tau_test = tau_x[test_idx]

cat("\n=== TRAIN-TEST SPLIT (80% / 20%) ===\n")
cat("Train — n:", length(train_idx),
    "| treated:", sum(w_train),
    "| rate:", round(mean(w_train), 3), "\n")
cat("Test  — n:", length(test_idx),
    "| treated:", sum(w_test),
    "| rate:", round(mean(w_test), 3), "\n")

# ============================================================
# SECTION 7: PRE-SPECIFIED CLINICAL SUBGROUPS ON TEST SET

# Fixed thresholds — identical across all seeds and iterations.
# Defined on test set only to avoid any circularity with training.

# SG1 — Lowest harm: young, good lungs, low stress, health-literate
# SG2 — Highest harm: old, poor lung function, high stress
# SG3 — Confounding trap: low income, low literacy
#   Naive learners conflate bad b(X) with large treatment effect and inflate their tau estimate.

# Target severity ordering: SG2 > SG3 > SG1
# ============================================================

x1_test = x_test[, "age"]
x2_test = x_test[, "income"]
x3_test = x_test[, "lung_baseline"]
x4_test = x_test[, "health_literacy"]
x5_test = x_test[, "stress"]

sg1 = x1_test <  60 &   # younger working-age adult
  x3_test >  70 &   # good baseline lung function
  x5_test <   5 &   # low stress
  x4_test >  12     # health-literate

sg2 = x1_test >  65 &   # elderly
  x3_test <  60 &   # clinically impaired lung function
  x5_test >   6     # chronically stressed

sg3 = x2_test <  50 &   # low income
  x4_test <   7     # low health literacy

min_sg_size = 5

cat("\n=== SUBGROUP SIZES (test set, pre-specified thresholds) ===\n")
cat("SG1 (young/good-lung/low-stress/literate):", sum(sg1),
    "— expect lowest tau\n")
cat("SG2 (old/poor-lung/high-stress):          ", sum(sg2),
    "— expect highest tau\n")
cat("SG3 (low-income/low-literacy):            ", sum(sg3),
    "— confounding trap, expect moderate tau\n\n")

if (sum(sg1) < min_sg_size)
  warning("SG1 has only ", sum(sg1), " patients — results unreliable.",
          " Consider relaxing: age<60, lung>70, stress<5, literacy>12")
if (sum(sg2) < min_sg_size)
  warning("SG2 has only ", sum(sg2), " patients — results unreliable.",
          " Consider relaxing: age>65, lung<60, stress>6")
if (sum(sg3) < min_sg_size)
  warning("SG3 has only ", sum(sg3), " patients — results unreliable.",
          " Consider relaxing: income<50, literacy<7")

true_sg1 = mean(tau_test[sg1])
true_sg2 = mean(tau_test[sg2])
true_sg3 = mean(tau_test[sg3])

cat("True tau by subgroup (test set):\n")
cat("  SG1:", round(true_sg1, 3), "(expect lowest)\n")
cat("  SG2:", round(true_sg2, 3), "(expect highest)\n")
cat("  SG3:", round(true_sg3, 3),
    "(expect moderate — confounding trap)\n")
cat("  Target ordering: SG2 > SG3 > SG1 =",
    (true_sg2 > true_sg3) & (true_sg3 > true_sg1), "\n")

# ============================================================
# SECTION 8: FIT ALL LEARNERS ON TRAINING DATA

# Boosting note:
#   ntrees_max = 100 (reduced from 300 in telemed) because with only ~18-22 treated training observations, the T-learner and X-learner fit separate models on each arm. The treated-arm model has ~18 observations — 300 trees would massively overfit.
#   Even at 100 trees, tboost and xboost are expected to be unstable.
#   All boosting fits are wrapped in tryCatch so failures return NA rather than crashing the script.
# ============================================================

boost_args = list(
  num_search_rounds     = 5,
  k_folds               = 5,
  ntrees_max            = 100,   # reduced from 300 — tiny treated arm
  early_stopping_rounds = 5,
  verbose               = FALSE
)

boost_args_tx = list(
  num_search_rounds     = 5,
  k_folds_mu1           = 5,
  k_folds_mu0           = 5,
  ntrees_max            = 100,   # reduced — tboost/xboost train on arm subsets
  early_stopping_rounds = 5,
  verbose               = FALSE
)

# Safe fitting wrapper — returns NA vector if learner crashes
# This is especially important for tboost/xboost in this setup
safe_fit = function(fit_fn, pred_fn, n_test, ...) {
  tryCatch({
    fit = fit_fn(...)
    pred_fn(fit, x_test)
  }, error = function(e) {
    cat("  [WARNING: learner failed —", conditionMessage(e), "]\n")
    rep(NA_real_, n_test)
  })
}

n_test = length(test_idx)

fit_time_start = Sys.time()
cat("\n=== FITTING LEARNERS ON TRAINING DATA (n_train=240, n_treated~18-22) ===\n")

cat("R-learner: rlasso...\n")
rlasso_est = safe_fit(rlasso, predict, n_test, x_train, w_train, y_train)

cat("R-learner: rboost...\n")
rboost_est = safe_fit(
  function(...) do.call(rboost, c(list(...), boost_args)),
  predict, n_test, x_train, w_train, y_train)

cat("R-learner: rkern...\n")
rkern_est  = safe_fit(rkern, predict, n_test, x_train, w_train, y_train)

cat("S-learner: slasso...\n")
slasso_est = safe_fit(slasso, predict, n_test, x_train, w_train, y_train)

cat("S-learner: sboost...\n")
sboost_est = safe_fit(
  function(...) do.call(sboost, c(list(...), boost_args)),
  predict, n_test, x_train, w_train, y_train)

cat("S-learner: skern...\n")
skern_est  = safe_fit(skern, predict, n_test, x_train, w_train, y_train)

cat("T-learner: tlasso...\n")
tlasso_est = safe_fit(tlasso, predict, n_test, x_train, w_train, y_train)

cat("T-learner: tboost (may be unstable — ~18 treated obs in training)...\n")
tboost_est = safe_fit(
  function(...) do.call(tboost, c(list(...), boost_args_tx)),
  predict, n_test, x_train, w_train, y_train)

cat("T-learner: tkern...\n")
tkern_est  = safe_fit(tkern, predict, n_test, x_train, w_train, y_train)

cat("X-learner: xlasso...\n")
xlasso_est = safe_fit(xlasso, predict, n_test, x_train, w_train, y_train)

cat("X-learner: xboost (may be unstable — ~18 treated obs in training)...\n")
xboost_est = safe_fit(
  function(...) do.call(xboost, c(list(...), boost_args_tx)),
  predict, n_test, x_train, w_train, y_train)

cat("X-learner: xkern...\n")
xkern_est  = safe_fit(xkern, predict, n_test, x_train, w_train, y_train)

# --- Baselines ---
# Zero predictor: predicts tau = 0 for everyone
# Normalised MSE of this = 1.0 by definition — our reference floor
zero_pred = rep(0, n_test)

# OLS with interactions: linear model for tau(X), fitted on training data
x_train_df          = as.data.frame(x_train)
colnames(x_train_df)= c("age", "income", "lung", "literacy", "stress")
x_test_df           = as.data.frame(x_test)
colnames(x_test_df) = c("age", "income", "lung", "literacy", "stress")

ols_fit = tryCatch({
  lm(y_train ~ age + income + lung + literacy + stress +
       w_train +
       I(w_train * age) + I(w_train * income) + I(w_train * lung) +
       I(w_train * literacy) + I(w_train * stress),
     data = cbind(x_train_df, y_train, w_train))
}, error = function(e) NULL)

if (!is.null(ols_fit)) {
  cf          = coef(ols_fit)
  ols_tau_est = cf["w_train"] +
    cf["I(w_train * age)"]      * x_test_df$age      +
    cf["I(w_train * income)"]   * x_test_df$income   +
    cf["I(w_train * lung)"]     * x_test_df$lung      +
    cf["I(w_train * literacy)"] * x_test_df$literacy +
    cf["I(w_train * stress)"]   * x_test_df$stress
} else {
  cat("[WARNING: OLS fit failed]\n")
  ols_tau_est = rep(NA_real_, n_test)
}

fit_time_end = Sys.time()
cat("\nTotal fitting time:",
    round(as.numeric(fit_time_end - fit_time_start, units = "mins"), 2),
    "minutes\n")

# Unified learner list — all predictions on test set
learners_all = list(
  rlasso    = rlasso_est,
  rboost    = rboost_est,
  rkern     = rkern_est,
  slasso    = slasso_est,
  sboost    = sboost_est,
  skern     = skern_est,
  tlasso    = tlasso_est,
  tboost    = tboost_est,
  tkern     = tkern_est,
  xlasso    = xlasso_est,
  xboost    = xboost_est,
  xkern     = xkern_est,
  zero_pred = zero_pred,
  ols_inter = as.numeric(ols_tau_est)
)

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

# ============================================================
# SECTION 9: EVALUATION ON TEST SET
# ============================================================

tau_variance = var(tau_test)

cat("\n=== EVALUATION ON TEST SET (n=60) ===\n")
cat("Variance of true tau on test set:", round(tau_variance, 4), "\n\n")

compute_metrics = function(est, name) {
  
  # Return NAs if learner failed
  if (all(is.na(est))) {
    return(list(
      learner       = name, failed = TRUE,
      raw_mse       = NA, norm_mse  = NA, quality_mse  = "FAILED",
      rank_corr     = NA, quality_rank = "FAILED",
      est_sg1       = NA, rec_sg1 = NA,
      est_sg2       = NA, rec_sg2 = NA,
      est_sg3       = NA, rec_sg3 = NA, dev_sg3 = NA,
      correct_order = NA, sg3_inflated = NA
    ))
  }
  
  raw_mse  = mean((est - tau_test)^2, na.rm = TRUE)
  norm_mse = raw_mse / tau_variance
  
  rank_corr = ifelse(sd(est, na.rm = TRUE) < 1e-10, NA,
                     cor(est, tau_test, method = "spearman",
                         use = "complete.obs"))
  
  quality_mse = ifelse(norm_mse < 0.25, "EXCELLENT",
                       ifelse(norm_mse < 0.75, "ACCEPTABLE",
                              ifelse(norm_mse < 1.00, "POOR",
                                     "WORSE THAN MEAN")))
  
  quality_rank = ifelse(is.na(rank_corr), "UNDEFINED",
                        ifelse(rank_corr > 0.80, "RELIABLE",
                               ifelse(rank_corr > 0.50, "MODERATE",
                                      ifelse(rank_corr > 0.30, "WEAK",
                                             "RANDOM"))))
  
  est_sg1 = mean(est[sg1], na.rm = TRUE)
  est_sg2 = mean(est[sg2], na.rm = TRUE)
  est_sg3 = mean(est[sg3], na.rm = TRUE)
  
  rec_sg1 = est_sg1 / true_sg1
  rec_sg2 = est_sg2 / true_sg2
  rec_sg3 = est_sg3 / true_sg3
  dev_sg3 = abs(est_sg3 - true_sg3)
  
  # Metric 4: correct ordering — does learner rank SG2 > SG3 > SG1?
  correct_order = (!is.na(est_sg2) & !is.na(est_sg3) & !is.na(est_sg1)) &&
    (est_sg2 > est_sg3) && (est_sg3 > est_sg1)
  
  # Metric 5: SG3 inflation — is learner overestimating tau for the
  # confounded subgroup (low-income/low-literacy) by more than 10%?
  sg3_inflated = (!is.na(rec_sg3)) && (rec_sg3 > 1.1)
  
  list(
    learner       = name,
    failed        = FALSE,
    raw_mse       = round(raw_mse,   4),
    norm_mse      = round(norm_mse,  4),
    quality_mse   = quality_mse,
    rank_corr     = ifelse(is.na(rank_corr), NA, round(rank_corr, 4)),
    quality_rank  = quality_rank,
    est_sg1       = round(est_sg1, 3),
    true_sg1      = round(true_sg1, 3),
    rec_sg1       = round(rec_sg1, 3),
    est_sg2       = round(est_sg2, 3),
    true_sg2      = round(true_sg2, 3),
    rec_sg2       = round(rec_sg2, 3),
    est_sg3       = round(est_sg3, 3),
    true_sg3      = round(true_sg3, 3),
    rec_sg3       = round(rec_sg3, 3),
    dev_sg3       = round(dev_sg3, 3),
    correct_order = correct_order,
    sg3_inflated  = sg3_inflated
  )
}

metrics_list = lapply(names(learners_all), function(nm)
  compute_metrics(learners_all[[nm]], nm))
names(metrics_list) = names(learners_all)

# ============================================================
# PRINT STRUCTURED COMPARISONS
# ============================================================

cat("===========================================\n")
cat("COMPARISON 1: BY META-LEARNER TYPE\n")
cat("===========================================\n\n")

for (base in c("lasso", "boost", "kern")) {
  cat("--- Base method:", toupper(base), "---\n")
  for (ml in c("r", "s", "t", "x")) {
    nm = paste0(ml, base)
    m  = metrics_list[[nm]]
    if (m$failed) {
      cat(sprintf("  %-10s | FAILED\n\n", m$learner)); next
    }
    cat(sprintf("  %-10s | NormMSE: %5.3f (%s) | RankCorr: %s (%s)\n",
                m$learner, m$norm_mse, m$quality_mse,
                ifelse(is.na(m$rank_corr), "  NA ", sprintf("%5.3f", m$rank_corr)),
                m$quality_rank))
    cat(sprintf("             | SG1 est=%6.3f true=%6.3f rec=%5.3f\n",
                m$est_sg1, m$true_sg1, m$rec_sg1))
    cat(sprintf("             | SG2 est=%6.3f true=%6.3f rec=%5.3f\n",
                m$est_sg2, m$true_sg2, m$rec_sg2))
    cat(sprintf("             | SG3 est=%6.3f true=%6.3f rec=%5.3f%s\n",
                m$est_sg3, m$true_sg3, m$rec_sg3,
                ifelse(m$sg3_inflated, " <<< SG3 INFLATED (confounding)", "")))
    cat(sprintf("             | Correct ordering (SG2>SG3>SG1): %s\n\n",
                m$correct_order))
  }
}

cat("===========================================\n")
cat("COMPARISON 2: BY BASE METHOD\n")
cat("===========================================\n\n")

for (ml in c("r", "s", "t", "x")) {
  ml_name = switch(ml, r="R-LEARNER", s="S-LEARNER",
                   t="T-LEARNER", x="X-LEARNER")
  cat("---", ml_name, "---\n")
  for (base in c("lasso", "boost", "kern")) {
    nm = paste0(ml, base)
    m  = metrics_list[[nm]]
    if (m$failed) {
      cat(sprintf("  %-10s | FAILED\n", m$learner)); next
    }
    cat(sprintf("  %-10s | NormMSE: %5.3f | RankCorr: %s | Order: %s | SG3 inflated: %s\n",
                m$learner, m$norm_mse,
                ifelse(is.na(m$rank_corr), "   NA",
                       sprintf("%5.3f", m$rank_corr)),
                ifelse(m$correct_order, "CORRECT", "WRONG"),
                ifelse(m$sg3_inflated,  "YES",     "no")))
  }
  cat("\n")
}

# Full summary table sorted by normalised MSE
cat("===========================================\n")
cat("FULL SUMMARY TABLE — ALL LEARNERS\n")
cat("Sorted by Norm_MSE (lower = better)\n")
cat("===========================================\n")

summary_df = do.call(rbind, lapply(metrics_list, function(m) {
  data.frame(
    Learner       = m$learner,
    Norm_MSE      = ifelse(m$failed, NA, m$norm_mse),
    MSE_Grade     = m$quality_mse,
    Rank_Corr     = ifelse(m$failed || is.na(m$rank_corr), "NA",
                           as.character(m$rank_corr)),
    Rank_Grade    = m$quality_rank,
    Rec_SG1       = ifelse(m$failed, NA, m$rec_sg1),
    Rec_SG2       = ifelse(m$failed, NA, m$rec_sg2),
    Rec_SG3       = ifelse(m$failed, NA, m$rec_sg3),
    Correct_Order = ifelse(m$failed, NA, m$correct_order),
    SG3_Inflated  = ifelse(m$failed, NA, m$sg3_inflated),
    stringsAsFactors = FALSE
  )
}))

summary_df = summary_df[order(as.numeric(summary_df$Norm_MSE),
                              na.last = TRUE), ]
print(summary_df, row.names = FALSE)

cat("\nInterpretation guide:\n")
cat("  Norm_MSE     < 1.0    better than predicting mean tau for everyone\n")
cat("  Rank_Corr    > 0.5    meaningful individual-level ranking\n")
cat("  Rec_SG1/2    ~ 1.0    correct subgroup magnitude\n")
cat("  Rec_SG3      > 1.1    learner is overestimating due to confounding\n")
cat("  Correct_Order TRUE    learner ranks subgroups SG2 > SG3 > SG1\n")
cat("  SG3_Inflated  YES     confounding inflating tau estimate for SG3\n")

# ============================================================
# SECTION 10: VISUALISATIONS
# ============================================================

# --- PLOT A: Propensity score distribution by arm ---
# Unique to this setup — visually demonstrates the imbalance challenge

hist(propensity[w == 0], breaks = 20,
     col  = rgb(0.2, 0.4, 0.8, 0.5),
     xlim = c(0, 0.22),
     xlab = "Propensity score",
     ylab = "Count",
     main = "Propensity by treatment arm\n(capped at 0.20)")
hist(propensity[w == 1], breaks = 10,
     col  = rgb(0.9, 0.2, 0.2, 0.5), add = TRUE)
legend("topright",
       legend = c(paste0("Control (n=", sum(w==0), ")"),
                  paste0("Treated (n=", sum(w==1), ")")),
       fill = c(rgb(0.2, 0.4, 0.8, 0.5), rgb(0.9, 0.2, 0.2, 0.5)),
       cex  = 0.8)
abline(v = mean(propensity), col = "black", lty = 2)

# --- PLOT B: KDE by meta-learner type ---
# One panel per meta-learner (R/S/T/X), three lines per panel (lasso/boost/kern)
# Black vertical line = true mean tau on test set
cols_base = c("blue", "red", "darkgreen")
ltys_base = c(1, 2, 3)

safe_kde = function(est) {
  if (all(is.na(est)) || sd(est, na.rm=TRUE) < 1e-10) return(NULL)
  density(est[!is.na(est)], n = 300)
}

par(mfrow = c(2, 2))
for (ml in c("R", "S", "T", "X")) {
  nms     = by_metalearner[[ml]]
  kdes    = lapply(learners_all[nms], safe_kde)
  valid   = !sapply(kdes, is.null)
  all_est = unlist(lapply(learners_all[nms], function(e) e[!is.na(e)]))
  
  if (length(all_est) == 0) next
  
  ymax = max(sapply(kdes[valid], function(d) max(d$y))) * 1.15
  
  plot(NULL,
       xlim = range(all_est),
       ylim = c(0, ymax),
       xlab = "Estimated tau(X) (lung decline units)",
       ylab = "Density",
       main = paste0(ml, "-Learner: lasso / boost / kern\n",
                     "(n_treated_train ~ 18-22)"))
  abline(v = 0,              col = "grey70", lwd = 1, lty = 2)
  abline(v = mean(tau_test), col = "black",  lwd = 2, lty = 1)
  
  for (k in seq_along(nms)) {
    nm = nms[k]
    if (is.null(kdes[[nm]])) {
      abline(v = mean(learners_all[[nm]], na.rm=TRUE),
             col = cols_base[k], lwd = 2, lty = ltys_base[k])
    } else {
      lines(kdes[[nm]]$x, kdes[[nm]]$y,
            col = cols_base[k], lwd = 2, lty = ltys_base[k])
    }
  }
  legend("topright",
         legend = c("True mean tau", nms),
         col    = c("black", cols_base),
         lwd    = 2, lty = c(1, ltys_base), cex = 0.65, bg = "white")
}
par(mfrow = c(1, 1))

# --- PLOT C: True vs estimated tau scatter for R-learner variants ---
r_ests = list(rlasso = rlasso_est, rboost = rboost_est, rkern = rkern_est)
r_cols = c("blue", "red", "darkgreen")

valid_r = sapply(r_ests, function(e) !all(is.na(e)))
all_r   = unlist(lapply(r_ests[valid_r], function(e) e[!is.na(e)]))

plot(NULL,
     xlim = range(tau_test),
     ylim = if (length(all_r) > 0) range(all_r) else c(0, 5),
     xlab = "True tau(X) on test set",
     ylab = "Estimated tau(X)",
     main = "True vs Estimated tau — R-learner variants\n(rare treatment: ~18-22 treated in training)")
abline(0, 1, col = "black", lwd = 2)
abline(h = mean(tau_test), col = "grey60", lty = 2)

for (k in seq_along(r_ests)) {
  est = r_ests[[k]]
  if (!all(is.na(est))) {
    points(tau_test, est,
           pch = 15 + k,
           col = adjustcolor(r_cols[k], alpha.f = 0.5),
           cex = 0.9)
  }
}
legend("topleft",
       legend = c("rlasso", "rboost", "rkern", "45-deg (perfect)"),
       col    = c(r_cols, "black"),
       pch    = c(16, 17, 18, NA),
       lty    = c(NA, NA, NA, 1),
       cex    = 0.75)

# --- PLOT D: Subgroup recovery bar chart ---
# Shows true vs estimated mean tau for each subgroup, per learner
# Easiest way to visually spot SG3 inflation
sg_learners = c("rlasso", "rkern", "slasso", "tlasso", "xlasso",
                "zero_pred", "ols_inter")

sg_true = c(true_sg1, true_sg2, true_sg3)
sg_names = c("SG1\n(low harm)", "SG2\n(high harm)", "SG3\n(confounding)")

n_learners = length(sg_learners)
n_sg       = 3
bar_width  = 0.8 / n_learners
x_pos      = seq(1, n_sg)

cols_sg = rainbow(n_learners)

plot(NULL,
     xlim = c(0.5, n_sg + 0.5),
     ylim = c(0, max(sg_true) * 1.6),
     xlab = "",
     ylab = "Mean tau estimate",
     main = "Subgroup recovery: true vs estimated mean tau\n(bar = estimated, dot = true)",
     xaxt = "n")
axis(1, at = x_pos, labels = sg_names)
abline(h = 0, col = "grey70")

for (i in seq_along(sg_learners)) {
  nm  = sg_learners[i]
  m   = metrics_list[[nm]]
  if (m$failed) next
  ests = c(m$est_sg1, m$est_sg2, m$est_sg3)
  offsets = (i - (n_learners + 1) / 2) * bar_width
  rect(x_pos + offsets - bar_width/2, 0,
       x_pos + offsets + bar_width/2, ests,
       col = adjustcolor(cols_sg[i], alpha.f = 0.6),
       border = NA)
}

# True tau as solid dots
points(x_pos, sg_true, pch = 16, cex = 1.8, col = "black")

legend("topright",
       legend = c(sg_learners, "True tau"),
       fill   = c(adjustcolor(cols_sg, alpha.f = 0.6), NA),
       pch    = c(rep(NA, n_learners), 16),
       col    = c(rep(NA, n_learners), "black"),
       border = NA, cex = 0.65, bg = "white")

# ============================================================
# SECTION 11: EXPORT RESULTS TO JSON
# ============================================================

cat("\n=== EXPORTING RESULTS TO JSON ===\n")

study_meta = list(
  study          = "Smoking and Longevity — Rare Treatment Imbalance",
  outcome        = "10-year lung function decline (higher = worse)",
  treatment      = "Heavy long-term smoking (W=1) vs non-heavy smoker (W=0)",
  key_challenge  = "Severe treatment imbalance (~7.5% treated)",
  n_total        = n,
  n_train        = length(train_idx),
  n_test         = length(test_idx),
  n_treated_train= sum(w_train),
  n_treated_test = sum(w_test),
  treatment_rate = round(mean(w), 3),
  seed           = 42,
  true_ate       = round(mean(tau_x),    4),
  true_ate_test  = round(mean(tau_test), 4),
  tau_sd         = round(sd(tau_x),      4)
)

subgroup_truth = list(
  SG1 = list(
    definition = "age<55 & lung>75 & stress<4 & literacy>14",
    clinical   = "Young, good lungs, low stress, health-literate — lowest harm",
    n          = sum(sg1),
    true_tau   = round(true_sg1, 4)
  ),
  SG2 = list(
    definition = "age>70 & lung<55 & stress>7",
    clinical   = "Elderly, impaired lungs, high stress — highest harm",
    n          = sum(sg2),
    true_tau   = round(true_sg2, 4)
  ),
  SG3 = list(
    definition = "income<40 & literacy<5",
    clinical   = "Low income, low literacy — confounding trap",
    n          = sum(sg3),
    true_tau   = round(true_sg3, 4)
  )
)

learner_results = lapply(metrics_list, function(m) {
  if (m$failed) return(list(learner = m$learner, failed = TRUE))
  list(
    learner       = m$learner,
    failed        = FALSE,
    norm_mse      = m$norm_mse,
    mse_grade     = m$quality_mse,
    rank_corr     = m$rank_corr,
    rank_grade    = m$quality_rank,
    correct_order = m$correct_order,
    subgroups     = list(
      SG1 = list(estimated = m$est_sg1, true = m$true_sg1,
                 recovery  = m$rec_sg1),
      SG2 = list(estimated = m$est_sg2, true = m$true_sg2,
                 recovery  = m$rec_sg2),
      SG3 = list(estimated    = m$est_sg3, true = m$true_sg3,
                 recovery     = m$rec_sg3,
                 abs_deviation= m$dev_sg3,
                 inflated     = m$sg3_inflated)
    )
  )
})

results_json = list(
  metadata       = study_meta,
  subgroup_truth = subgroup_truth,
  learner_results= learner_results
)

json_path = "smoking_results_1000.json"
write(toJSON(results_json, auto_unbox = TRUE), json_path)
cat("Results exported to:", json_path, "\n")

cat("\nDone. Key summary:\n")
cat("  True ATE (test):", round(mean(tau_test), 3), "\n")
cat("  Treatment rate: ", round(mean(w), 3), "\n")
cat("  n_treated_train:", sum(w_train), "\n")
non_failed = summary_df[!is.na(summary_df$Norm_MSE), ]
if (nrow(non_failed) > 0)
  cat("  Best learner:   ", non_failed$Learner[1],
      "(NormMSE =", non_failed$Norm_MSE[1], ")\n")
inflated = summary_df$Learner[summary_df$SG3_Inflated %in% TRUE]
cat("  Learners with SG3 inflation (confounding detected):",
    ifelse(length(inflated) > 0, paste(inflated, collapse=", "), "none"), "\n")
correct_ord = summary_df$Learner[summary_df$Correct_Order %in% TRUE]
cat("  Learners with correct severity ordering:",
    ifelse(length(correct_ord) > 0, paste(correct_ord, collapse=", "), "none"), "\n")

# n=300 results
#  True ATE (test): 2.593 
#  Treatment rate:  0.05 
#  n_treated_train: 12 
#  Best learner:    xboost (NormMSE = 1.0601 )
#  Learners with SG3 inflation (confounding detected): xboost, rlasso, xlasso, tlasso
#  Learners with correct severity ordering: xboost

# n=500 results
#  True ATE (test): 2.54 
#  Treatment rate:  0.058 
#  n_treated_train: 24
#  Best learner:    xkern (NormMSE = 0.6967 )
#  Learners with SG3 inflation (confounding detected): rboost, xboost 
#  Learners with correct severity ordering: skern 






