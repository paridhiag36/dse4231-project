# ============================================================
# TELEMEDICINE SETUP — SINGLE ITERATION
# Outcome: Raw systolic BP at 12 months (lower = better)
# Treatment: Telemedicine enrolment (W = 1) vs in-person (W = 0)
# Key feature: tau(X) has sign change across patient subgroups
# Population average treatment effect is close to zero
#
# CHANGES FROM PREVIOUS VERSION:
#   - n increased to 250 (200 train / 50 test)
#   - Learners fitted on training set, evaluated on test set
#   - Pre-specified clinical subgroup thresholds
#   - All 12 learners + baselines in unified list
#   - Consistent boosting settings across all learners
#   - Y floored at 80 mmHg
#   - ATE verification check
#   - Summary table consolidated
#   - JSON output for results
# ============================================================

library(MASS)       # mvrnorm — correlated covariate generation
library(rlearner)   # all learners
library(KRLS2)      # kernel ridge regression
library(jsonlite)   # JSON export
library(rjson)

set.seed(42)
n = 250

# ============================================================
# SECTION 1: COVARIATE GENERATION
# X1: baseline systolic BP (mmHg)       — range 110 to 190
# X2: travel time to clinic (minutes)   — range 20 to 80
# X3: prior digital health interactions — range 0 to 20 (count)
# X4: number of comorbidities           — range 0 to 5
# X5: age (years)                       — range 40 to 80
#
# Correlation structure:
#   age <-> comorbidities:         +0.45
#   age <-> digital engagement:    -0.40
#   baseline BP <-> comorbidities: +0.35
#   baseline BP <-> age:           +0.25
#   travel time largely independent
#   comorbidities <-> digital:     -0.20
# ============================================================

cor_matrix = matrix(c(
  # X1_bp  X2_travel  X3_digital  X4_comorbid  X5_age
  1.00,    0.05,      -0.10,       0.35,        0.25,
  0.05,    1.00,       0.05,       0.00,       -0.05,
  -0.10,    0.05,       1.00,      -0.20,       -0.40,
  0.35,    0.00,      -0.20,       1.00,        0.45,
  0.25,   -0.05,      -0.40,       0.45,        1.00
), nrow = 5, byrow = TRUE)

z = mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)

x1_bp       = 110 + 80 * pnorm(z[,1])
x2_travel   = 20  + 60 * pnorm(z[,2])
x3_digital  = round(20 * pnorm(z[,3]))
x4_comorbid = round(5  * pnorm(z[,4]))
x5_age      = round(40 + 40 * pnorm(z[,5]))

x = cbind(x1_bp, x2_travel, x3_digital, x4_comorbid, x5_age)
colnames(x) = c("bp_baseline", "travel_time", "digital_prior",
                "comorbidities", "age")

cat("=== COVARIATE SUMMARY ===\n")
print(summary(x))
cat("\n=== COVARIATE CORRELATION ===\n")
print(round(cor(x), 2))

# ============================================================
# SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
# ============================================================

x1_s = as.numeric(scale(x1_bp))
x2_s = as.numeric(scale(x2_travel))
x3_s = as.numeric(scale(x3_digital))
x4_s = as.numeric(scale(x4_comorbid))
x5_s = as.numeric(scale(x5_age))

log_odds_w = -0.5 +
  1.5 * x2_s +
  0.5 * x1_s +
  -0.4 * x5_s

propensity = pmax(0.05, pmin(plogis(log_odds_w), 0.95))
w          = rbinom(n, 1, propensity)

cat("\n=== TREATMENT ASSIGNMENT ===\n")
cat("Proportion enrolled:", round(mean(w), 3), "\n")
cat("Propensity range:",
    round(min(propensity), 3), "to", round(max(propensity), 3), "\n")
cat("Mean propensity:", round(mean(propensity), 3), "\n")
cat("Mean travel time — treated:", round(mean(x2_travel[w==1]), 1),
    "| control:", round(mean(x2_travel[w==0]), 1), "\n")
cat("Mean baseline BP — treated:", round(mean(x1_bp[w==1]), 1),
    "| control:", round(mean(x1_bp[w==0]), 1), "\n")
cat("Mean age         — treated:", round(mean(x5_age[w==1]), 1),
    "| control:", round(mean(x5_age[w==0]), 1), "\n")

# ============================================================
# SECTION 3: TRUE TREATMENT EFFECT tau(X)
# Sign change logic:
#   X2 travel:       negative — remote patients benefit (BP reduced)
#   X1 baseline BP:  positive — uncontrolled patients harmed
#   X4 comorbidities:positive — complex patients harmed
#   X3 digital:      negative — digitally engaged patients benefit
#   X5 age:          positive — older patients slightly harmed
#   Constant -0.7:   calibrates population mean towards zero
# ============================================================

tau_x = -2.0 * x2_s +
  2.0 * x1_s +
  1.5 * x4_s +
  -0.8 * x3_s +
  0.5 * x5_s +
  -0.7

cat("\n=== TRUE TREATMENT EFFECT SUMMARY ===\n")
cat("Mean tau(X):", round(mean(tau_x), 3), "(should be close to zero)\n")
cat("SD tau(X):  ", round(sd(tau_x), 3), "\n")
cat("Range:      ", round(min(tau_x), 3), "to", round(max(tau_x), 3), "\n")
cat("Proportion negative (beneficial):", round(mean(tau_x < 0), 3), "\n")
cat("Proportion positive (harmful):   ", round(mean(tau_x > 0), 3), "\n")

# ATE verification — flag if mean strays too far from zero
if (abs(mean(tau_x)) > 1.0) {
  warning("True ATE = ", round(mean(tau_x), 2),
          " mmHg — further from zero than intended.",
          " Consider adjusting calibration constant in tau_x.")
}

# ============================================================
# SECTION 4: BASELINE OUTCOME b(X)
# ============================================================

b_x = 0.6  * x1_bp       +
  2.5  * x4_comorbid  +
  0.3  * x5_age       +
  -1.2  * x3_digital   +
  20

cat("\n=== BASELINE OUTCOME b(X) ===\n")
cat("Mean b(X):", round(mean(b_x), 2), "mmHg\n")
cat("Range:    ", round(min(b_x), 2), "to", round(max(b_x), 2), "mmHg\n")

# ============================================================
# SECTION 5: OBSERVED OUTCOME Y
# Floored at 80 mmHg — systolic BP cannot fall below this
# ============================================================

sigma   = 5
epsilon = rnorm(n, mean = 0, sd = sigma)
y_raw   = b_x + (w - propensity) * tau_x + epsilon
y       = pmax(y_raw, 80)

cat("\n=== OBSERVED OUTCOME Y ===\n")
cat("Proportion floored at 80 mmHg:", round(mean(y_raw < 80), 3), "\n")
cat("Mean Y (all):    ", round(mean(y), 2), "mmHg\n")
cat("Mean Y treated:  ", round(mean(y[w==1]), 2), "mmHg\n")
cat("Mean Y control:  ", round(mean(y[w==0]), 2), "mmHg\n")
cat("Naive ATE:       ", round(mean(y[w==1]) - mean(y[w==0]), 3),
    "mmHg (confounded)\n")
cat("True ATE:        ", round(mean(tau_x), 3), "mmHg\n")

# ============================================================
# SECTION 6: TRAIN-TEST SPLIT
# 200 train / 50 test
# Learners fitted on train only
# All evaluation on held-out test set
# ============================================================

set.seed(42)
train_idx = sample(1:n, size = 200, replace = FALSE)
test_idx  = setdiff(1:n, train_idx)

x_train = x[train_idx, ];  x_test  = x[test_idx, ]
w_train = w[train_idx];    w_test  = w[test_idx]
y_train = y[train_idx];    y_test  = y[test_idx]
tau_test = tau_x[test_idx]

cat("\n=== TRAIN-TEST SPLIT ===\n")
cat("Train size:", length(train_idx),
    "| Treatment rate:", round(mean(w_train), 3), "\n")
cat("Test size: ", length(test_idx),
    "| Treatment rate:", round(mean(w_test),  3), "\n")

# ============================================================
# SECTION 7: PRE-SPECIFIED CLINICAL SUBGROUPS (TEST SET)
#
# Thresholds are clinically grounded — not data-driven quantiles
# This prevents circularity between training and evaluation
#
# SG1 — Young, remote, digitally engaged (expect to benefit)
#   Age < 55 years (working-age adult)
#   Travel time > 45 minutes (genuinely burdensome journey)
#   Digital interactions > 5 (demonstrated regular digital use)
#
# SG2 — Severely ill, complex (expect to be harmed)
#   Baseline BP > 160 mmHg (Stage 2 hypertension by guidelines)
#   Comorbidities > 2 (3+ conditions = high clinical complexity)
#
# SG3 — Elderly, low digital engagement (near-zero / mildly harmful)
#   Age > 70 years (elderly by clinical definition)
#   Digital interactions < 2 (essentially no digital engagement)
# ============================================================

x1_bp_test      = x_test[, "bp_baseline"]
x2_travel_test  = x_test[, "travel_time"]
x3_digital_test = x_test[, "digital_prior"]
x4_comorbid_test= x_test[, "comorbidities"]
x5_age_test     = x_test[, "age"]

sg1 = x5_age_test     <  55 &
  x2_travel_test  >  45 &
  x3_digital_test >  5

sg2 = x1_bp_test      > 160 &
  x4_comorbid_test > 2

sg3 = x5_age_test     >  70 &
  x3_digital_test <  2

min_sg_size = 10

cat("\n=== SUBGROUP SIZES (TEST SET, PRE-SPECIFIED THRESHOLDS) ===\n")
cat("SG1 (age<55 & travel>45min & digital>5): ", sum(sg1), "patients\n")
cat("SG2 (BP>160mmHg & comorbidities>2):      ", sum(sg2), "patients\n")
cat("SG3 (age>70 & digital<2):                ", sum(sg3), "patients\n\n")

if (sum(sg1) < min_sg_size)
  warning("SG1 < ", min_sg_size, " patients — consider relaxing: age<60 or travel>40 or digital>3")
if (sum(sg2) < min_sg_size)
  warning("SG2 < ", min_sg_size, " patients — consider relaxing: BP>155 or comorbidities>1")
if (sum(sg3) < min_sg_size)
  warning("SG3 < ", min_sg_size, " patients — consider relaxing: age>65 or digital<3")

true_sg1 = mean(tau_test[sg1])
true_sg2 = mean(tau_test[sg2])
true_sg3 = mean(tau_test[sg3])

cat("True tau by subgroup (test set):\n")
cat("  SG1:", round(true_sg1, 3), "(expect negative — beneficial)\n")
cat("  SG2:", round(true_sg2, 3), "(expect positive — harmful)\n")
cat("  SG3:", round(true_sg3, 3), "(expect near zero or mildly positive)\n")

# ============================================================
# SECTION 8: FIT ALL LEARNERS ON TRAINING DATA
# Consistent boosting settings across all learners
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
  k_folds_mu1               = 5,
  k_folds_mu0               = 5,
  ntrees_max            = 300,
  early_stopping_rounds = 5,
  verbose               = FALSE
)

fit_time_start = Sys.time()
cat("\n=== FITTING LEARNERS ON TRAINING DATA (n=200) ===\n")

cat("R-learner: rlasso...\n")
rlasso_fit = rlasso(x_train, w_train, y_train)
rlasso_est = predict(rlasso_fit, x_test)

cat("R-learner: rboost...\n")
rboost_fit = do.call(rboost, c(list(x=x_train, w=w_train, y=y_train), boost_args))
rboost_est = predict(rboost_fit, x_test)

cat("R-learner: rkern...\n")
rkern_fit  = rkern(x_train, w_train, y_train)
rkern_est  = predict(rkern_fit, x_test)

cat("S-learner: slasso...\n")
slasso_fit = slasso(x_train, w_train, y_train)
slasso_est = predict(slasso_fit, x_test)

cat("S-learner: sboost...\n")
sboost_fit = do.call(sboost, c(list(x=x_train, w=w_train, y=y_train), boost_args))
sboost_est = predict(sboost_fit, x_test)

cat("S-learner: skern...\n")
skern_fit  = skern(x_train, w_train, y_train)
skern_est  = predict(skern_fit, x_test)

cat("T-learner: tlasso...\n")
tlasso_fit = tlasso(x_train, w_train, y_train)
tlasso_est = predict(tlasso_fit, x_test)

cat("T-learner: tboost...\n")
tboost_fit = do.call(tboost, c(list(x=x_train, w=w_train, y=y_train), boost_args_others))
tboost_est = predict(tboost_fit, x_test)

cat("T-learner: tkern...\n")
tkern_fit  = tkern(x_train, w_train, y_train)
tkern_est  = predict(tkern_fit, x_test)

cat("X-learner: xlasso...\n")
xlasso_fit = xlasso(x_train, w_train, y_train)
xlasso_est = predict(xlasso_fit, x_test)

cat("X-learner: xboost...\n")
xboost_fit = do.call(xboost, c(list(x=x_train, w=w_train, y=y_train), boost_args_others))
xboost_est = predict(xboost_fit, x_test)

cat("X-learner: xkern...\n")
xkern_fit  = xkern(x_train, w_train, y_train)
xkern_est  = predict(xkern_fit, x_test)

# --- Baselines fitted on training data, evaluated on test ---
zero_pred = rep(0, length(test_idx))

x_train_df = as.data.frame(x_train)
colnames(x_train_df) = c("bp", "travel", "digital", "comorbid", "age")
x_test_df  = as.data.frame(x_test)
colnames(x_test_df)  = c("bp", "travel", "digital", "comorbid", "age")

ols_train_data = data.frame(
  y           = y_train,
  w           = w_train,
  x_train_df,
  w_bp        = w_train * x_train_df$bp,
  w_travel    = w_train * x_train_df$travel,
  w_digital   = w_train * x_train_df$digital,
  w_comorbid  = w_train * x_train_df$comorbid,
  w_age       = w_train * x_train_df$age
)

ols_fit = lm(y ~ bp + travel + digital + comorbid + age +
               w + w_bp + w_travel + w_digital + w_comorbid + w_age,
             data = ols_train_data)

ols_tau_est = coef(ols_fit)["w"] +
  coef(ols_fit)["w_bp"]       * x_test_df$bp +
  coef(ols_fit)["w_travel"]   * x_test_df$travel +
  coef(ols_fit)["w_digital"]  * x_test_df$digital +
  coef(ols_fit)["w_comorbid"] * x_test_df$comorbid +
  coef(ols_fit)["w_age"]      * x_test_df$age

fit_time_end = Sys.time()
cat("\nTotal fitting time:", round(as.numeric(fit_time_end - fit_time_start, units="mins"), 2), "minutes\n")

# Unified learner list — all evaluated on test set
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

# ============================================================
# SECTION 9: EVALUATION ON TEST SET
# All metrics computed against tau_test (true tau on test set)
# ============================================================

tau_variance = var(tau_test)

cat("\n=== EVALUATION ON TEST SET (n=50) ===\n")
cat("All metrics computed on held-out test data\n")
cat("Variance of true tau(X) on test set:", round(tau_variance, 4), "\n\n")

# Helper function to compute all metrics for one learner
compute_metrics = function(est, name, tau_true, sg1, sg2, sg3,
                           true_sg1, true_sg2, true_sg3,
                           tau_var) {
  
  raw_mse   = mean((est - tau_true)^2)
  norm_mse  = raw_mse / tau_var
  
  rank_corr = ifelse(sd(est) < 1e-10, NA,
                     cor(est, tau_true, method = "spearman"))
  
  quality_mse = ifelse(norm_mse < 0.25, "EXCELLENT",
                       ifelse(norm_mse < 0.75, "ACCEPTABLE",
                              ifelse(norm_mse < 1.00, "POOR", "WORSE THAN MEAN")))
  
  quality_rank = ifelse(is.na(rank_corr), "UNDEFINED",
                        ifelse(rank_corr > 0.80, "RELIABLE",
                               ifelse(rank_corr > 0.50, "MODERATE",
                                      ifelse(rank_corr > 0.30, "WEAK", "RANDOM"))))
  
  est_sg1 = mean(est[sg1])
  est_sg2 = mean(est[sg2])
  est_sg3 = mean(est[sg3])
  
  rec_sg1 = est_sg1 / true_sg1
  rec_sg2 = est_sg2 / true_sg2
  rec_sg3 = ifelse(abs(true_sg3) < 0.1, NA, est_sg3 / true_sg3)
  
  dev_sg3 = abs(est_sg3 - true_sg3)
  
  sign_sg1 = sign(est_sg1) == sign(true_sg1)
  sign_sg2 = sign(est_sg2) == sign(true_sg2)
  
  list(
    learner      = name,
    raw_mse      = round(raw_mse,   4),
    norm_mse     = round(norm_mse,  4),
    quality_mse  = quality_mse,
    rank_corr    = ifelse(is.na(rank_corr), NA, round(rank_corr, 4)),
    quality_rank = quality_rank,
    est_sg1      = round(est_sg1, 3),
    true_sg1     = round(true_sg1, 3),
    rec_sg1      = round(rec_sg1, 3),
    sign_sg1     = sign_sg1,
    est_sg2      = round(est_sg2, 3),
    true_sg2     = round(true_sg2, 3),
    rec_sg2      = round(rec_sg2, 3),
    sign_sg2     = sign_sg2,
    est_sg3      = round(est_sg3, 3),
    true_sg3     = round(true_sg3, 3),
    rec_sg3      = ifelse(is.na(rec_sg3), NA, round(rec_sg3, 3)),
    dev_sg3      = round(dev_sg3, 3)
  )
}

metrics_list = mapply(
  compute_metrics,
  learners_all,
  names(learners_all),
  MoreArgs = list(
    tau_true  = tau_test,
    sg1       = sg1,
    sg2       = sg2,
    sg3       = sg3,
    true_sg1  = true_sg1,
    true_sg2  = true_sg2,
    true_sg3  = true_sg3,
    tau_var   = tau_variance
  ),
  SIMPLIFY = FALSE
)

# ============================================================
# PRINT STRUCTURED COMPARISON 1: BY META-LEARNER TYPE
# Fixed base method, vary R vs S vs T vs X
# ============================================================

cat("===========================================\n")
cat("COMPARISON 1: BY META-LEARNER TYPE\n")
cat("Within each base method — R vs S vs T vs X\n")
cat("===========================================\n\n")

for (base in c("lasso", "boost", "kern")) {
  cat("--- Base method:", toupper(base), "---\n")
  for (ml in c("r", "s", "t", "x")) {
    nm = paste0(ml, base)
    m  = metrics_list[[nm]]
    cat(sprintf("  %-10s | MSE: %7.4f | NormMSE: %5.3f | RankCorr: %s (%s)\n",
                m$learner,
                m$raw_mse, m$norm_mse,
                ifelse(is.na(m$rank_corr), "  NA ", sprintf("%5.3f", m$rank_corr)),
                m$quality_rank))
    cat(sprintf("             | SG1: est=%6.3f true=%6.3f rec=%5.3f sign=%s\n",
                m$est_sg1, m$true_sg1, m$rec_sg1,
                ifelse(m$sign_sg1, "CORRECT", "WRONG")))
    cat(sprintf("             | SG2: est=%6.3f true=%6.3f rec=%5.3f sign=%s\n",
                m$est_sg2, m$true_sg2, m$rec_sg2,
                ifelse(m$sign_sg2, "CORRECT", "WRONG")))
    cat(sprintf("             | SG3: est=%6.3f true=%6.3f dev=%5.3f\n\n",
                m$est_sg3, m$true_sg3, m$dev_sg3))
  }
}

# ============================================================
# PRINT STRUCTURED COMPARISON 2: BY BASE METHOD
# Fixed meta-learner, vary lasso vs boost vs kern
# ============================================================

cat("===========================================\n")
cat("COMPARISON 2: BY BASE METHOD\n")
cat("Within each meta-learner — lasso vs boost vs kern\n")
cat("===========================================\n\n")

for (ml in c("r", "s", "t", "x")) {
  ml_name = switch(ml, r="R-LEARNER", s="S-LEARNER",
                   t="T-LEARNER", x="X-LEARNER")
  cat("---", ml_name, "---\n")
  for (base in c("lasso", "boost", "kern")) {
    nm = paste0(ml, base)
    m  = metrics_list[[nm]]
    cat(sprintf("  %-10s | MSE: %7.4f | NormMSE: %5.3f | RankCorr: %s | Sign SG1: %s | Sign SG2: %s\n",
                m$learner,
                m$raw_mse, m$norm_mse,
                ifelse(is.na(m$rank_corr), "   NA", sprintf("%5.3f", m$rank_corr)),
                ifelse(m$sign_sg1, "CORRECT", "WRONG"),
                ifelse(m$sign_sg2, "CORRECT", "WRONG")))
  }
  cat("\n")
}

# ============================================================
# FULL SUMMARY TABLE
# ============================================================

cat("===========================================\n")
cat("FULL SUMMARY TABLE — ALL LEARNERS\n")
cat("Evaluated on test set (n=50)\n")
cat("===========================================\n")

summary_df = do.call(rbind, lapply(metrics_list, function(m) {
  data.frame(
    Learner   = m$learner,
    MSE       = m$raw_mse,
    Norm_MSE  = m$norm_mse,
    MSE_Grade = m$quality_mse,
    Rank_Corr = ifelse(is.na(m$rank_corr), "NA",
                       as.character(m$rank_corr)),
    Rank_Grade= m$quality_rank,
    Rec_SG1   = m$rec_sg1,
    Sign_SG1  = m$sign_sg1,
    Rec_SG2   = m$rec_sg2,
    Sign_SG2  = m$sign_sg2,
    Dev_SG3   = m$dev_sg3,
    stringsAsFactors = FALSE
  )
}))

summary_df = summary_df[order(summary_df$MSE), ]
print(summary_df, row.names = FALSE)

cat("\nInterpretation guide:\n")
cat("  MSE: raw mean squared error between estimated and true tau\n")
cat("  Norm_MSE  < 1.0 means better than constant zero predictor\n")
cat("  Rank_Corr > 0.5 means meaningful individual ranking\n")
cat("  Rec_SG1/2 close to 1.0 means correct subgroup magnitude\n")
cat("  Sign_SG1/2 = TRUE means correct direction of effect\n")
cat("  Dev_SG3: absolute deviation from true SG3 effect (lower = better)\n")

# ============================================================
# SECTION 10: VISUALISATIONS
# ============================================================

# --- Colour and line type scheme ---
learner_colours = c(
  rlasso="blue",  rboost="blue",  rkern="blue",
  slasso="red",   sboost="red",   skern="red",
  tlasso="darkgreen", tboost="darkgreen", tkern="darkgreen",
  xlasso="purple",xboost="purple",xkern="purple",
  zero_pred="grey50", ols_inter="orange"
)
learner_lty = c(
  rlasso=1, rboost=2, rkern=3,
  slasso=1, sboost=2, skern=3,
  tlasso=1, tboost=2, tkern=3,
  xlasso=1, xboost=2, xkern=3,
  zero_pred=5, ols_inter=4
)

# Helper: safe KDE
safe_kde = function(est) {
  if (sd(est) < 1e-10) return(NULL)
  density(est, n = 500)
}

# --- PLOT A: KDE by meta-learner type ---
par(mfrow = c(2, 2))
for (ml in c("R", "S", "T", "X")) {
  nms = by_metalearner[[ml]]
  all_est = unlist(learners_all[nms])
  kdes    = lapply(learners_all[nms], safe_kde)
  ymax    = max(sapply(kdes[!sapply(kdes, is.null)],
                       function(d) max(d$y))) * 1.15
  
  plot(NULL,
       xlim = c(min(all_est)-0.2, max(all_est)+0.2),
       ylim = c(0, ymax),
       xlab = "Estimated tau(X) (mmHg)",
       ylab = "Density",
       main = paste0(ml, "-Learner: lasso / boost / kern"))
  
  abline(v = 0, col = "grey70", lwd = 1, lty = 2)
  abline(v = mean(tau_test), col = "black", lwd = 2, lty = 1)
  
  cols = c("blue", "red", "darkgreen")
  ltys = c(1, 2, 3)
  for (k in seq_along(nms)) {
    nm = nms[k]
    if (is.null(kdes[[nm]])) {
      abline(v = mean(learners_all[[nm]]),
             col = cols[k], lwd = 2, lty = ltys[k])
    } else {
      lines(kdes[[nm]]$x, kdes[[nm]]$y,
            col = cols[k], lwd = 2, lty = ltys[k])
    }
  }
  legend("topright",
         legend = c("Mean true tau", nms),
         col    = c("black", cols),
         lwd    = 2, lty = c(1, ltys), cex = 0.65, bg = "white")
}
par(mfrow = c(1, 1))

# --- PLOT B: PDP for R-learner variants on test set ---
covariate_labels = c("Baseline BP (mmHg)", "Travel Time (min)",
                     "Digital Interactions", "Comorbidities", "Age (years)")
covariate_vals_test = list(x1_bp_test, x2_travel_test, x3_digital_test,
                           x4_comorbid_test, x5_age_test)

par(mfrow = c(2, 3))
all_r_est = c(rlasso_est, rboost_est, rkern_est, tau_test)
y_range_r = range(all_r_est)

for (j in 1:5) {
  cov_vals = covariate_vals_test[[j]]
  ord      = order(cov_vals)
  
  plot(NULL,
       xlim = range(cov_vals),
       ylim = y_range_r,
       xlab = covariate_labels[j],
       ylab = "Treatment effect tau (mmHg)",
       main = paste("PDP:", covariate_labels[j]))
  
  abline(h = 0, col = "grey70", lwd = 1, lty = 2)
  lines(cov_vals[ord], tau_test[ord],    col = "black",     lwd = 2, lty = 1)
  lines(cov_vals[ord], rlasso_est[ord],  col = "blue",      lwd = 2, lty = 2)
  lines(cov_vals[ord], rboost_est[ord],  col = "red",       lwd = 2, lty = 3)
  lines(cov_vals[ord], rkern_est[ord],   col = "darkgreen", lwd = 2, lty = 4)
  
  legend("topleft",
         legend = c("True tau", "rlasso", "rboost", "rkern"),
         col    = c("black", "blue", "red", "darkgreen"),
         lwd    = 2, lty = c(1,2,3,4), cex = 0.6, bg = "white")
}

# Scatter: true vs estimated (R-learner variants)
plot(NULL,
     xlim = range(tau_test),
     ylim = range(c(rlasso_est, rboost_est, rkern_est)),
     xlab = "True tau(X) on test set (mmHg)",
     ylab = "Estimated tau_hat(X) (mmHg)",
     main = "True vs Estimated tau\nR-learner variants")
abline(0, 1, col = "black", lwd = 2)
abline(h = 0, col = "grey60", lty = 2)
abline(v = 0, col = "grey60", lty = 2)
points(tau_test, rlasso_est, pch=16, col=rgb(0,0,1,0.4), cex=0.9)
points(tau_test, rboost_est, pch=17, col=rgb(1,0,0,0.4), cex=0.9)
points(tau_test, rkern_est,  pch=18, col=rgb(0,0.6,0,0.4), cex=0.9)
legend("topleft",
       legend = c("rlasso","rboost","rkern","45-deg line"),
       col    = c("blue","red","darkgreen","black"),
       pch    = c(16,17,18,NA), lty=c(NA,NA,NA,1), cex=0.7)

par(mfrow = c(1, 1))

# ============================================================
# SECTION 11: EXPORT RESULTS TO JSON
# ============================================================

cat("\n=== EXPORTING RESULTS TO JSON ===\n")

# Study metadata
study_meta = list(
  study        = "Telemedicine — Hypertension Follow-up",
  outcome      = "Systolic BP at 12 months (mmHg, lower = better)",
  treatment    = "Telemedicine enrolment vs in-person care",
  key_feature  = "Sign change in tau(X) — beneficial for some, harmful for others",
  n_total      = n,
  n_train      = length(train_idx),
  n_test       = length(test_idx),
  seed         = 42,
  true_ate     = round(mean(tau_x), 4),
  true_ate_test= round(mean(tau_test), 4),
  tau_sd       = round(sd(tau_x), 4),
  pct_beneficial = round(mean(tau_x < 0), 3),
  pct_harmful    = round(mean(tau_x > 0), 3)
)

# Subgroup ground truth
subgroup_truth = list(
  SG1 = list(
    definition  = "Age < 55 & Travel > 45min & Digital > 5",
    clinical    = "Young, remote, digitally engaged — expected to benefit",
    n           = sum(sg1),
    true_tau    = round(true_sg1, 4)
  ),
  SG2 = list(
    definition  = "BP > 160mmHg & Comorbidities > 2",
    clinical    = "Severely ill, complex — expected to be harmed",
    n           = sum(sg2),
    true_tau    = round(true_sg2, 4)
  ),
  SG3 = list(
    definition  = "Age > 70 & Digital < 2",
    clinical    = "Elderly, low digital engagement — near-zero or mildly harmful",
    n           = sum(sg3),
    true_tau    = round(true_sg3, 4)
  )
)

# Per-learner results
learner_results = lapply(metrics_list, function(m) {
  list(
    learner      = m$learner,
    raw_mse      = m$raw_mse,
    norm_mse     = m$norm_mse,
    mse_grade    = m$quality_mse,
    rank_corr    = m$rank_corr,
    rank_grade   = m$quality_rank,
    subgroups    = list(
      SG1 = list(
        estimated    = m$est_sg1,
        true         = m$true_sg1,
        recovery     = m$rec_sg1,
        correct_sign = m$sign_sg1
      ),
      SG2 = list(
        estimated    = m$est_sg2,
        true         = m$true_sg2,
        recovery     = m$rec_sg2,
        correct_sign = m$sign_sg2
      ),
      SG3 = list(
        estimated    = m$est_sg3,
        true         = m$true_sg3,
        abs_deviation= m$dev_sg3
      )
    )
  )
})

# Comparison tables as flat data frames
comparison_by_metalearner = lapply(names(by_metalearner), function(ml) {
  nms = by_metalearner[[ml]]
  lapply(nms, function(nm) {
    m = metrics_list[[nm]]
    list(learner    = nm,
         base       = sub("^[rstx]", "", nm),
         metalearner= ml,
         raw_mse    = m$raw_mse,
         norm_mse   = m$norm_mse,
         rank_corr  = m$rank_corr,
         sign_sg1   = m$sign_sg1,
         sign_sg2   = m$sign_sg2)
  })
})
names(comparison_by_metalearner) = names(by_metalearner)

comparison_by_base = lapply(names(by_base_method), function(base) {
  nms = by_base_method[[base]]
  lapply(nms, function(nm) {
    m = metrics_list[[nm]]
    list(learner    = nm,
         metalearner= toupper(substr(nm, 1, 1)),
         base       = base,
         raw_mse    = m$raw_mse,
         norm_mse   = m$norm_mse,
         rank_corr  = m$rank_corr,
         sign_sg1   = m$sign_sg1,
         sign_sg2   = m$sign_sg2)
  })
})
names(comparison_by_base) = names(by_base_method)

# Assemble full JSON object
results_json = list(
  metadata              = study_meta,
  subgroup_truth        = subgroup_truth,
  learner_results       = learner_results,
  comparison_by_metalearner = comparison_by_metalearner,
  comparison_by_base    = comparison_by_base
)

# Write to file
json_path = "telemed_results.json"
write(toJSON(results_json), json_path)
cat("Results exported to:", json_path, "\n")

cat("\nDone. Key findings:\n")
cat("  True ATE (test):     ", round(mean(tau_test), 3), "mmHg\n")
cat("  Best MSE learner:    ", summary_df$Learner[1], "(",
    summary_df$MSE[1], ")\n")
cat("  Learners with correct SG1 AND SG2 sign:\n")
correct_both = summary_df$Learner[summary_df$Sign_SG1 == TRUE &
                                    summary_df$Sign_SG2 == TRUE]
cat("   ", paste(correct_both, collapse = ", "), "\n")