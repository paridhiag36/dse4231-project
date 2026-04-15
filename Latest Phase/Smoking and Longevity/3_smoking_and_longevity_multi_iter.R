# ============================================================
# SMOKING AND LONGEVITY — MULTI-ITERATION VERSION
#
# Base setup preserved from 2_smoking_and_longevity_all.R:
#   - same covariate generation
#   - same propensity function and treatment assignment logic
#   - same tau(X), b(X), sigma
#   - same train/test split structure
#   - same clinical subgroup definitions (SG1, SG2, SG3)
#   - same learner set and zero_pred baseline
#
# Additions adapted from later multi-iteration files:
#   - 50 iterations for n = 500
#
# Output saved as CSV to:
#   Latest Phase/Smoking and Longevity/
# ============================================================

library(MASS)
library(rlearner)
library(KRLS2)
library(future)
library(furrr)

# ============================================================
# CONFIGURATION
# ============================================================

set.seed(42)
N_ITER     = 50
N_TOTAL    = 500
TRAIN_FRAC = 0.80
SIGMA_Y    = 3
TOP_K_FRAC = 0.075   # evaluate top 7.5% highest-harm patients
OUTPUT_DIR = "Latest Phase/Smoking and Longevity"

N_TRAIN = floor(TRAIN_FRAC * N_TOTAL)
N_TEST  = N_TOTAL - N_TRAIN

BOOST_ARGS = list(
  num_search_rounds = 5, k_folds = 5,
  ntrees_max = 100, early_stopping_rounds = 5, verbose = FALSE
)
BOOST_ARGS_TX = list(
  num_search_rounds = 5, k_folds_mu1 = 5, k_folds_mu0 = 5,
  ntrees_max = 100, early_stopping_rounds = 5, verbose = FALSE
)

N_WORKERS = max(1L, future::availableCores() - 1L)
future::plan(future::multisession, workers = N_WORKERS)

cat("============================================================\n")
cat("SMOKING AND LONGEVITY — MULTI-ITERATION ADAPTED VERSION\n")
cat(sprintf("N=%d | Train=%d | Test=%d | Iters=%d | Workers=%d\n",
            N_TOTAL, N_TRAIN, N_TEST, N_ITER, N_WORKERS))
cat(sprintf("Expected treated in training: ~%.0f (at 7.5%% rate)\n",
            N_TRAIN * 0.075))
cat("============================================================\n\n")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

safe_mean = function(x) {
  x = x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

safe_sd = function(x) {
  x = x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  sd(x)
}

safe_cor = function(x, y) {
  ok = is.finite(x) & is.finite(y)
  if (sum(ok) <= 1) return(NA_real_)
  if (sd(x[ok]) < 1e-10 || sd(y[ok]) < 1e-10) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "spearman"))
}

mean_ci = function(x) {
  x = x[is.finite(x)]
  n = length(x)
  if (!n) return(list(mean = NA, sd = NA, ci_lo = NA, ci_hi = NA, n = 0L))
  m = mean(x)
  s = if (n > 1) sd(x) else NA_real_
  se = if (n > 1) s / sqrt(n) else NA_real_
  tc = if (n > 1) qt(0.975, df = n - 1) else NA_real_
  list(
    mean = m,
    sd   = s,
    ci_lo = if (n > 1) m - tc * se else NA_real_,
    ci_hi = if (n > 1) m + tc * se else NA_real_,
    n = n
  )
}

prop_ci = function(x) {
  x = as.numeric(x[!is.na(x)])
  n = length(x)
  if (!n) return(list(mean = NA, ci_lo = NA, ci_hi = NA, n = 0L))
  p = mean(x)
  se = sqrt(p * (1 - p) / n)
  list(
    mean = p,
    ci_lo = max(0, p - 1.96 * se),
    ci_hi = min(1, p + 1.96 * se),
    n = n
  )
}

top_k_metrics = function(est, truth, k_frac = 0.075) {
  ok = is.finite(est) & is.finite(truth)
  e = est[ok]
  t = truth[ok]
  if (length(e) < 2) return(list(overlap = NA_real_, lift = NA_real_))

  k = max(1L, ceiling(k_frac * length(t)))
  pred_top = order(e, decreasing = TRUE)[seq_len(k)]
  true_top = order(t, decreasing = TRUE)[seq_len(k)]

  overlap = length(intersect(pred_top, true_top)) / k
  lift = mean(t[pred_top]) / mean(t)
  list(overlap = overlap, lift = lift)
}

# ============================================================
# SINGLE ITERATION
# Exact DGP and subgroup definitions preserved
# ============================================================

run_one_iteration = function(iter_seed,
                             n_total = N_TOTAL,
                             n_train = N_TRAIN,
                             n_test = N_TEST,
                             sigma = SIGMA_Y,
                             boost_args = BOOST_ARGS,
                             boost_args_tx = BOOST_ARGS_TX,
                             top_k_frac = TOP_K_FRAC) {
  library(MASS)
  library(rlearner)
  library(KRLS2)

  set.seed(iter_seed)

  # ----------------------------------------------------------
  # SECTION 1: COVARIATE GENERATION
  # ----------------------------------------------------------

  cor_matrix = matrix(c(
    1.00, -0.10, -0.35, -0.20,  0.15,
    -0.10,  1.00,  0.15,  0.50, -0.30,
    -0.35,  0.15,  1.00,  0.20, -0.25,
    -0.20,  0.50,  0.20,  1.00, -0.35,
     0.15, -0.30, -0.25, -0.35,  1.00
  ), nrow = 5, byrow = TRUE)

  z = MASS::mvrnorm(n_total, mu = rep(0, 5), Sigma = cor_matrix)

  x1_age      = round(40 + 45 * pnorm(z[, 1]))
  x2_income   = 20  + 130 * pnorm(z[, 2])
  x3_lung     = 40  +  70 * pnorm(z[, 3])
  x4_literacy = round(20  * pnorm(z[, 4]))
  x5_stress   =  10 * pnorm(z[, 5])

  x = cbind(x1_age, x2_income, x3_lung, x4_literacy, x5_stress)
  colnames(x) = c("age", "income", "lung_baseline", "health_literacy", "stress")

  # ----------------------------------------------------------
  # SECTION 2: PROPENSITY AND TREATMENT ASSIGNMENT
  # ----------------------------------------------------------

  x1_s          = as.numeric(scale(x1_age))
  x2_lowinc_s   = as.numeric(scale(-x2_income))
  x3_poorlung_s = as.numeric(scale(-x3_lung))
  x4_lowlit_s   = as.numeric(scale(-x4_literacy))
  x5_s          = as.numeric(scale(x5_stress))

  linpred_no_intercept =
    0.7 * x1_s +
    1.5 * x2_lowinc_s +
    0.9 * x4_lowlit_s +
    1.3 * x5_s

  target_prev  = 0.075
  intercept_fn = function(a) mean(plogis(a + linpred_no_intercept)) - target_prev
  alpha        = uniroot(intercept_fn, interval = c(-10, 0))$root

  propensity = pmax(0.01, pmin(plogis(alpha + linpred_no_intercept), 0.20))

  attempt = 0
  max_attempts = 50
  repeat {
    w = rbinom(n_total, 1, propensity)
    attempt = attempt + 1
    if (mean(w) >= 0.05 && mean(w) <= 0.10) break
    if (attempt >= max_attempts) break
  }

  # ----------------------------------------------------------
  # SECTION 3: TRUE TREATMENT EFFECT tau(X)
  # ----------------------------------------------------------

  tau_x = 3.0 +
    1.1 * x1_s +
    1.4 * x3_poorlung_s +
    0.8 * x5_s +
    0.5 * x4_lowlit_s

  tau_x = pmax(0.5, tau_x)

  # ----------------------------------------------------------
  # SECTION 4: BASELINE OUTCOME b(X)
  # ----------------------------------------------------------

  b_x = 18 +
    3.5 * x1_s +
    3.0 * x2_lowinc_s +
    5.0 * x3_poorlung_s +
    2.0 * x4_lowlit_s +
    4.0 * x5_s +
    1.5 * x1_s * x3_poorlung_s

  # ----------------------------------------------------------
  # SECTION 5: OBSERVED OUTCOME Y
  # ----------------------------------------------------------

  y = b_x + w * tau_x + rnorm(n_total, 0, sigma)

  # ----------------------------------------------------------
  # SECTION 6: TRAIN-TEST SPLIT
  # ----------------------------------------------------------

  train_idx = sample(seq_len(n_total), size = n_train, replace = FALSE)
  test_idx  = setdiff(seq_len(n_total), train_idx)

  x_train  = x[train_idx, ]
  x_test   = x[test_idx, ]
  w_train  = w[train_idx]
  w_test   = w[test_idx]
  y_train  = y[train_idx]
  y_test   = y[test_idx]
  tau_test = tau_x[test_idx]

  # ----------------------------------------------------------
  # SECTION 7: PRE-SPECIFIED SUBGROUPS ON TEST SET
  # ----------------------------------------------------------

  x1_t = x_test[, "age"]
  x2_t = x_test[, "income"]
  x3_t = x_test[, "lung_baseline"]
  x4_t = x_test[, "health_literacy"]
  x5_t = x_test[, "stress"]

  sg1 = x1_t < 60 & x3_t > 75 & x5_t < 4
  sg2 = x1_t > 68 & x3_t < 58 & x5_t > 6
  sg3 = x2_t < 45 & x4_t < 6

  if (sum(sg1) < 5 || sum(sg2) < 5 || sum(sg3) < 5) {
    return(list(
      seed = iter_seed,
      skipped = TRUE,
      reason = paste0("SG too small: sg1=", sum(sg1),
                      " sg2=", sum(sg2),
                      " sg3=", sum(sg3))
    ))
  }

  true_sg1 = mean(tau_test[sg1])
  true_sg2 = mean(tau_test[sg2])
  true_sg3 = mean(tau_test[sg3])

  # ----------------------------------------------------------
  # SECTION 8: FIT ALL LEARNERS
  # ----------------------------------------------------------

  safe_fit = function(fit_fn, pred_fn, ...) {
    tryCatch({
      fit = fit_fn(...)
      pred_fn(fit, x_test)
    }, error = function(e) {
      rep(NA_real_, n_test)
    })
  }

  rlasso_est = safe_fit(rlasso, predict, x_train, w_train, y_train)
  rboost_est = safe_fit(function(...) do.call(rboost, c(list(...), boost_args)),
                        predict, x_train, w_train, y_train)
  rkern_est  = safe_fit(rkern, predict, x_train, w_train, y_train)

  slasso_est = safe_fit(slasso, predict, x_train, w_train, y_train)
  sboost_est = safe_fit(function(...) do.call(sboost, c(list(...), boost_args)),
                        predict, x_train, w_train, y_train)
  skern_est  = safe_fit(skern, predict, x_train, w_train, y_train)

  tlasso_est = safe_fit(tlasso, predict, x_train, w_train, y_train)
  tboost_est = safe_fit(function(...) do.call(tboost, c(list(...), boost_args_tx)),
                        predict, x_train, w_train, y_train)
  tkern_est  = safe_fit(tkern, predict, x_train, w_train, y_train)

  xlasso_est = safe_fit(xlasso, predict, x_train, w_train, y_train)
  xboost_est = safe_fit(function(...) do.call(xboost, c(list(...), boost_args_tx)),
                        predict, x_train, w_train, y_train)
  xkern_est  = safe_fit(xkern, predict, x_train, w_train, y_train)

  x_tr_df = as.data.frame(x_train)
  x_te_df = as.data.frame(x_test)
  colnames(x_tr_df) = colnames(x_te_df) = c("age", "income", "lung", "literacy", "stress")

  ols_fit = tryCatch({
    lm(y_train ~ age + income + lung + literacy + stress + w_train +
         I(w_train * age) + I(w_train * income) + I(w_train * lung) +
         I(w_train * literacy) + I(w_train * stress),
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

  const_ate_train = if (sum(w_train == 1) > 0 && sum(w_train == 0) > 0) {
    mean(y_train[w_train == 1]) - mean(y_train[w_train == 0])
  } else {
    NA_real_
  }

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
    zero_pred = rep(0, n_test),
    const_pred = rep(const_ate_train, n_test),
    ols_inter = as.numeric(ols_tau_est)
  )

  # ----------------------------------------------------------
  # SECTION 9: EVALUATION
  # ----------------------------------------------------------

  tau_variance = var(tau_test)

  eval_one = function(est, name) {
    if (all(is.na(est))) {
      return(data.frame(
        seed = iter_seed,
        learner = name,
        failed = TRUE,
        raw_mse = NA,
        norm_mse = NA,
        rank_corr = NA,
        est_sg1 = NA,
        true_sg1 = round(true_sg1, 4),
        rec_sg1 = NA,
        est_sg2 = NA,
        true_sg2 = round(true_sg2, 4),
        rec_sg2 = NA,
        est_sg3 = NA,
        true_sg3 = round(true_sg3, 4),
        rec_sg3 = NA,
        dev_sg3 = NA,
        correct_order = NA,
        sg3_inflated = NA,
        topk_overlap = NA,
        topk_lift = NA,
        stringsAsFactors = FALSE
      ))
    }

    raw_mse  = mean((est - tau_test)^2, na.rm = TRUE)
    norm_mse = raw_mse / tau_variance
    rc       = safe_cor(est, tau_test)

    e1 = mean(est[sg1], na.rm = TRUE)
    e2 = mean(est[sg2], na.rm = TRUE)
    e3 = mean(est[sg3], na.rm = TRUE)
    tk = top_k_metrics(est, tau_test, top_k_frac)

    data.frame(
      seed = iter_seed,
      learner = name,
      failed = FALSE,
      raw_mse = round(raw_mse, 6),
      norm_mse = round(norm_mse, 6),
      rank_corr = ifelse(is.na(rc), NA, round(rc, 6)),
      est_sg1 = round(e1, 4),
      true_sg1 = round(true_sg1, 4),
      rec_sg1 = round(e1 / true_sg1, 4),
      est_sg2 = round(e2, 4),
      true_sg2 = round(true_sg2, 4),
      rec_sg2 = round(e2 / true_sg2, 4),
      est_sg3 = round(e3, 4),
      true_sg3 = round(true_sg3, 4),
      rec_sg3 = round(e3 / true_sg3, 4),
      dev_sg3 = round(abs(e3 - true_sg3), 4),
      correct_order = (e2 > e3) & (e3 > e1),
      sg3_inflated  = e3 / true_sg3 > 1.1,
      topk_overlap  = round(tk$overlap, 4),
      topk_lift     = round(tk$lift, 4),
      stringsAsFactors = FALSE
    )
  }

  iter_rows = do.call(rbind, mapply(eval_one, learners_all,
                                    names(learners_all), SIMPLIFY = FALSE))

  iter_rows$iteration        = iter_seed - 41
  iter_rows$treatment_rate   = mean(w)
  iter_rows$n_treated_train  = sum(w_train)
  iter_rows$n_treated_test   = sum(w_test)
  iter_rows$sg1_n            = sum(sg1)
  iter_rows$sg2_n            = sum(sg2)
  iter_rows$sg3_n            = sum(sg3)
  iter_rows$true_ate         = round(mean(tau_x), 4)
  iter_rows$tau_sd           = round(sd(tau_x), 4)

  list(seed = iter_seed, skipped = FALSE, rows = iter_rows)
}

# ============================================================
# RUN PARALLEL SIMULATION
# ============================================================

cat("Starting parallel simulation...\n")
start_time = Sys.time()

all_results = furrr::future_map(
  seq_len(N_ITER),
  function(i) run_one_iteration(iter_seed = 41 + i),
  .options = furrr::furrr_options(
    seed = TRUE,
    packages = c("MASS", "rlearner", "KRLS2")
  ),
  .progress = TRUE
)

elapsed_mins = round(as.numeric(Sys.time() - start_time, units = "mins"), 2)
cat("\nDone in", elapsed_mins, "mins\n")

completed = Filter(function(x) !x$skipped, all_results)
skipped   = Filter(function(x)  x$skipped, all_results)

cat("Completed:", length(completed), "| Skipped:", length(skipped), "\n")
if (length(skipped) > 0) {
  for (s in skipped) cat(" seed", s$seed, "—", s$reason, "\n")
}

iter_df = do.call(rbind, lapply(completed, function(x) x$rows))
rownames(iter_df) = NULL

# ============================================================
# SUMMARY ACROSS ITERATIONS
# ============================================================

learner_names = c(
  "rlasso", "rboost", "rkern",
  "slasso", "sboost", "skern",
  "tlasso", "tboost", "tkern",
  "xlasso", "xboost", "xkern",
  "zero_pred", "const_pred", "ols_inter"
)

summarise_learner = function(nm) {
  d = iter_df[iter_df$learner == nm, ]
  raw = mean_ci(d$raw_mse)
  mse = mean_ci(d$norm_mse)
  rc  = mean_ci(d$rank_corr)
  r2  = mean_ci(d$rec_sg2)
  r3  = mean_ci(d$rec_sg3)
  d3  = mean_ci(d$dev_sg3)
  tk  = mean_ci(d$topk_overlap)
  tl  = mean_ci(d$topk_lift)
  ord = prop_ci(d$correct_order)
  inf = prop_ci(d$sg3_inflated)

  data.frame(
    learner = nm,
    n_iter = nrow(d),
    n_success = sum(!d$failed, na.rm = TRUE),
    failure_rate = round(mean(d$failed, na.rm = TRUE), 3),

    mean_raw_mse = raw$mean,
    sd_raw_mse   = raw$sd,
    ci_lo_raw_mse = raw$ci_lo,
    ci_hi_raw_mse = raw$ci_hi,

    mean_norm_mse = mse$mean,
    sd_norm_mse   = mse$sd,
    ci_lo_norm_mse = mse$ci_lo,
    ci_hi_norm_mse = mse$ci_hi,

    mean_rank_corr = rc$mean,
    sd_rank_corr   = rc$sd,
    ci_lo_rank_corr = rc$ci_lo,
    ci_hi_rank_corr = rc$ci_hi,

    mean_rec_sg2 = r2$mean,
    ci_lo_rec_sg2 = r2$ci_lo,
    ci_hi_rec_sg2 = r2$ci_hi,

    mean_rec_sg3 = r3$mean,
    ci_lo_rec_sg3 = r3$ci_lo,
    ci_hi_rec_sg3 = r3$ci_hi,

    mean_dev_sg3 = d3$mean,
    ci_lo_dev_sg3 = d3$ci_lo,
    ci_hi_dev_sg3 = d3$ci_hi,

    mean_topk_overlap = tk$mean,
    ci_lo_topk_overlap = tk$ci_lo,
    ci_hi_topk_overlap = tk$ci_hi,

    mean_topk_lift = tl$mean,
    ci_lo_topk_lift = tl$ci_lo,
    ci_hi_topk_lift = tl$ci_hi,

    subgroup_order_acc = ord$mean,
    ci_lo_order_acc = ord$ci_lo,
    ci_hi_order_acc = ord$ci_hi,

    sg3_inflation_rate = inf$mean,
    ci_lo_sg3_infl = inf$ci_lo,
    ci_hi_sg3_infl = inf$ci_hi,

    stringsAsFactors = FALSE
  )
}

summary_df = do.call(rbind, lapply(learner_names, summarise_learner))
summary_df = summary_df[order(summary_df$mean_norm_mse, na.last = TRUE), ]
rownames(summary_df) = NULL

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================

cat("\n=== MULTI-ITERATION SUMMARY ===\n")
cat(sprintf("%-12s | %-22s | %-22s | %-22s | %-8s | %-8s | %-8s | %-8s | %-5s\n",
            "Learner", "RawMSE [95%CI]", "NormMSE [95%CI]", "RankCorr [95%CI]",
            "Ord%", "SG3Inf%", "TopKOv%", "TopKLift", "Fail%"))
cat(strrep("-", 150), "\n")

for (i in seq_len(nrow(summary_df))) {
  r = summary_df[i, ]

  raw_str = ifelse(is.na(r$mean_raw_mse), "NA",
                   sprintf("%.3f [%.3f,%.3f]",
                           r$mean_raw_mse, r$ci_lo_raw_mse, r$ci_hi_raw_mse))

  mse_str = ifelse(is.na(r$mean_norm_mse), "NA",
                   sprintf("%.3f [%.3f,%.3f]",
                           r$mean_norm_mse, r$ci_lo_norm_mse, r$ci_hi_norm_mse))

  rc_str = ifelse(is.na(r$mean_rank_corr), "NA",
                  sprintf("%.3f [%.3f,%.3f]",
                          r$mean_rank_corr, r$ci_lo_rank_corr, r$ci_hi_rank_corr))

  lift_str = ifelse(is.na(r$mean_topk_lift), "NA",
                    sprintf("%.3f", r$mean_topk_lift))

  cat(sprintf("%-12s | %-22s | %-22s | %-22s | %6.0f%% | %6.0f%% | %6.0f%% | %-8s | %4.0f%%\n",
              r$learner,
              raw_str,
              mse_str,
              rc_str,
              r$subgroup_order_acc * 100,
              r$sg3_inflation_rate * 100,
              r$mean_topk_overlap * 100,
              lift_str,
              r$failure_rate * 100))
}

cat("\nNote: zero_pred  = zero-effect floor benchmark\n")
cat("      const_pred = training-sample constant-effect benchmark\n")
cat("      RawMSE     = mean((tau_hat - tau_true)^2)\n")
cat("      NormMSE    = RawMSE / Var(true tau on test set)\n")
cat("      TopK overlap = share of truly highest-harm patients captured in predicted top group\n")
cat("      TopK lift    = average true tau in predicted top group divided by overall mean tau\n")

# ============================================================
# SAVE RESULTS
# ============================================================

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

iter_csv = file.path(OUTPUT_DIR,
                     sprintf("3_smoking_multi_iter_n%d_%diters.csv", N_TOTAL, N_ITER))
summary_csv = file.path(OUTPUT_DIR,
                        sprintf("3_smoking_multi_iter_summary_n%d_%diters.csv", N_TOTAL, N_ITER))

write.csv(iter_df, iter_csv, row.names = FALSE)
write.csv(summary_df, summary_csv, row.names = FALSE)

cat("\nIteration-level results:", iter_csv, "\n")
cat("Summary results:         ", summary_csv, "\n")
cat("Total time:              ", elapsed_mins, "mins\n")
