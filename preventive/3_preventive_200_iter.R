# ============================================================
# SMS CALIBRATION STUDY — MULTI-ITERATION PARALLEL SIMULATION
# tau*(X) = 0.01 for everyone — tests whether learners
# hallucinate heterogeneity when none exists
# Structure mirrors telemedicine multi-iteration study
# ============================================================

library(MASS)
library(rlearner)
library(jsonlite)
library(future)
library(furrr)

# ============================================================
# CONFIGURATION
# ============================================================

n_iter   = 200     # number of independent iterations
n        = 300    # total sample size per iteration
n_train  = 200    # training set size
n_test   = 100     # test set size
sigma    = 1.0    # noise SD for outcome Y

# Calibration thresholds — same as single iteration study
pass_threshold    = 0.20   # IQR threshold for calibration pass
false_pos_threshold = 0.10  # subgroup false positive threshold

# Boosting settings
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

# Parallelisation
n_workers = max(1, parallel::detectCores() - 1)
plan(multisession, workers = n_workers)
cat("Running", n_iter, "iterations on", n_workers, "workers\n\n")

# ============================================================
# SINGLE ITERATION FUNCTION
# Self-contained — no global state dependencies
# Takes a seed, generates SMS data, fits all learners,
# evaluates calibration, returns named list
# ============================================================

run_one_iteration = function(iter_seed, n, n_train, n_test,
                             sigma, boost_args, boost_args_others,
                             pass_threshold, false_pos_threshold) {
  
  library(MASS)
  library(rlearner)
  
  set.seed(iter_seed)
  
  # ----------------------------------------------------------
  # DATA GENERATION — same DGP as single iteration study
  # ----------------------------------------------------------
  
  cor_matrix = matrix(c(
    1.00,  0.35,  0.25, -0.05, -0.15,
    0.35,  1.00,  0.40, -0.05, -0.20,
    0.25,  0.40,  1.00, -0.05, -0.30,
    -0.05, -0.05, -0.05,  1.00,  0.05,
    -0.15, -0.20, -0.30,  0.05,  1.00
  ), nrow = 5, byrow = TRUE)
  
  z = MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)
  
  x1_age         = round(30 + 45 * pnorm(z[,1]))
  x2_visits      = round(12 * pnorm(z[,2]))
  x3_adherence   = pnorm(z[,3])
  x4_travel      = 5 + 55 * pnorm(z[,4])
  x5_deprivation = as.numeric(scale(z[,5]))
  
  x = cbind(x1_age, x2_visits, x3_adherence,
            x4_travel, x5_deprivation)
  colnames(x) = c("age", "past_visits", "med_adherence",
                  "travel_time", "deprivation")
  
  # Propensity
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
  
  # TRUE treatment effect — flat 0.01 for everyone
  tau_x    = rep(0.01, n)
  true_tau = 0.01
  
  # Baseline outcome
  b_x = 3.0 +
    0.15 * as.numeric(x2_s) +
    0.10 * as.numeric(x3_s) +
    -0.08 * as.numeric(x4_s) +
    -0.08 * as.numeric(x5_s) +
    0.05 * as.numeric(x1_s)
  
  # Observed outcome
  y_raw = b_x + (w - propensity) * tau_x + rnorm(n, 0, sigma)
  y     = pmax(y_raw, 0)
  
  # ----------------------------------------------------------
  # TRAIN-TEST SPLIT
  # ----------------------------------------------------------
  
  train_idx = sample(1:n, size = n_train, replace = FALSE)
  test_idx  = setdiff(1:n, train_idx)
  
  x_train   = x[train_idx, ];   x_test  = x[test_idx, ]
  w_train   = w[train_idx];     w_test  = w[test_idx]
  y_train   = y[train_idx];     y_test  = y[test_idx]
  tau_test  = tau_x[test_idx]   # always 0.01
  
  # ----------------------------------------------------------
  # SUBGROUPS — defined on test set
  # Same definitions as single iteration study
  # True tau = 0.01 in ALL subgroups by design
  # ----------------------------------------------------------
  
  x1_t = x_test[, "age"]
  x2_t = x_test[, "past_visits"]
  x3_t = x_test[, "med_adherence"]
  x4_t = x_test[, "travel_time"]
  
  sg1 = x2_t > median(x2_visits) & x3_t > median(x3_adherence)
  sg2 = x4_t > median(x4_travel) & x2_t < median(x2_visits)
  sg3 = x1_t > quantile(x1_age, 0.75) & x3_t > median(x3_adherence)
  
  # Skip iteration if any subgroup is too small
  if (sum(sg1) < 3 | sum(sg2) < 3 | sum(sg3) < 3) {
    return(list(
      seed    = iter_seed,
      skipped = TRUE,
      reason  = paste0("Subgroup too small: SG1=", sum(sg1),
                       " SG2=", sum(sg2), " SG3=", sum(sg3))
    ))
  }
  
  # ----------------------------------------------------------
  # FIT ALL LEARNERS
  # ----------------------------------------------------------
  
  safe_fit = function(fit_fn, pred_fn, ...) {
    tryCatch({
      fit = fit_fn(...)
      pred_fn(fit, x_test)
    }, error = function(e) {
      rep(NA_real_, nrow(x_test))
    })
  }
  
  # R-learners
  rlasso_est = safe_fit(rlasso, predict, x_train, w_train, y_train)
  rboost_est = safe_fit(
    function(...) do.call(rboost, c(list(...), boost_args)),
    predict, x_train, w_train, y_train)
  rkern_est  = safe_fit(rkern,  predict, x_train, w_train, y_train)
  
  # S-learners
  slasso_est = safe_fit(slasso, predict, x_train, w_train, y_train)
  sboost_est = safe_fit(
    function(...) do.call(sboost, c(list(...), boost_args)),
    predict, x_train, w_train, y_train)
  skern_est  = safe_fit(skern,  predict, x_train, w_train, y_train)
  
  # T-learners
  tlasso_est = safe_fit(tlasso, predict, x_train, w_train, y_train)
  tboost_est = safe_fit(
    function(...) do.call(tboost, c(list(...), boost_args_others)),
    predict, x_train, w_train, y_train)
  tkern_est  = safe_fit(tkern,  predict, x_train, w_train, y_train)
  
  # X-learners
  xlasso_est = safe_fit(xlasso, predict, x_train, w_train, y_train)
  xboost_est = safe_fit(
    function(...) do.call(xboost, c(list(...), boost_args_others)),
    predict, x_train, w_train, y_train)
  xkern_est  = safe_fit(xkern,  predict, x_train, w_train, y_train)
  
  # Baselines
  const_pred = rep(0.01, nrow(x_test))
  zero_pred  = rep(0.00, nrow(x_test))
  
  x_tr_df = as.data.frame(x_train)
  x_te_df = as.data.frame(x_test)
  colnames(x_tr_df) = colnames(x_te_df) = c("age","visits",
                                            "adherence","travel","deprivation")
  
  ols_fit = tryCatch({
    lm(y_train ~ age + visits + adherence + travel + deprivation +
         w_train +
         I(w_train * age)         +
         I(w_train * visits)      +
         I(w_train * adherence)   +
         I(w_train * travel)      +
         I(w_train * deprivation),
       data = cbind(x_tr_df, y_train, w_train))
  }, error = function(e) NULL)
  
  if (!is.null(ols_fit)) {
    cf = coef(ols_fit)
    ols_tau_est = cf["w_train"] +
      cf["I(w_train * age)"]         * x_te_df$age +
      cf["I(w_train * visits)"]      * x_te_df$visits +
      cf["I(w_train * adherence)"]   * x_te_df$adherence +
      cf["I(w_train * travel)"]      * x_te_df$travel +
      cf["I(w_train * deprivation)"] * x_te_df$deprivation
  } else {
    ols_tau_est = rep(NA_real_, nrow(x_test))
  }
  
  # ----------------------------------------------------------
  # COLLECT ALL LEARNERS
  # ----------------------------------------------------------
  
  learners_all = list(
    rlasso     = rlasso_est,
    rboost     = rboost_est,
    rkern      = rkern_est,
    slasso     = slasso_est,
    sboost     = sboost_est,
    skern      = skern_est,
    tlasso     = tlasso_est,
    tboost     = tboost_est,
    tkern      = tkern_est,
    xlasso     = xlasso_est,
    xboost     = xboost_est,
    xkern      = xkern_est,
    const_pred = const_pred,
    zero_pred  = zero_pred,
    ols_inter  = as.numeric(ols_tau_est)
  )
  
  # ----------------------------------------------------------
  # EVALUATION
  # Primary metrics are calibration-specific:
  #   IQR of tau_hat — main spread metric
  #   bias — is the mean right
  #   FP_SG — does learner wrongly conclude subgroups differ
  # ----------------------------------------------------------
  
  eval_one = function(est, name) {
    
    if (all(is.na(est))) {
      return(list(
        learner  = name,
        failed   = TRUE,
        bias     = NA, iqr = NA, spread_sd = NA,
        range_90 = NA, pass_cal = NA,
        est_sg1  = NA, fp_sg1 = NA,
        est_sg2  = NA, fp_sg2 = NA,
        est_sg3  = NA, fp_sg3 = NA,
        mse      = NA
      ))
    }
    
    bias     = mean(est, na.rm = TRUE) - true_tau
    iqr_est  = IQR(est, na.rm = TRUE)
    spread_sd= sd(est,  na.rm = TRUE)
    range_90 = as.numeric(
      quantile(est, 0.95, na.rm = TRUE) -
        quantile(est, 0.05, na.rm = TRUE))
    mse      = mean((est - tau_test)^2, na.rm = TRUE)
    pass_cal = iqr_est < pass_threshold
    
    est_sg1 = mean(est[sg1], na.rm = TRUE)
    est_sg2 = mean(est[sg2], na.rm = TRUE)
    est_sg3 = mean(est[sg3], na.rm = TRUE)
    
    fp_sg1  = abs(est_sg1 - true_tau) > false_pos_threshold
    fp_sg2  = abs(est_sg2 - true_tau) > false_pos_threshold
    fp_sg3  = abs(est_sg3 - true_tau) > false_pos_threshold
    
    list(
      learner   = name,
      failed    = FALSE,
      bias      = round(bias,     4),
      iqr       = round(iqr_est,  4),
      spread_sd = round(spread_sd,4),
      range_90  = round(range_90, 4),
      mse       = round(mse,      6),
      pass_cal  = pass_cal,
      est_sg1   = round(est_sg1,  4),
      fp_sg1    = fp_sg1,
      est_sg2   = round(est_sg2,  4),
      fp_sg2    = fp_sg2,
      est_sg3   = round(est_sg3,  4),
      fp_sg3    = fp_sg3
    )
  }
  
  eval_results = mapply(eval_one,
                        learners_all,
                        names(learners_all),
                        SIMPLIFY = FALSE)
  
  # ----------------------------------------------------------
  # RETURN
  # ----------------------------------------------------------
  
  list(
    seed        = iter_seed,
    skipped     = FALSE,
    true_tau    = true_tau,
    prop_treated= round(mean(w_train), 3),
    prop_floored= round(mean(y_raw[train_idx] < 0), 3),
    sg_sizes    = list(sg1 = sum(sg1),
                       sg2 = sum(sg2),
                       sg3 = sum(sg3)),
    results     = eval_results
  )
}

# ============================================================
# RUN ITERATIONS IN PARALLEL
# ============================================================

cat("Starting parallel SMS calibration simulation...\n")
sim_start = Sys.time()

all_iterations = future_map(
  1:n_iter,
  function(i) run_one_iteration(
    iter_seed         = i,
    n                 = n,
    n_train           = n_train,
    n_test            = n_test,
    sigma             = sigma,
    boost_args        = boost_args,
    boost_args_others = boost_args_others,
    pass_threshold    = pass_threshold,
    false_pos_threshold = false_pos_threshold
  ),
  .options  = furrr_options(seed = TRUE),
  .progress = TRUE
)

sim_end    = Sys.time()
total_time = round(as.numeric(sim_end - sim_start, units = "mins"), 2)
cat("\nAll iterations complete in", total_time, "minutes\n")

# ============================================================
# SUMMARISE ACROSS ITERATIONS
# ============================================================

completed  = Filter(function(x) !x$skipped, all_iterations)
skipped    = Filter(function(x)  x$skipped, all_iterations)
n_complete = length(completed)
n_skipped  = length(skipped)

cat("Completed:", n_complete, "| Skipped:", n_skipped, "\n")

learner_names = c("rlasso","rboost","rkern",
                  "slasso","sboost","skern",
                  "tlasso","tboost","tkern",
                  "xlasso","xboost","xkern",
                  "const_pred","zero_pred","ols_inter")

# CI helper
ci = function(x) {
  x  = x[!is.na(x)]
  if (length(x) < 2) return(list(mean=NA,sd=NA,ci_lo=NA,ci_hi=NA))
  m  = mean(x)
  s  = sd(x)
  se = s / sqrt(length(x))
  list(mean  = round(m, 4),
       sd    = round(s, 4),
       ci_lo = round(m - 1.96*se, 4),
       ci_hi = round(m + 1.96*se, 4))
}

summarise_learner = function(lname) {
  
  get = function(metric) sapply(completed, function(it)
    it$results[[lname]][[metric]])
  
  failed_vec   = get("failed")
  n_valid      = sum(!failed_vec, na.rm = TRUE)
  
  list(
    learner      = lname,
    n_valid      = n_valid,
    n_failed     = sum(failed_vec, na.rm = TRUE),
    # PRIMARY calibration metrics
    iqr          = ci(get("iqr")),
    spread_sd    = ci(get("spread_sd")),
    range_90     = ci(get("range_90")),
    bias         = ci(get("bias")),
    mse          = ci(get("mse")),
    # Calibration pass rate across iterations
    pass_cal_rate= round(mean(get("pass_cal"), na.rm = TRUE), 4),
    # False positive rates per subgroup
    fp_sg1_rate  = round(mean(get("fp_sg1"),   na.rm = TRUE), 4),
    fp_sg2_rate  = round(mean(get("fp_sg2"),   na.rm = TRUE), 4),
    fp_sg3_rate  = round(mean(get("fp_sg3"),   na.rm = TRUE), 4)
  )
}

summary_by_learner = lapply(learner_names, summarise_learner)
names(summary_by_learner) = learner_names

# ============================================================
# PRINT SUMMARY TABLES
# ============================================================

cat("\n===========================================\n")
cat("SMS CALIBRATION — MULTI-ITERATION SUMMARY\n")
cat("n_iter =", n_complete, "completed | tau*(X) = 0.01 for all\n")
cat("===========================================\n\n")

print_table = function(learner_subset, title) {
  cat("---", title, "---\n")
  cat(sprintf("%-12s | %s | %s | %s | %s | %s\n",
              "Learner", "IQR mean[CI]", "Bias mean",
              "PassCal%", "FP_SG1%", "FP_SG2%"))
  cat(strrep("-", 80), "\n")
  for (nm in learner_subset) {
    s = summary_by_learner[[nm]]
    cat(sprintf("%-12s | %.3f [%.3f,%.3f] | %+.4f | %.0f%% | %.0f%% | %.0f%%\n",
                nm,
                s$iqr$mean, s$iqr$ci_lo, s$iqr$ci_hi,
                s$bias$mean,
                s$pass_cal_rate * 100,
                s$fp_sg1_rate   * 100,
                s$fp_sg2_rate   * 100))
  }
  cat("\n")
}

# Comparison 1: by meta-learner type
cat("COMPARISON 1: BY META-LEARNER TYPE\n\n")
for (base in c("lasso","boost","kern")) {
  print_table(
    paste0(c("r","s","t","x"), base),
    paste("Base method:", toupper(base), "— R vs S vs T vs X")
  )
}

# Comparison 2: by base method
cat("COMPARISON 2: BY BASE METHOD\n\n")
for (ml in c("r","s","t","x")) {
  ml_name = switch(ml, r="R-LEARNER", s="S-LEARNER",
                   t="T-LEARNER", x="X-LEARNER")
  print_table(
    paste0(ml, c("lasso","boost","kern")),
    paste(ml_name, "— lasso vs boost vs kern")
  )
}

# Comparison 3: baselines
print_table(
  c("const_pred","zero_pred","ols_inter"),
  "BASELINES"
)

# ============================================================
# EXPORT TO JSON
# ============================================================

cat("=== EXPORTING TO JSON ===\n")

json_output = list(
  
  simulation_config = list(
    study             = "SMS Calibration — Null Effect tau*(X)=0.01",
    n_iterations      = n_iter,
    n_completed       = n_complete,
    n_skipped         = n_skipped,
    n_total           = n,
    n_train           = n_train,
    n_test            = n_test,
    sigma             = sigma,
    true_tau          = 0.01,
    pass_threshold    = pass_threshold,
    fp_threshold      = false_pos_threshold,
    total_time_mins   = total_time
  ),
  
  summary_by_learner = summary_by_learner,
  
  # Full per-iteration results for downstream plotting
  iterations = all_iterations
)

json_path = "preventive_results_200.json"
write(toJSON(json_output, auto_unbox = TRUE), json_path)
cat("Results written to:", json_path, "\n")
cat("File size:", round(file.size(json_path) / 1024^2, 2), "MB\n")