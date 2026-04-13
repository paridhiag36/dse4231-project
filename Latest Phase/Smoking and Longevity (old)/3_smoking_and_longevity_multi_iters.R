# ============================================================
# SMOKING AND LONGEVITY — MULTI-ITERATION PARALLEL SIMULATION

# Runs n_iter independent iterations in parallel using furrr.
# Each iteration: fresh data generation + all learners + evaluation.
# Results aggregated with means and 95% CIs across iterations.

# Core challenge tested: severe treatment imbalance (~7.5% treated).
# With only ~22 treated patients in training per iteration, does the R-learner's propensity residualisation give it a structural advantage over S/T/X learners? 
# Results across 200 iterations give a stable answer that is not dependent on any single random seed.

# INSTALL INSTRUCTIONS (run once in console before sourcing this file):
#   install.packages("future")
#   install.packages("furrr")
#   install.packages("KRLS2")   # may fail on Apple Silicon Macs
#   install.packages("jsonlite")
# ============================================================

library(MASS)
library(rlearner)
library(KRLS2)
library(jsonlite)
library(future)
library(furrr)

# ============================================================
# CONFIGURATION — edit these values to change the simulation
# ============================================================

n_iter  = 200    # number of iterations — reduce to 100 if runtime is too long
n       = 400    # total sample size per iteration
n_train = 300    # training set size
n_test  = 100    # test set size
sigma   = 3      # noise SD for outcome Y

# Boosting settings — reduced ntrees_max because T/X learners
# fit separate models on each arm; treated arm has only ~22 obs
boost_args = list(
  num_search_rounds     = 5,
  k_folds               = 5,
  ntrees_max            = 100,   # reduced from 300 — because of tiny treated arm
  early_stopping_rounds = 5,
  verbose               = FALSE
)

boost_args_tx = list(
  num_search_rounds     = 5,
  k_folds_mu1           = 5,
  k_folds_mu0           = 5,
  ntrees_max            = 100,   # reduced — T/X learner arm models
  early_stopping_rounds = 5,
  verbose               = FALSE
)

# Parallelisation — uses all available cores minus one
# Reduce workers = 2 if memory is a concern
n_workers = max(1, parallel::detectCores() - 1)
plan(multisession, workers = n_workers)
cat("Running", n_iter, "iterations on", n_workers, "workers\n")
cat("Config: n =", n, "| n_train =", n_train, "| n_test =", n_test,
    "| sigma =", sigma, "\n")
cat("Estimated treatment rate: ~7.5% =>",
    round(n_train * 0.075), "treated in training per iteration\n\n")

# ============================================================
# SINGLE ITERATION FUNCTION

# Fully self-contained — no global state dependencies.
# All libraries loaded inside the function for parallel workers.
# Takes a seed, generates data, fits all learners, evaluates, returns a named list of results or a skip record.
# ============================================================

run_one_iteration = function(iter_seed, n, n_train, n_test,
                             sigma, boost_args, boost_args_tx) {
  
  library(MASS)
  library(rlearner)
  library(KRLS2)
  
  set.seed(iter_seed)
  
  # ----------------------------------------------------------
  # DATA GENERATION
  # ----------------------------------------------------------
  
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
  colnames(x) = c("age", "income", "lung_baseline",
                  "health_literacy", "stress")
  
  # Risk-oriented standardised covariates for propensity and tau
  x1_s          = as.numeric(scale(x1_age))
  x2_lowinc_s   = as.numeric(scale(-x2_income))
  x3_poorlung_s = as.numeric(scale(-x3_lung))
  x4_lowlit_s   = as.numeric(scale(-x4_literacy))
  x5_s          = as.numeric(scale(x5_stress))
  
  # Propensity: calibrate intercept to hit 7.5% average treatment rate
  linpred_no_intercept =
    0.7 * x1_s          +
    1.5 * x2_lowinc_s   +
    0.4 * x3_poorlung_s +
    0.9 * x4_lowlit_s   +
    1.3 * x5_s
  
  target_prev  = 0.075
  intercept_fn = function(a)
    mean(plogis(a + linpred_no_intercept)) - target_prev
  alpha      = uniroot(intercept_fn, interval = c(-10, 0))$root
  propensity = pmax(0.01, pmin(plogis(alpha + linpred_no_intercept), 0.20))
  
  # Resample until treatment share is 5-10%
  max_attempts = 50
  attempt      = 0
  repeat {
    w = rbinom(n, 1, propensity)
    attempt = attempt + 1
    if (mean(w) >= 0.05 && mean(w) <= 0.10) break
    if (attempt >= max_attempts) break   # safety exit after 50 tries
  }
  
  # True treatment effect — always positive (harmful), heterogeneous in magnitude
  tau_x = pmax(0.5,
               2.0 +
                 1.1 * x1_s          +
                 1.4 * x3_poorlung_s +
                 0.8 * x5_s          +
                 0.5 * x4_lowlit_s
  )
  
  # Baseline outcome (complex nuisance with interaction)
  b_x = 18 +
    3.5 * x1_s          +
    3.0 * x2_lowinc_s   +
    5.0 * x3_poorlung_s +
    2.0 * x4_lowlit_s   +
    4.0 * x5_s          +
    1.5 * x1_s * x3_poorlung_s
  
  # Observed outcome — correct DGP: W * tau(X), not (W - propensity) * tau(X)
  y = b_x + w * tau_x + rnorm(n, 0, sigma)
  
  # ----------------------------------------------------------
  # TRAIN-TEST SPLIT
  # ----------------------------------------------------------
  
  train_idx = sample(1:n, size = n_train, replace = FALSE)
  test_idx  = setdiff(1:n, train_idx)
  
  x_train  = x[train_idx, ];  x_test  = x[test_idx, ]
  w_train  = w[train_idx];    w_test  = w[test_idx]
  y_train  = y[train_idx];    y_test  = y[test_idx]
  tau_test = tau_x[test_idx]
  
  # ----------------------------------------------------------
  # PRE-SPECIFIED SUBGROUPS ON TEST SET
  
  # Thresholds relaxedfrom smoking_and_longevity_all.R to ensure subgroups are populated across all random seeds with n_test = 100.
  
  # SG1 (lowest harm): relaxed from age<55 to age<58, lung>75 to lung>70, stress<4 to stress<5, literacy>14 to literacy>12
  # SG2 (highest harm): relaxed from age>70 to age>65, lung<55 to lung<60, stress>7 to stress>6
  # SG3 (confounding trap): relaxed from income<40 to income<50, literacy<5 to literacy<7
  
  # Clinical thresholds still meaningful after relaxation — just slightly broader definitions of each patient profile.
  # ----------------------------------------------------------
  
  x1_t = x_test[, "age"]
  x2_t = x_test[, "income"]
  x3_t = x_test[, "lung_baseline"]
  x4_t = x_test[, "health_literacy"]
  x5_t = x_test[, "stress"]
  
  sg1 = x1_t <  58 &   # relaxed from 55
    x3_t >  70 &   # relaxed from 75
    x5_t <   5 &   # relaxed from 4
    x4_t >  12     # relaxed from 14
  
  sg2 = x1_t >  65 &   # relaxed from 70
    x3_t <  60 &   # relaxed from 55
    x5_t >   6     # relaxed from 7
  
  sg3 = x2_t <  50 &   # relaxed from 40
    x4_t <   7     # relaxed from 5
  
  # Skip iteration if any subgroup has fewer than 5 patients
  # This avoids unreliable subgroup estimates polluting the averages
  if (sum(sg1) < 5 | sum(sg2) < 5 | sum(sg3) < 5) {
    return(list(
      seed    = iter_seed,
      skipped = TRUE,
      reason  = paste0("Subgroup too small: SG1=", sum(sg1),
                       " SG2=", sum(sg2), " SG3=", sum(sg3))
    ))
  }
  
  true_sg1 = mean(tau_test[sg1])
  true_sg2 = mean(tau_test[sg2])
  true_sg3 = mean(tau_test[sg3])
  
  # ----------------------------------------------------------
  # FIT ALL LEARNERS
  # Wrapped in tryCatch — a failed learner returns NA vector rather than crashing the entire iteration.
  # tboost and xboost are most likely to fail (~22 treated in training).
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
  rboost_est = safe_fit(
    function(...) do.call(rboost, c(list(...), boost_args)),
    predict, x_train, w_train, y_train)
  rkern_est  = safe_fit(rkern,  predict, x_train, w_train, y_train)
  
  slasso_est = safe_fit(slasso, predict, x_train, w_train, y_train)
  sboost_est = safe_fit(
    function(...) do.call(sboost, c(list(...), boost_args)),
    predict, x_train, w_train, y_train)
  skern_est  = safe_fit(skern,  predict, x_train, w_train, y_train)
  
  tlasso_est = safe_fit(tlasso, predict, x_train, w_train, y_train)
  tboost_est = safe_fit(
    function(...) do.call(tboost, c(list(...), boost_args_tx)),
    predict, x_train, w_train, y_train)
  tkern_est  = safe_fit(tkern,  predict, x_train, w_train, y_train)
  
  xlasso_est = safe_fit(xlasso, predict, x_train, w_train, y_train)
  xboost_est = safe_fit(
    function(...) do.call(xboost, c(list(...), boost_args_tx)),
    predict, x_train, w_train, y_train)
  xkern_est  = safe_fit(xkern,  predict, x_train, w_train, y_train)
  
  # OLS with W x covariate interactions as linear baseline
  x_tr_df = as.data.frame(x_train)
  x_te_df = as.data.frame(x_test)
  colnames(x_tr_df) = colnames(x_te_df) =
    c("age", "income", "lung", "literacy", "stress")
  
  ols_fit = tryCatch({
    lm(y_train ~ age + income + lung + literacy + stress +
         w_train +
         I(w_train * age)      + I(w_train * income) +
         I(w_train * lung)     + I(w_train * literacy) +
         I(w_train * stress),
       data = cbind(x_tr_df, y_train, w_train))
  }, error = function(e) NULL)
  
  if (!is.null(ols_fit)) {
    cf          = coef(ols_fit)
    ols_tau_est = cf["w_train"] +
      cf["I(w_train * age)"]      * x_te_df$age      +
      cf["I(w_train * income)"]   * x_te_df$income   +
      cf["I(w_train * lung)"]     * x_te_df$lung      +
      cf["I(w_train * literacy)"] * x_te_df$literacy +
      cf["I(w_train * stress)"]   * x_te_df$stress
  } else {
    ols_tau_est = rep(NA_real_, n_test)
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
    ols_inter = as.numeric(ols_tau_est)
  )
  
  # ----------------------------------------------------------
  # EVALUATION
  # ----------------------------------------------------------
  
  tau_variance = var(tau_test)
  
  eval_one = function(est, name) {
    
    if (all(is.na(est))) {
      return(list(
        learner       = name, failed = TRUE,
        norm_mse      = NA, rank_corr = NA, mean_tau = NA,
        est_sg1       = NA, rec_sg1 = NA,
        est_sg2       = NA, rec_sg2 = NA,
        est_sg3       = NA, rec_sg3 = NA, dev_sg3 = NA,
        correct_order = NA, sg3_inflated = NA
      ))
    }
    
    raw_mse  = mean((est - tau_test)^2, na.rm = TRUE)
    norm_mse = raw_mse / tau_variance
    rank_corr= ifelse(sd(est, na.rm=TRUE) < 1e-10, NA,
                      cor(est, tau_test, method = "spearman",
                          use = "complete.obs"))
    
    est_sg1 = mean(est[sg1], na.rm = TRUE)
    est_sg2 = mean(est[sg2], na.rm = TRUE)
    est_sg3 = mean(est[sg3], na.rm = TRUE)
    
    rec_sg1 = est_sg1 / true_sg1
    rec_sg2 = est_sg2 / true_sg2
    rec_sg3 = est_sg3 / true_sg3
    dev_sg3 = abs(est_sg3 - true_sg3)
    
    # Correct severity ordering: does learner rank SG2 > SG3 > SG1?
    correct_order = (!is.na(est_sg2) & !is.na(est_sg3) & !is.na(est_sg1)) &&
      (est_sg2 > est_sg3) && (est_sg3 > est_sg1)
    
    # SG3 inflation: recovery > 1.1 signals confounding contamination
    sg3_inflated = (!is.na(rec_sg3)) && (rec_sg3 > 1.1)
    
    list(
      learner       = name,
      failed        = FALSE,
      norm_mse      = round(norm_mse,  6),
      rank_corr     = ifelse(is.na(rank_corr), NA, round(rank_corr, 6)),
      mean_tau      = round(mean(est, na.rm=TRUE), 6),
      est_sg1       = round(est_sg1, 6),
      rec_sg1       = round(rec_sg1, 6),
      est_sg2       = round(est_sg2, 6),
      rec_sg2       = round(rec_sg2, 6),
      est_sg3       = round(est_sg3, 6),
      rec_sg3       = round(rec_sg3, 6),
      dev_sg3       = round(dev_sg3, 6),
      correct_order = correct_order,
      sg3_inflated  = sg3_inflated
    )
  }
  
  eval_results = mapply(eval_one, learners_all, names(learners_all),
                        SIMPLIFY = FALSE)
  
  # ----------------------------------------------------------
  # RETURN ITERATION RESULTS
  # ----------------------------------------------------------
  
  list(
    seed           = iter_seed,
    skipped        = FALSE,
    true_ate       = round(mean(tau_x),    6),
    true_ate_test  = round(mean(tau_test), 6),
    tau_sd         = round(sd(tau_test),   6),
    n_treated_train= sum(w_train),
    n_treated_test = sum(w_test),
    treatment_rate = round(mean(w),        4),
    sg_sizes       = list(sg1=sum(sg1), sg2=sum(sg2), sg3=sum(sg3)),
    true_sg1       = round(true_sg1, 6),
    true_sg2       = round(true_sg2, 6),
    true_sg3       = round(true_sg3, 6),
    results        = eval_results
  )
}

# ============================================================
# RUN ITERATIONS IN PARALLEL
# ============================================================

cat("Starting parallel simulation...\n")
cat("To stop early: press Escape or Ctrl+C\n\n")
sim_start = Sys.time()

all_iterations = future_map(
  1:n_iter,
  function(i) run_one_iteration(
    iter_seed    = i,
    n            = n,
    n_train      = n_train,
    n_test       = n_test,
    sigma        = sigma,
    boost_args   = boost_args,
    boost_args_tx= boost_args_tx
  ),
  .options = furrr_options(seed = TRUE),
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
if (n_skipped > 0) {
  cat("Skip reasons:\n")
  for (s in skipped) cat(" seed", s$seed, "—", s$reason, "\n")
}

learner_names = c("rlasso", "rboost", "rkern",
                  "slasso", "sboost", "skern",
                  "tlasso", "tboost", "tkern",
                  "xlasso", "xboost", "xkern",
                  "zero_pred", "ols_inter")

# Summarise each learner across completed iterations
# ci() computes mean, SD, and 95% CI for any numeric vector
summarise_learner = function(lname) {
  
  ci = function(x) {
    x = as.numeric(x[!is.na(x)])
    if (length(x) < 2) return(list(mean=NA, sd=NA, ci_lo=NA, ci_hi=NA))
    m  = mean(x);  s = sd(x);  se = s / sqrt(length(x))
    list(mean  = round(m,            4),
         sd    = round(s,            4),
         ci_lo = round(m - 1.96*se,  4),
         ci_hi = round(m + 1.96*se,  4))
  }
  
  extract = function(field)
    sapply(completed, function(it) it$results[[lname]][[field]])
  
  failed_vec        = extract("failed")
  norm_mse_vec      = extract("norm_mse")
  rank_corr_vec     = extract("rank_corr")
  rec_sg1_vec       = extract("rec_sg1")
  rec_sg2_vec       = extract("rec_sg2")
  rec_sg3_vec       = extract("rec_sg3")
  dev_sg3_vec       = extract("dev_sg3")
  correct_order_vec = extract("correct_order")
  sg3_inflated_vec  = extract("sg3_inflated")
  
  list(
    learner            = lname,
    n_valid            = sum(!failed_vec, na.rm = TRUE),
    n_failed           = sum( failed_vec, na.rm = TRUE),
    norm_mse           = ci(norm_mse_vec),
    rank_corr          = ci(rank_corr_vec),
    rec_sg1            = ci(rec_sg1_vec),
    rec_sg2            = ci(rec_sg2_vec),
    rec_sg3            = ci(rec_sg3_vec),
    dev_sg3            = ci(dev_sg3_vec),
    # Rate metrics — proportion of iterations where condition holds
    correct_order_rate = round(mean(correct_order_vec, na.rm=TRUE), 4),
    sg3_inflated_rate  = round(mean(sg3_inflated_vec,  na.rm=TRUE), 4)
  )
}

summary_by_learner = lapply(learner_names, summarise_learner)
names(summary_by_learner) = learner_names

# True ATE distribution across iterations
true_ate_vec = sapply(completed, function(it) it$true_ate)
true_ate_summary = list(
  mean  = round(mean(true_ate_vec), 4),
  sd    = round(sd(true_ate_vec),   4),
  ci_lo = round(mean(true_ate_vec) - 1.96*sd(true_ate_vec)/sqrt(n_complete), 4),
  ci_hi = round(mean(true_ate_vec) + 1.96*sd(true_ate_vec)/sqrt(n_complete), 4),
  min   = round(min(true_ate_vec),  4),
  max   = round(max(true_ate_vec),  4)
)

# Treatment rate distribution across iterations
treat_rate_vec = sapply(completed, function(it) it$treatment_rate)

# ============================================================
# PRINT SUMMARY TABLE TO CONSOLE
# ============================================================

cat("\n===========================================\n")
cat("MULTI-ITERATION SUMMARY\n")
cat("Study: Smoking and Longevity — Rare Treatment Imbalance\n")
cat(n_complete, "completed iterations |", n_skipped, "skipped\n")
cat("===========================================\n\n")

cat(sprintf("True ATE:  mean=%.3f  SD=%.3f  95%%CI [%.3f, %.3f]\n",
            true_ate_summary$mean, true_ate_summary$sd,
            true_ate_summary$ci_lo, true_ate_summary$ci_hi))
cat(sprintf("Treat rate: mean=%.3f  SD=%.3f\n\n",
            mean(treat_rate_vec), sd(treat_rate_vec)))

# Main comparison table
cat(sprintf("%-12s | %-18s | %-18s | %-9s | %-12s | %-12s\n",
            "Learner", "NormMSE mean[CI]", "RankCorr mean[CI]",
            "Ord%", "RecSG3 mean", "SG3Infl%"))
cat(strrep("-", 95), "\n")

for (nm in learner_names) {
  s = summary_by_learner[[nm]]
  
  mse_str  = ifelse(is.na(s$norm_mse$mean), "  NA [  NA,  NA]",
                    sprintf("%.3f [%.3f,%.3f]",
                            s$norm_mse$mean,
                            s$norm_mse$ci_lo,
                            s$norm_mse$ci_hi))
  
  corr_str = ifelse(is.na(s$rank_corr$mean), "  NA [  NA,  NA]",
                    sprintf("%.3f [%.3f,%.3f]",
                            s$rank_corr$mean,
                            s$rank_corr$ci_lo,
                            s$rank_corr$ci_hi))
  
  rec_sg3_str = ifelse(is.na(s$rec_sg3$mean), "  NA",
                       sprintf("%.3f", s$rec_sg3$mean))
  
  cat(sprintf("%-12s | %-18s | %-18s | %7.0f%% | %-12s | %9.0f%%\n",
              nm, mse_str, corr_str,
              s$correct_order_rate * 100,
              rec_sg3_str,
              s$sg3_inflated_rate  * 100))
}

cat("\nColumn guide:\n")
cat("  NormMSE      mean normalised MSE across iterations (< 1.0 = beats zero predictor)\n")
cat("  RankCorr     mean Spearman rank correlation with true tau\n")
cat("  Ord%         % of iterations where learner correctly orders SG2 > SG3 > SG1\n")
cat("  RecSG3 mean  mean recovery ratio for SG3 (> 1.1 = confounding inflation)\n")
cat("  SG3Infl%     % of iterations where SG3 recovery > 1.1 (confounding detected)\n")

# ============================================================
# EXPORT TO JSON
# ============================================================

cat("\n=== EXPORTING TO JSON ===\n")

json_output = list(
  
  simulation_config = list(
    study            = "Smoking and Longevity — Rare Treatment Imbalance",
    n_iterations     = n_iter,
    n_completed      = n_complete,
    n_skipped        = n_skipped,
    n_total          = n,
    n_train          = n_train,
    n_test           = n_test,
    sigma            = sigma,
    target_treat_rate= 0.075,
    total_time_mins  = total_time
  ),
  
  true_ate_distribution = true_ate_summary,
  
  summary_by_learner = summary_by_learner,
  
  # Full per-iteration results — for downstream plots or re-analysis
  iterations = all_iterations
)

json_path = "smoking_simulation.json"
write(toJSON(json_output, auto_unbox = TRUE), json_path)
cat("Results written to:", json_path, "\n")
cat("File size:", round(file.size(json_path) / 1024^2, 2), "MB\n")