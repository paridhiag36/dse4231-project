# =============================================================
# FILE: code/test_lasso_stability.R
# PURPOSE: Verify lasso learner results are stable across multiple runs on all three setups

# Single-run results are noisy — one unlucky seed can make any learner look good or bad. This file averages over 10 reps to get a stable picture of which learner wins on each setup.

# This is the lasso version of the paper's Figure 3.
# We run 10 reps (not 500 like the paper) just to confirm the direction is right before running the full simulation.

# What we expect:
# Setup 1: R-learner wins (confounding present, linear tau)
# Setup 2: all learners similar (no confounding)
# Setup 3: unclear for lasso — tau is nonlinear
# =============================================================

source("code/utils.R")
source("code/1_dgp.R")
source("code/2_dgp.R")
source("code/3_dgp.R")
source("code/learners_lasso.R")

# helper: run n_reps replications for one setup
# returns matrix of MSE values (n_reps x 4 learners)
run_stability_test <- function(dgp_fn, n = 1000, n_reps = 500,
                               setup_name = "") {
  
  results <- matrix(NA, nrow = n_reps, ncol = 4)
  colnames(results) <- c("S", "T", "X", "R")
  
  cat(sprintf("\nRunning %s (%d reps)...\n", setup_name, n_reps))
  
  for (i in seq_len(n_reps)) {
    # fresh dataset each rep — different random seed each time
    dat <- dgp_fn(n = n)
    tr  <- 1:800
    te  <- 801:1000
    
    # run all four learners
    # wrap in tryCatch so one failure doesn't stop everything
    results[i, "S"] <- tryCatch(
      mean((s_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr],
                            dat$X[te,]) - dat$tau[te])^2),
      error = function(e) { cat("  S failed rep", i, "\n"); NA })
    
    results[i, "T"] <- tryCatch(
      mean((t_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr],
                            dat$X[te,]) - dat$tau[te])^2),
      error = function(e) { cat("  T failed rep", i, "\n"); NA })
    
    results[i, "X"] <- tryCatch(
      mean((x_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr],
                            dat$X[te,]) - dat$tau[te])^2),
      error = function(e) { cat("  X failed rep", i, "\n"); NA })
    
    results[i, "R"] <- tryCatch(
      mean((r_learner_lasso(dat$X[tr,], dat$Y[tr], dat$W[tr],
                            dat$X[te,]) - dat$tau[te])^2),
      error = function(e) { cat("  R failed rep", i, "\n"); NA })
    
    cat(sprintf("  rep %2d done | S:%.4f T:%.4f X:%.4f R:%.4f\n",
                i,
                results[i,"S"], results[i,"T"],
                results[i,"X"], results[i,"R"]))
  }
  
  return(results)
}

# print results for one setup cleanly
print_results <- function(results, setup_name) {
  means  <- colMeans(results, na.rm = TRUE)
  sds    <- apply(results, 2, sd, na.rm = TRUE)
  winner <- names(which.min(means))
  
  cat(sprintf("\n--- %s ---\n", setup_name))
  cat(sprintf("  %-10s %-10s %-10s %-10s\n", "S", "T", "X", "R"))
  cat(sprintf("  %-10.4f %-10.4f %-10.4f %-10.4f  (mean MSE)\n",
              means["S"], means["T"], means["X"], means["R"]))
  cat(sprintf("  %-10.4f %-10.4f %-10.4f %-10.4f  (std dev)\n",
              sds["S"], sds["T"], sds["X"], sds["R"]))
  cat(sprintf("  Winner: %s-learner\n", winner))
}


# RUN ALL THREE SETUPS

set.seed(42)

res_1 <- run_stability_test(gen_setup_1,
                            n          = 1000,
                            n_reps     = 500,
                            setup_name = "Setup 1: Hard confounding")

res_2 <- run_stability_test(gen_setup_2,
                            n          = 1000,
                            n_reps     = 500,
                            setup_name = "Setup 2: RCT")

res_3 <- run_stability_test(gen_setup_3,
                            n          = 1000,
                            n_reps     = 500,
                            setup_name = "Setup 3: Sign flip")


# FINAL SUMMARY TABLE

cat("\n")
cat("LASSO STABILITY SUMMARY — mean MSE over 10 reps (n=1000)\n")
cat(sprintf("%-12s %-10s %-10s %-10s %-10s %-10s\n",
            "Setup", "S", "T", "X", "R", "Winner"))
cat(sprintf("%-12s %-10s %-10s %-10s %-10s %-10s\n",
            "------", "----", "----", "----", "----", "------"))

for (setup in list(list(res_1, "Setup 1"),
                   list(res_2, "Setup 2"),
                   list(res_3, "Setup 3"))) {
  means  <- colMeans(setup[[1]], na.rm = TRUE)
  winner <- names(which.min(means))
  cat(sprintf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f %-10s\n",
              setup[[2]],
              means["S"], means["T"], means["X"], means["R"],
              winner))
}

cat("Lower MSE = better\n")
cat("Expected: R wins Setup 1, all similar Setup 2\n")
cat("Setup 3 winner depends on whether lasso can recover sigmoid tau\n")

# save results for later use
# dir.create("results", showWarnings = FALSE)
# saveRDS(list(setup1 = res_1, setup2 = res_2, setup3 = res_3),
        "results/lasso_stability_10reps.rds")
# cat("\nResults saved to results/lasso_stability_10reps.rds\n")