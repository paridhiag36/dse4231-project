# =============================================================
# FILE: code/test_lasso_all_setups.R
# PURPOSE: Run all 4 lasso learners on all 3 setups

# This is a single test — not the full simulation yet, a sanity check before we run 50 reps.

# What we expect to see:

# Setup 1 (hard confounding, simple tau):
#   R-learner should win clearly
#   S-learner likely worst — can't separate confounding

# Setup 2 (RCT, no confounding):
#   All learners should be similar
#   Nobody has a big advantage — no confounding to overcome

# Setup 3 (sign flip, strong confounding):
#   R-learner should handle best since designed for confounding but not really sure cuz linear function and tau is sigmoid
#   T-learner and X-learner don't residualise aggressively — they fit the outcome surfaces directly, which works better here
#  This result could go eitherways for LASSO
# =============================================================

source("code/utils.R")
source("code/1_dgp.R")
source("code/2_dgp.R")
source("code/3_dgp.R")
source("code/learners_lasso.R")

# helper to run all 4 learners and print results neatly
run_lasso_test <- function(dat, setup_name) {
  
  # 80/20 train/test split
  n   <- nrow(dat$X)
  tr  <- 1:floor(n * 0.8)
  te  <- (floor(n * 0.8) + 1):n
  
  X_tr <- dat$X[tr, ];  Y_tr <- dat$Y[tr];  W_tr <- dat$W[tr]
  X_te <- dat$X[te, ];  tau_true <- dat$tau[te]
  
  # run all four learners
  s_mse <- mean((s_learner_lasso(X_tr, Y_tr, W_tr, X_te) - tau_true)^2)
  t_mse <- mean((t_learner_lasso(X_tr, Y_tr, W_tr, X_te) - tau_true)^2)
  x_mse <- mean((x_learner_lasso(X_tr, Y_tr, W_tr, X_te) - tau_true)^2)
  r_mse <- mean((r_learner_lasso(X_tr, Y_tr, W_tr, X_te) - tau_true)^2)
  
  # find winner
  mses   <- c(S = s_mse, T = t_mse, X = x_mse, R = r_mse)
  winner <- names(which.min(mses))
  
  # print results
  cat(sprintf("\n=== %s (n=%d) ===\n", setup_name, n))
  cat(sprintf("  S-learner MSE: %.4f\n", s_mse))
  cat(sprintf("  T-learner MSE: %.4f\n", t_mse))
  cat(sprintf("  X-learner MSE: %.4f\n", x_mse))
  cat(sprintf("  R-learner MSE: %.4f\n", r_mse))
  cat(sprintf("  Winner: %s-learner\n", winner))
  
  # return results as a named vector for easy comparison later
  invisible(mses)
}


# Run all 3 setups (Strong confounding, RCT, Sign Flip in treatment) using n=1000 — large enough to see patterns, small enough to run quickly
set.seed(42)

results_1 <- run_lasso_test(gen_setup_1(n = 1000), "Setup 1: Hard confounding")
results_2 <- run_lasso_test(gen_setup_2(n = 1000), "Setup 2: RCT")
results_3 <- run_lasso_test(gen_setup_3(n = 1000), "Setup 3: Sign flip")

# print a clean comparison across all setups
cat("\n")
cat("SUMMARY: lasso learners across all 3 setups (n=1000)\n")
cat(sprintf("%-12s %-10s %-10s %-10s %-10s\n",
            "Setup", "S", "T", "X", "R"))
cat(sprintf("%-12s %-10s %-10s %-10s %-10s\n",
            "------", "----", "----", "----", "----"))
cat(sprintf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n",
            "Setup 1", results_1["S"], results_1["T"],
            results_1["X"], results_1["R"]))
cat(sprintf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n",
            "Setup 2", results_2["S"], results_2["T"],
            results_2["X"], results_2["R"]))
cat(sprintf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n",
            "Setup 3", results_3["S"], results_3["T"],
            results_3["X"], results_3["R"]))
cat("========================================================\n")
cat("Lower MSE = better. Bold = winner per setup.\n")
cat("Expected: R wins Setup 1, all similar Setup 2,\n")
cat("          R or X wins Setup 3\n")

