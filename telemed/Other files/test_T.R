library(MASS)
library(rlearner)

# Use seed 1 — first completed iteration from your simulation
set.seed(1)
n = 300

# Regenerate data exactly as in the simulation
cor_matrix = matrix(c(
  1.00,  0.05, -0.10,  0.35,  0.25,
  0.05,  1.00,  0.05,  0.00, -0.05,
  -0.10,  0.05,  1.00, -0.20, -0.40,
  0.35,  0.00, -0.20,  1.00,  0.45,
  0.25, -0.05, -0.40,  0.45,  1.00
), nrow = 5, byrow = TRUE)

z           = MASS::mvrnorm(n, mu = rep(0, 5), Sigma = cor_matrix)
x1_bp       = 110 + 80 * pnorm(z[,1])
x2_travel   = 20  + 60 * pnorm(z[,2])
x3_digital  = round(20 * pnorm(z[,3]))
x4_comorbid = round(5  * pnorm(z[,4]))
x5_age      = round(40 + 40 * pnorm(z[,5]))
x           = cbind(x1_bp, x2_travel, x3_digital, x4_comorbid, x5_age)
colnames(x) = c("bp_baseline","travel_time","digital_prior","comorbidities","age")

x1_s = as.numeric(scale(x1_bp))
x2_s = as.numeric(scale(x2_travel))
x3_s = as.numeric(scale(x3_digital))
x4_s = as.numeric(scale(x4_comorbid))
x5_s = as.numeric(scale(x5_age))

propensity = pmax(0.05, pmin(plogis(-0.5 + 1.5*x2_s + 0.5*x1_s - 0.4*x5_s), 0.95))
w          = rbinom(n, 1, propensity)
tau_x      = -2.0*x2_s + 2.0*x1_s + 1.5*x4_s - 0.8*x3_s + 0.5*x5_s - 0.7
b_x        = 0.6*x1_bp + 2.5*x4_comorbid + 0.3*x5_age - 1.2*x3_digital + 20
y          = pmax(b_x + (w - propensity)*tau_x + rnorm(n, 0, 5), 80)

train_idx = sample(1:n, 200)
test_idx  = setdiff(1:n, train_idx)
x_train = x[train_idx,]; x_test = x[test_idx,]
w_train = w[train_idx];  w_test = w[test_idx]
y_train = y[train_idx];  y_test = y[test_idx]
tau_test = tau_x[test_idx]

# --- Step 1: Check what tboost actually produces ---
cat("Fitting tboost directly with verbose=TRUE...\n")
tboost_fit = tboost(x_train, w_train, y_train,
                    num_search_rounds     = 5,
                    k_folds_mu1               = 5,
                    k_folds_mu0 = 5,
                    ntrees_max            = 200,
                    early_stopping_rounds = 5,
                    verbose               = TRUE)

tboost_est = predict(tboost_fit, x_test)

cat("\n--- tboost estimate summary ---\n")
cat("Mean:  ", round(mean(tboost_est), 4), "\n")
cat("SD:    ", round(sd(tboost_est), 4), "\n")
cat("Min:   ", round(min(tboost_est), 4), "\n")
cat("Max:   ", round(max(tboost_est), 4), "\n")
cat("Range: ", round(max(tboost_est) - min(tboost_est), 4), "\n")

# --- Step 2: Compare against true tau on test set ---
cat("\n--- Comparison with true tau ---\n")
cat("True tau mean: ", round(mean(tau_test), 4), "\n")
cat("True tau SD:   ", round(sd(tau_test), 4), "\n")
cat("Norm MSE:      ", round(mean((tboost_est - tau_test)^2) / var(tau_test), 4), "\n")
cat("Rank corr:     ", round(cor(tboost_est, tau_test, method="spearman"), 4), "\n")

# --- Step 3: Look at what the T-learner is actually fitting ---
# The T-learner fits mu_hat(1)(x) and mu_hat(0)(x) separately
# tboost_est = mu_hat(1)(x_test) - mu_hat(0)(x_test)
# If either fit is unstable the difference will be noisy

cat("\n--- Treatment arm sizes in training data ---\n")
cat("Treated (W=1):", sum(w_train == 1), "patients\n")
cat("Control (W=0):", sum(w_train == 0), "patients\n")
cat("Ratio:        ", round(sum(w_train==1)/sum(w_train==0), 3), "\n")

# --- Step 4: Check if the problem is arm imbalance ---
# The T-learner trains a separate model on each arm
# With only ~80-100 patients per arm and 300 trees
# boosting can easily overfit each arm separately
# producing a noisy difference
cat("\n--- Variance of predictions by arm model ---\n")
cat("Check: is tboost producing wildly different predictions\n")
cat("for treated vs control arms?\n")
print(quantile(tboost_est, probs = c(0.05, 0.25, 0.5, 0.75, 0.95)))

# Regenerate seed 1 data as before, then:

cat("Fitting xboost directly...\n")
xboost_fit = xboost(x_train, w_train, y_train,
                    num_search_rounds     = 5,
                    k_folds_mu1               = 5,
                    k_folds_mu0 = 5,
                    ntrees_max            = 200,
                    early_stopping_rounds = 5,
                    verbose               = TRUE)

xboost_est = predict(xboost_fit, x_test)

cat("\n--- xboost vs tboost comparison ---\n")
cat("              tboost    xboost\n")
cat("Mean:      ", sprintf("%8.4f", mean(tboost_est)),
    sprintf("%8.4f\n", mean(xboost_est)))
cat("SD:        ", sprintf("%8.4f", sd(tboost_est)),
    sprintf("%8.4f\n", sd(xboost_est)))
cat("Min:       ", sprintf("%8.4f", min(tboost_est)),
    sprintf("%8.4f\n", min(xboost_est)))
cat("Max:       ", sprintf("%8.4f", max(tboost_est)),
    sprintf("%8.4f\n", max(xboost_est)))
cat("Range:     ", sprintf("%8.4f", max(tboost_est)-min(tboost_est)),
    sprintf("%8.4f\n", max(xboost_est)-min(xboost_est)))
cat("Norm MSE:  ", sprintf("%8.4f", mean((tboost_est-tau_test)^2)/var(tau_test)),
    sprintf("%8.4f\n", mean((xboost_est-tau_test)^2)/var(tau_test)))
cat("Rank corr: ", sprintf("%8.4f", cor(tboost_est, tau_test, method="spearman")),
    sprintf("%8.4f\n", cor(xboost_est, tau_test, method="spearman")))

# Check quantile spread
cat("\nQuantile comparison:\n")
qt = quantile(tboost_est, c(0.05, 0.25, 0.5, 0.75, 0.95))
qx = quantile(xboost_est, c(0.05, 0.25, 0.5, 0.75, 0.95))
comparison = data.frame(
  Quantile = c("5th","25th","50th","75th","95th"),
  tboost   = round(qt, 3),
  xboost   = round(qx, 3)
)
print(comparison, row.names = FALSE)