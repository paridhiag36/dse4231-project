# =============================================================
# FILE: code/learners_lasso.R
# PURPOSE: S, T, X, R learners implemented via lasso (glmnet)

# These implementations follow Nie & Wager (2021) Section 6.2 as closely as possible, using the paper's GitHub code as the reference implementation.

# Key Design Decisions
# 1. X is standardised before fitting (center + scale)
#    — ensures lasso penalises all features equally
#    — standardisation params from train set applied to test set
# 2. R-learner uses the direct formulation (no pseudo_Y division)
#    — multiply X by W_tilde, regress Y_tilde on X*W_tilde
#    — mathematically equivalent to weighted pseudo-outcome
#    — but numerically stable (no division by near-zero W_tilde)
# 3. S-learner uses (2W-1) encoding, not W as raw covariate
#    — forces the model to find treatment heterogeneity explicitly
#    — avoids the regularisation-toward-zero problem
# 4. All cross-validation uses foldid for consistency
#    — same fold assignments used for m(X), e(X), and tau(X)
#    — reduces variance from random fold splits
#
# Dependencies:
#   source("code/utils.R") — for standardise_x()
# =============================================================

source("code/utils.R")

# =============================================================
# S-LEARNER (lasso)
# What the paper does (slasso.R):Instead of just adding W as a covariate, the paper creates an augmented feature matrix: cbind( (2W-1)*[1, X],  X )
# In plain English:
#   - first block: interaction terms (W flipped to +1/-1) * each X this lets the treatment effect vary with each covariate
#   - second block: main effects of X

# Why (2W-1) instead of W? Because W=1 becomes +1, W=0 becomes -1
#   The treatment effect = 2 * (coefficient on interaction block)
#   This symmetric encoding is more stable for lasso than 0/1

# The intercept column (the "1" in cbind(1, x_scl)) gets penalty_factor = 0 — meaning it is never penalised. This allows a non-zero average treatment effect freely.
# =============================================================

s_learner_lasso <- function(X_train, Y_train, W_train, X_test) {
  
  # standardise X — fit scaler on train, apply to both
  scl      <- standardise_x(X_train)
  X_tr_scl <- scl$X
  X_te_scl <- scl$apply(X_test)
  
  n    <- nrow(X_tr_scl)
  p    <- ncol(X_tr_scl)
  
  # fold assignments — reused across all CV calls for consistency
  k_folds <- floor(max(3, min(10, n / 4)))
  foldid  <- sample(rep(seq(k_folds), length.out = n))
  
  # build augmented feature matrix for training
  # column layout: [(2W-1), (2W-1)*X1, ..., (2W-1)*Xp, X1, ..., Xp]
  # the (2W-1) column is the intercept for the treatment effect
  w_pm1    <- as.numeric(2 * W_train - 1)   # +1 if treated, -1 if control
  X_aug_tr <- cbind(
    w_pm1 * cbind(1, X_tr_scl),   # treatment interactions
    X_tr_scl                       # main effects
  )
  
  # penalty factor: don't penalise the intercept column (column 1)
  # penalise everything else equally
  pen_factor <- c(0, rep(1, 2 * p))
  
  # fit one lasso on the augmented matrix
  s_fit <- cv.glmnet(
    X_aug_tr, Y_train,
    foldid         = foldid,
    penalty.factor = pen_factor,
    standardize    = FALSE,   # we already standardised manually
    alpha          = 1
  )
  
  # extract coefficients (excluding intercept row from glmnet)
  s_beta <- as.vector(t(coef(s_fit, s = "lambda.min")[-1]))
  
  # predict tau on test set
  # for prediction, we want the treatment effect not the outcome
  # set W=1 for everyone in the treatment interaction block
  # set the main effects block to X_test
  # tau(X) = 2 * (treatment interaction block) %*% beta_treatment
  X_aug_te_pred <- cbind(1, X_te_scl, X_te_scl * 0)
  # the *2 comes from the (2W-1) encoding:
  # tau = f(W=1, X) - f(W=0, X) = 2 * (interaction coefficients) %*% X
  tau_hat <- as.numeric(2 * X_aug_te_pred %*% s_beta)
  
  return(tau_hat)
}


# =============================================================
# T-LEARNER (lasso)
# What the paper does (tlasso.R): Fits two separate lasso models:
#   mu1(X): outcome model for treated group only
#   mu0(X): outcome model for control group only
# tau(X) = mu1(X) - mu0(X)

# The paper uses separate fold IDs for each group since the treated and control subsets have different sizes.
# =============================================================

t_learner_lasso <- function(X_train, Y_train, W_train, X_test) {
  
  # standardise X
  scl      <- standardise_x(X_train)
  X_tr_scl <- scl$X
  X_te_scl <- scl$apply(X_test)
  
  # split into treated and control
  idx1 <- which(W_train == 1)
  idx0 <- which(W_train == 0)
  
  X_1 <- X_tr_scl[idx1, ];  Y_1 <- Y_train[idx1]
  X_0 <- X_tr_scl[idx0, ];  Y_0 <- Y_train[idx0]
  
  # separate fold IDs for each group because treated and control are different sizes, a single fold assignment would give unbalanced folds
  k1 <- floor(max(3, min(10, length(idx1) / 4)))
  k0 <- floor(max(3, min(10, length(idx0) / 4)))
  
  foldid1 <- sample(rep(seq(k1), length.out = length(idx1)))
  foldid0 <- sample(rep(seq(k0), length.out = length(idx0)))
  
  # fit separate outcome model for each group
  fit1 <- cv.glmnet(X_1, Y_1, foldid = foldid1, alpha = 1)
  fit0 <- cv.glmnet(X_0, Y_0, foldid = foldid0, alpha = 1)
  
  # predict on test set from both models, take difference
  mu1_hat <- as.numeric(predict(fit1, X_te_scl, s = "lambda.min"))
  mu0_hat <- as.numeric(predict(fit0, X_te_scl, s = "lambda.min"))
  
  return(mu1_hat - mu0_hat)
}


# =============================================================
# X-LEARNER (lasso)
# What the paper does (xlasso.R):
# Stage 1: fit T-learner, predict mu0 and mu1 on all training X (not just on the opposite group — this is key)
# Stage 2: form pseudo-outcomes
#   D1_i = Y_i - mu0(X_i)  for treated people
#   D0_i = mu1(X_i) - Y_i  for control people
# Stage 3: fit tau models on D1 (treated) and D0 (control)
# Stage 4: combine with propensity score  tau(X) = (1 - e(X)) * tau1(X) + e(X) * tau0(X)
# =============================================================

x_learner_lasso <- function(X_train, Y_train, W_train, X_test) {
  
  # standardise X
  scl      <- standardise_x(X_train)
  X_tr_scl <- scl$X
  X_te_scl <- scl$apply(X_test)
  
  idx1 <- which(W_train == 1)
  idx0 <- which(W_train == 0)
  
  X_1 <- X_tr_scl[idx1, ];  Y_1 <- Y_train[idx1]
  X_0 <- X_tr_scl[idx0, ];  Y_0 <- Y_train[idx0]
  
  k1 <- floor(max(3, min(10, length(idx1) / 4)))
  k0 <- floor(max(3, min(10, length(idx0) / 4)))
  
  foldid1 <- sample(rep(seq(k1), length.out = length(idx1)))
  foldid0 <- sample(rep(seq(k0), length.out = length(idx0)))
  
  # base outcome models 
  fit1 <- cv.glmnet(X_1, Y_1, foldid = foldid1, alpha = 1)
  fit0 <- cv.glmnet(X_0, Y_0, foldid = foldid0, alpha = 1)
  
  # predict on Aall training X (not just opposite group)
  # this is what the paper does — predict full mu0 and mu1 then subset to the relevant indices for pseudo-outcomes
  mu1_all <- as.numeric(predict(fit1, X_tr_scl, s = "lambda.min"))
  mu0_all <- as.numeric(predict(fit0, X_tr_scl, s = "lambda.min"))
  
  # pseudo-outcomes
  # for treated: how much did treatment add beyond control baseline?
  D1 <- Y_1 - mu0_all[idx1]
  # for control: how much would treatment have added?
  D0 <- mu1_all[idx0] - Y_0
  
  # fit tau models on pseudo-outcomes 
  tau_fit1 <- cv.glmnet(X_1, D1, foldid = foldid1, alpha = 1)
  tau_fit0 <- cv.glmnet(X_0, D0, foldid = foldid0, alpha = 1)
  
  tau1_hat <- as.numeric(predict(tau_fit1, X_te_scl, s = "lambda.min"))
  tau0_hat <- as.numeric(predict(tau_fit0, X_te_scl, s = "lambda.min"))
  
  # propensity-weighted combination 
  # fit propensity model on full training data
  n    <- nrow(X_tr_scl)
  k_p  <- floor(max(3, min(10, n / 4)))
  f_p  <- sample(rep(seq(k_p), length.out = n))
  
  e_fit <- cv.glmnet(X_tr_scl, W_train,
                     foldid = f_p,
                     family = "binomial",
                     alpha  = 1)
  
  # get propensity on test set
  e_hat <- as.numeric(predict(e_fit, X_te_scl,
                              s    = "lambda.min",
                              type = "response"))
  e_hat <- pmin(pmax(e_hat, 0.05), 0.95)
  
  # weighted combination: trust tau1 when propensity is low
  # (control-like people), trust tau0 when propensity is high
  return((1 - e_hat) * tau1_hat + e_hat * tau0_hat)
}


# =============================================================
# R-LEARNER (lasso)
# What the paper does (rlasso.R): Uses the direct formulation — no division by W_tilde.

# From Robinson's transformation:
#   Y - m(X) = tau(X) * (W - e(X)) + noise
#   Y_tilde  = tau(X) * W_tilde + noise

# If tau(X) = X * beta (linear in X), then: Y_tilde = (W_tilde * X) * beta + noise
# So we can estimate beta by regressing Y_tilde on X*W_tilde. This is equivalent to the weighted pseudo-outcome approach but avoids dividing by W_tilde (which can be near zero).

# Implementation:
# Build X_tilde = cbind(W_tilde, W_tilde * X) — each row of X multiplied by that person's W_tilde value
# Regress Y_tilde on X_tilde
# For prediction: just use X (not X_tilde) since W_tilde is not available at prediction time
# Cross-fitting: nuisance functions m(X) and e(X) are estimated on held-out folds so they don't leak into the tau estimate
# =============================================================

r_learner_lasso <- function(X_train, Y_train, W_train, X_test) {
  
  # standardise X
  scl      <- standardise_x(X_train)
  X_tr_scl <- scl$X
  X_te_scl <- scl$apply(X_test)
  
  n       <- nrow(X_tr_scl)
  k_folds <- floor(max(3, min(10, n / 4)))
  foldid  <- sample(rep(seq(k_folds), length.out = n))
  
  # cross-fitted nuisance estimation 
  # use glmnet's keep=TRUE to get out-of-fold predictions without running a separate loop
  # keep=TRUE stores predictions for each observation using the fold it was held out from
  
  # m(X) = E[Y|X]: expected outcome ignoring treatment
  m_fit <- cv.glmnet(
    X_tr_scl, Y_train,
    foldid = foldid,
    keep   = TRUE,    # stores out-of-fold predictions
    alpha  = 1
  )
  
  # extract out-of-fold m(X) predictions
  # this is the paper's elegant trick — no explicit loop needed
  # fit.preval contains predictions at each lambda for each fold
  m_lambda_idx <- which(m_fit$lambda == m_fit$lambda.min)
  m_hat <- m_fit$fit.preval[, m_lambda_idx]
  
  # e(X) = E[W|X]: propensity score
  e_fit <- cv.glmnet(
    X_tr_scl, W_train,
    foldid     = foldid,
    family     = "binomial",
    keep       = TRUE,
    alpha      = 1
  )
  
  # extract out-of-fold propensity predictions
  # fit.preval stores log-odds for binomial — convert to probability
  e_lambda_idx <- which(e_fit$lambda == e_fit$lambda.min)
  e_hat_logodds <- e_fit$fit.preval[, e_lambda_idx]
  e_hat <- 1 / (1 + exp(-e_hat_logodds))
  e_hat <- pmin(pmax(e_hat, 0.05), 0.95)
  
  # form residuals 
  Y_tilde <- Y_train - m_hat   # outcome residual
  W_tilde <- W_train - e_hat   # treatment residual
  
  # build X_tilde for the direct formulation
  # multiply each row of [1, X] by that person's W_tilde
  # the intercept column (1*W_tilde) allows a constant tau that is never penalised (penalty_factor = 0 for first col)
  X_tilde <- cbind(
    as.numeric(W_tilde) * cbind(1, X_tr_scl)
  )
  
  # penalty: don't penalise the intercept term (average tau)
  pen_factor <- c(0, rep(1, ncol(X_tr_scl)))
  
  # regress Y_tilde on X_tilde — this recovers tau(X) = X * beta
  tau_fit <- cv.glmnet(
    X_tilde, Y_tilde,
    foldid         = foldid,
    penalty.factor = pen_factor,
    standardize    = FALSE,
    alpha          = 1
  )
  
  tau_beta <- as.vector(t(coef(tau_fit, s = "lambda.min")[-1]))
  
  # predict tau on test set
  # at prediction time, W_tilde is unknown — use X directly tau(X_test) = [1, X_test] %*% beta
  X_te_pred <- cbind(1, X_te_scl)
  tau_hat   <- as.numeric(X_te_pred %*% tau_beta)
  
  return(tau_hat)
}