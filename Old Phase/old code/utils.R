# =============================================================
# FILE: code/utils.R
# PURPOSE: Shared utility functions used across all learner files
#

# Two things are used by multiple learner files and don't belong in any one learner file:
# 1. standardise_x() — the paper standardises X (center + scale) before fitting any lasso model. This ensures lasso penalises all coefficients fairly regardless of variable scale.
# Used by: all lasso learners (S, T, X, R)
# 2. cvboost() — the paper's XGBoost fitting function with random hyperparameter search and early stopping.
# Used by: all boost learners (S, T, X, R)
# =============================================================


library(glmnet)
library(caret)    # for preProcess standardisation
library(xgboost)  # for cvboost
library(ranger)   # for RF learners (our addition replacing KRR)



# =============================================================
# UTILITY 1: standardise_x
# Centers each column of X to mean 0 and scales to std dev 1. Returns both the scaled X and the standardisation parameters so we can apply the same scaling to test data later.
# This matter for LASSO because Lasso adds a penalty of lambda * sum(|beta_j|) to the loss.
# If X1 is measured in dollars (range 0-100000) and X2 is age (range 0-100), the same lambda shrinks beta_1 much more than beta_2 just because of scale differences.
# Standardising first puts all variables on equal footing.

# Usage:
#   scl <- standardise_x(X_train)
#   X_train_scaled <- scl$X          # use for training
#   X_test_scaled  <- scl$apply(X_test)  # apply same scaling to test
# =============================================================

standardise_x <- function(X) {
  # compute mean and sd from training data only
  # never use test data statistics — that would leak information
  params <- caret::preProcess(as.data.frame(X),
                              method = c("center", "scale"))
  
  # apply to training data
  X_scaled <- as.matrix(
    predict(params, as.data.frame(X))
  )
  
  # remove any columns that became NA (zero variance columns)
  # these can't be standardised and cause errors in glmnet
  valid_cols <- !is.na(colSums(X_scaled))
  X_scaled   <- X_scaled[, valid_cols, drop = FALSE]
  
  list(
    # the scaled training matrix — use this for fitting
    X = X_scaled,
    
    # a function to apply the SAME scaling to new data
    # always call this on test data, never refit the scaler
    apply = function(X_new) {
      X_new_scaled <- as.matrix(
        predict(params, as.data.frame(X_new))
      )
      X_new_scaled[, valid_cols, drop = FALSE]
    }
  )
}


# =============================================================
# UTILITY 2: cvboost
# Fits an XGBoost model with random hyperparameter search. Instead of fixed hyperparameters, it tries num_search_rounds random combinations and picks the best one by cross-validation.
# Random search instead of fixed defaults because XGBoost has many hyperparameters that interact with each other and with the data. No single default works well across all setups. The paper searches over:
#   - subsample:        how much data each tree sees
#   - colsample_bytree: how many features each tree uses
#   - eta:              learning rate (smaller = more careful)
#   - max_depth:        how complex each tree is
#   - gamma:            minimum gain to make a split
#   - min_child_weight: minimum data in each leaf
#   - max_delta_step:   maximum step size per update
# With early stopping: if the CV error stops improving for early_stopping_rounds consecutive rounds, stop adding trees. This prevents overfitting and saves computation.
# Weights: passed directly into xgb.DMatrix so the weighted R-learner loss is correctly optimised.
# =============================================================

cvboost <- function(X,
                    y,
                    weights            = NULL,
                    objective          = c("reg:squarederror",
                                           "binary:logistic"),
                    k_folds            = NULL,
                    ntrees_max         = 1000,
                    num_search_rounds  = 10,
                    early_stopping_rounds = 10,
                    verbose            = FALSE) {
  
  objective <- match.arg(objective)
  eval_metric <- if (objective == "reg:squarederror") "rmse" else "logloss"
  
  # auto-set number of folds based on sample size
  # at least 3, at most 10, roughly n/4
  if (is.null(k_folds)) {
    k_folds <- floor(max(3, min(10, length(y) / 4)))
  }
  
  # if no weights provided, use equal weights (standard regression)
  if (is.null(weights)) weights <- rep(1, length(y))
  
  # XGBoost requires its own DMatrix format
  dtrain <- xgb.DMatrix(
    data   = as.matrix(X),
    label  = y,
    weight = weights
  )
  
  # --- random hyperparameter search ---
  # try num_search_rounds random combinations, keep the best
  best_loss  <- Inf
  best_param <- list()
  best_seed  <- 1234
  best_cvfit <- NULL
  
  for (iter in seq_len(num_search_rounds)) {
    
    # randomly sample one combination of hyperparameters
    # this is the paper's exact grid
    param <- list(
      objective        = objective,
      eval_metric      = eval_metric,
      subsample        = sample(c(0.5, 0.75, 1), 1),
      colsample_bytree = sample(c(0.6, 0.8, 1), 1),
      eta              = sample(c(5e-3, 1e-2, 0.015, 0.025,
                                  5e-2, 8e-2, 1e-1, 2e-1), 1),
      max_depth        = sample(3:20, 1),
      gamma            = runif(1, 0, 0.2),
      min_child_weight = sample(1:20, 1),
      max_delta_step   = sample(1:10, 1)
    )
    
    seed_i <- sample.int(100000, 1)
    set.seed(seed_i)
    
    # cross-validate with early stopping to find optimal n trees
    cv_fit <- xgb.cv(
      params                = param,
      data                  = dtrain,
      nfold                 = k_folds,
      nrounds               = ntrees_max,
      early_stopping_rounds = early_stopping_rounds,
      maximize              = FALSE,
      verbose               = verbose,
      prediction            = TRUE,
      callbacks             = list(cb.cv.predict(save_models = TRUE))
    )
    
    # extract best CV loss for this hyperparameter combination
    metric_col <- paste0("test_", eval_metric, "_mean")
    min_loss   <- min(cv_fit$evaluation_log[[metric_col]])
    
    # keep this combination if it beats the current best
    if (min_loss < best_loss) {
      best_loss  <- min_loss
      best_param <- param
      best_seed  <- seed_i
      best_cvfit <- cv_fit
    }
  }
  
  # refit on full training data using best hyperparameters
  set.seed(best_seed)
  final_fit <- xgb.train(
    params  = best_param,
    data    = dtrain,
    nrounds = best_cvfit$best_ntreelimit,
    verbose = 0
  )
  
  # return everything needed for prediction
  list(
    fit        = final_fit,
    best_param = best_param,
    best_loss  = best_loss,
    cv_preds   = best_cvfit$pred   # in-sample CV predictions
  )
}


# helper: predict from a cvboost object on new data
predict_cvboost <- function(cvboost_obj, X_new = NULL) {
  if (is.null(X_new)) {
    # return in-sample CV predictions (used by rboost nuisance step)
    return(cvboost_obj$cv_preds)
  }
  dtest <- xgb.DMatrix(data = as.matrix(X_new))
  as.numeric(predict(cvboost_obj$fit, dtest))
}
