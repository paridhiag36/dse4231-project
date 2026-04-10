# =============================================================
# FILE: 2_dgp.R
# PURPOSE: Setup 2 DGP — Randomised Trial (no confounding)

# Based on Setup B from Nie & Wager (2021), Section 6.1

# A real-world story eg:
# A clinical drug trial where patients are randomly assigned to treatment or control by a coin flip. 
# No self-selection, no confounding — whoever gets the drug is determined purely by randomisation.

# This is the easiset possible case for all learners. Since e(X) = 0.5 for everyone, there is literally no confounding to disentangle. Even naive methods should work.
# If R-learner fails here, that means the code is broken. If all learners do similarly well, that's expected and correct.

# Diff from setup 1:
# Setup 1: propensity depends on X (confounded)
# Setup 2: propensity = 0.5 always (not confounded)
# But tau(X) is more complex here — nonlinear function of X.
# The challenge shifts from "remove confounding" to "recover a complex treatment effect function."

# The math setup:
# X ~ N(0, I_d)          — standard normal covariates
# e(X) = 0.5             — pure randomisation, no confounding
# b(X) = complex         — complicated baseline (irrelevant here)
# tau(X) = X1 + log(1 + exp(X2))  — nonlinear, varies with X
# Y = b(X) + (W-0.5)*tau(X) + noise
# =============================================================

gen_setup_2 <- function(n,         # number of people
                        d  = 10,   # number of covariates
                        sigma = 1  # noise level
) {
  
  # covariates: standard normal
  # unlike Setup 1 which uses Uniform(0,1), Setup B uses Normal(0,1) — can be negative
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  
  # propensity: exactly 0.5 for everyone 
  # this is pure randomisation — no person is more or less likely to be treated based on their characteristics
  e <- rep(0.5, n)
  W <- rbinom(n, size = 1, prob = e)
  
  # baseline outcome b(X) 
  # complex function but irrelevant for treatment effect recovery since there's no confounding, learners don't need to model this carefully — it just adds noise
  b <- pmax(X[,1] + X[,2], X[,3], 0) +
    pmax(X[,4] + X[,5], 0)
  
  # True treatment effect tau(X)
  # nonlinear: X1 shifts it linearly, X2 adds a soft-plus curve
  # log(1 + exp(X2)) is always positive and nonlinear in X2
  # this is harder to recover than Setup 1's linear tau
  tau <- X[,1] + log(1 + exp(X[,2]))
  
  # observed outcome
  Y <- b + (W - 0.5) * tau + sigma * rnorm(n)
  
  list(X = X, W = W, Y = Y, tau = tau, e = e)
}