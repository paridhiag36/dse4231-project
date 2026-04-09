# =============================================================
# FILE: dgp.R
# PURPOSE: Functions that generate fake datasets for simulation
#
# What is a DGP (Data Generating Process)?
# Imagine we want to test whether a thermometer is accurate. The easiest way is to put it in water. We know it is exactly 100 degrees if we see and it reads 100.
#
# That's what we're doing here. We create fake people where we decide the true treatment effect for each person. Then we run our learners and check if they recovered it.
#
# Each function in this file is one "test scenario" with different difficulty levels for the learner.
# =============================================================


# -------------------------------------------------------------
# SETUP 1: Hard confounding, simple treatment effect (Based on Setup A from Nie & Wager (2021))
#
# A real-world story eg
# Imagine a job training programme. Motivated people are more likely to both (a) sign up for training and (b) get better jobs anyway because they're motivated.
# This is called confounding: the same thing (motivation) drives both who gets treated as well as what the outcome is.
#
# A naive learner will think "training works amazingly!" when really it's just selecting "already motivated people."
#
# The R-learner from our paper is designed to see through this.
#
# THE MATH SETUP:
# - X: 6 covariates, each drawn from Uniform(0,1). Think of them as: age, education, motivation, etc.
# - e(X): probability of being treated (also called propensity score). Depends on X1*X2 — same interaction as the baseline! This is the confounding.
# - b(X): baseline outcome (i.e. what happens regardless of treatment). It is a complex function of X — the "background noise"
# - tau(X): TRUE treatment effect (i.e. what we want to recover). Simple: just the average of X1 and X2. Would be easy to find if we can remove the confounding first.
# - Y: what we actually observe. Y = baseline + treatment_contribution + random_noise
# -------------------------------------------------------------

gen_setup_1 <- function(n,        # number of people to generate
                        d = 6,    # number of covariates (let's stick to 6)
                        sigma = 1 # noise level (1 = standard)
) {
  
  # --- STEP 1: Generate covariates ---
  # Each person gets d random numbers between 0 and 1
  # Think of each column as one characteristic (age, education, etc)
  # X is a matrix: n rows (people) x d columns (characteristics)
  X <- matrix(
    runif(n * d),   # n*d random numbers from Uniform(0,1)
    nrow = n,       # arrange into n rows
    ncol = d        # and d columns
  )
  
  # --- STEP 2: Compute the TRUE propensity score e(X) ---
  # This is the real probability each person gets treated.
  # In real life we never know this — but we're making fake data here so we get to decide it.
  #
  # Why sin(pi * X1 * X2)?
  # Because this is the same interaction used in the baseline b(X).
  # That overlap is what creates confounding — the same signal that predicts treatment also predicts the outcome.
  #
  # Why trim between 0.1 and 0.9?
  # If propensity is exactly 0 or 1, some people are never or always treated. Then we have no comparison group for them and the math breaks down.
  # Trimming keeps everyone in a "comparable" range.
  e <- pmin(                        # pmin = element-wise minimum
    pmax(                       # pmax = element-wise maximum
      sin(pi * X[,1] * X[,2]), # raw propensity: between -1 and 1
      0.1                       # floor: never below 10% chance
    ),
    0.9                         # ceiling: never above 90% chance
  )
  # after trimming: every person has between 10%-90% chance of treatment
  
  # --- STEP 3: Assign actual treatment W ---
  # We flip a weighted coin for each person.
  # Person i gets treated (W=1) with probability e[i],and not treated (W=0) with probability 1 - e[i].
  #
  # rbinom(n, size=1, prob=e) = n independent coin flips where coin i has P(heads) = e[i]
  W <- rbinom(n, size = 1, prob = e)
  
  # --- STEP 4: Compute the baseline outcome b(X) ---
  # This is what would happen to each person regardless of whether they get treated. It's the "background noise" the learner has to see through.
  #
  # Why is this complicated?
  # To make the problem hard! If the baseline were simple, every learner could find the treatment effect easily.
  # The complex baseline is what separates good methods from bad.
  b <- sin(pi * X[,1] * X[,2]) +   # nonlinear interaction of X1, X2
    2 * (X[,3] - 0.5)^2 +        # quadratic in X3
    X[,4] +                       # linear in X4
    0.5 * X[,5]                   # linear in X5 (weaker)
  # X6 does nothing — it's a red herring covariate
  
  # --- STEP 5: Compute the TRUE treatment effect tau(X) ---
  # This is what we are trying to recover with our learners. But we know it here because we made it up.
  #
  # Why is it simple (just the average of X1 and X2)?
  # The paper's point is that the treatment effect is simple, but that it's buried under complicated confounding and baseline.
  # A good method strips away the noise and finds the simple truth.
  tau <- (X[,1] + X[,2]) / 2
  # tau ranges from 0 to 1, averaging 0.5
  # people with high X1 and X2 benefit more from treatment
  
  # --- STEP 6: Compute the observed outcome Y ---
  # This is the only thing the learner gets to see (along with X and W). It has to reverse-engineer tau from this.
  # The formula comes from the potential outcomes framework:
  # Y = baseline + treatment_effect_contribution + noise
  #
  # Why (W - 0.5) * tau instead of just W * tau?
  # Centering W around 0.5 instead of 0 is a common trick because it makes the intercept of b(X) interpretable as the average outcome, not the control outcome. (The paper uses this convention throughout)
  Y <- b +                      # baseline: what happens anyway
    (W - 0.5) * tau +        # treatment contribution
    sigma * rnorm(n)         # random noise: N(0, sigma^2)
  
  # --- Returning everything as a named list ---
  # The learner will only receive X, W, Y
  # We keep tau and e for evaluation (to see how close the learner gets)
  list(
    X   = X,    # covariates (n x d matrix) — learner sees this
    W   = W,    # treatment assignment (0/1 vector) — learner sees this
    Y   = Y,    # observed outcome (numeric vector) — learner sees this
    tau = tau,  # TRUE treatment effect — only for checking answers
    e   = e     # TRUE propensity — only for checking answers
  )
}
