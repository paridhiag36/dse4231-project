# load the function
source("code/2_dgp.R")

set.seed(42)
dat <- gen_setup_2(n = 20000)

# BASIC SHAPE CHECK
cat("X shape:  ", nrow(dat$X), "rows x", ncol(dat$X), "cols\n")
cat("W values: ", table(dat$W), "\n")
# should be exactly 50/50 since e=0.5 always

cat("Y range:  ", round(range(dat$Y), 2), "\n")
cat("tau range:", round(range(dat$tau), 2), "\n")
cat("e range:", round(range(dat$e), 2), "\n")
# this should print 0.5 0.5 — everyone has identical propensity

# TAU SHAPE CHECK 
# tau = X1 + log(1 + exp(X2))
# log(1+exp(X2)) is always positive, so tau can be negative only when X1 is sufficiently negative
plot(dat$X[,1], dat$tau,
     xlab = "X1",
     ylab = "True tau",
     main = "tau vs X1\nShould be linear with positive slope",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.3))
abline(lm(dat$tau ~ dat$X[,1]), col = "red", lwd = 2)

plot(dat$X[,2], dat$tau,
     xlab = "X2",
     ylab = "True tau",
     main = "tau vs X2\nShould be nonlinear (log-exp curve)",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.3))

# NO CONFOUNDING CHECK
# This is different from setup 1
# e(X) = 0.5 always, so it literally cannot correlate with b(X)
# this helps confirm the RCT property

b_check <- pmax(dat$X[,1] + dat$X[,2], dat$X[,3], 0) +
  pmax(dat$X[,4] + dat$X[,5], 0)

plot(dat$e, b_check,
     xlab = "Propensity score e(X) — should be 0.5 for everyone",
     ylab = "Baseline outcome b(X)",
     main = "No confounding check: e(X) vs b(X)\nShould be a VERTICAL LINE at 0.5, not a slope",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.3))
# abline(lm(b_check ~ dat$e), col = "red", lwd = 2) - will throw an error, expected


cat("Correlation between e(X) and b(X):",
    round(cor(dat$e, b_check), 3), "\n")
# e is constant (0.5) in an RCT so correlation is undefined

# NAIVE ESTIMATOR CHECK
# Also diff from setup 1: since there is no confounding, the naive estimator should be approximately unbiased — close to the true ATE
# this is what makes RCTs the gold standard
naive_ate <- mean(dat$Y[dat$W == 1]) - mean(dat$Y[dat$W == 0])
true_ate  <- mean(dat$tau)

cat("True average treatment effect:    ", round(true_ate, 3), "\n")
cat("Naive estimate (treated-control): ", round(naive_ate, 3), "\n")
cat("Bias from confounding:            ", round(naive_ate - true_ate, 3), "\n")
# bias should be near ZERO — compare to Setup 1 bias of 0.399
# this directly shows why RCTs eliminate confounding bias

# HETEROGENEITY CHECK
cat("--- Treatment effect heterogeneity ---\n")
cat("Mean tau:   ", round(mean(dat$tau), 3), "\n")
cat("Std dev tau:", round(sd(dat$tau), 3), "\n")
cat("Min tau:    ", round(min(dat$tau), 3), "\n")
cat("Max tau:    ", round(max(dat$tau), 3), "\n")
# tau has real variation — the challenge here is recovering
# the nonlinear shape, not removing confounding

hist(dat$tau,
     breaks = 20,
     main   = "Distribution of true tau(X) in Setup 2\nNonlinear, should be right-skewed (log-exp term always positive)",
     xlab   = "True tau(X) for each person",
     col    = "steelblue",
     border = "white")
abline(v = mean(dat$tau), col = "red", lwd = 2, lty = 2)
legend("topright",
       legend = paste("mean =", round(mean(dat$tau), 2)),
       col = "red", lty = 2, lwd = 2)

