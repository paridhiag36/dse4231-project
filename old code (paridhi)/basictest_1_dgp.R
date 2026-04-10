# load the function
source("code/1_dgp.R")


# SIMPLEST TEST
# generate 20000 fake people
dat <- gen_setup_1(n = 20000)

# check the shape of everything
cat("X shape:  ", nrow(dat$X), "rows x", ncol(dat$X), "cols\n")
cat("W values: ", table(dat$W), "\n")          # should be roughly half treated, half not
cat("Y range:  ", round(range(dat$Y), 2), "\n") # should be roughly -3 to 4
cat("tau range:", round(range(dat$tau), 2), "\n") # should be 0 to 1
cat("e range:  ", round(range(dat$e), 2), "\n")   # should be 0.1 to 0.9

# visually check that tau is what we expect
# high X1 + X2 should mean high tau
plot(dat$X[,1] + dat$X[,2], dat$tau,
     xlab = "X1 + X2",
     ylab = "True tau",
     main = "tau should increase linearly with X1 + X2")


# CONFOUNDING CHECK
# If confounding is real, people who are more likely to be treated should also have higher baseline outcomes — even before treatment.
# We check this by looking at whether e(X) predicts b(X).
# In plain english: do the "likely to be treated" people also tend to have better outcomes for unrelated reasons?

# recompute b(X) directly from X so we can inspect it
b_check <- sin(pi * dat$X[,1] * dat$X[,2]) +
  2 * (dat$X[,3] - 0.5)^2 +
  dat$X[,4] +
  0.5 * dat$X[,5]

# plot propensity vs baseline
plot(dat$e, b_check,
     xlab = "Propensity score e(X) — how likely to be treated",
     ylab = "Baseline outcome b(X) — outcome ignoring treatment",
     main = "Confounding check: e(X) vs b(X)\nIf correlated, confounding exists",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.5))
# add a trend line to make it obvious
abline(lm(b_check ~ dat$e), col = "red", lwd = 2)

# print the correlation number
cat("Correlation between e(X) and b(X):", round(cor(dat$e, b_check), 3), "\n")
# if this is clearly non-zero, confounding is real


# NAIVE ESTIMATOR CHECK
# The simplest possible "treatment effect" estimate: just compare average Y for treated vs untreated people.
# This is what a non-statistician would do. But, in this case, it should be wrong because of confounding.

naive_ate <- mean(dat$Y[dat$W == 1]) - mean(dat$Y[dat$W == 0])
true_ate  <- mean(dat$tau)   # the real average treatment effect

cat("True average treatment effect:  ", round(true_ate, 3), "\n")
cat("Naive estimate (treated - control):", round(naive_ate, 3), "\n")
cat("Bias from confounding:           ", round(naive_ate - true_ate, 3), "\n")
# the naive estimate should be noticeably higher than the truth
# because treated people were already going to do better (confounding)


# HETEROGENEITY CHECK
# We want tau(X) to vary meaningfully across people.
# If everyone has tau ≈ 0.5, there's nothing interesting to find. If tau ranges widely, the learner has a real job to do.

cat("--- Treatment effect heterogeneity ---\n")
cat("Mean tau:   ", round(mean(dat$tau), 3), "\n")
cat("Std dev tau:", round(sd(dat$tau), 3), "\n")
cat("Min tau:    ", round(min(dat$tau), 3), "\n")
cat("Max tau:    ", round(max(dat$tau), 3), "\n")

# visualise the distribution of true treatment effects
hist(dat$tau,
     breaks = 20,
     main   = "Distribution of true treatment effects tau(X)\nShould be spread out, not all the same",
     xlab   = "True tau(X) for each person",
     col    = "steelblue",
     border = "white")

# add a vertical line at the mean
abline(v = mean(dat$tau), col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("mean =", round(mean(dat$tau), 2)),
       col = "red", lty = 2, lwd = 2)


