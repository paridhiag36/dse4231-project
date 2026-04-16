# load the function
source("code/3_dgp.R")

set.seed(42)
dat <- gen_setup_3(n = 20000)

# BASIC SHAPE CHECK
cat("X shape:  ", nrow(dat$X), "rows x", ncol(dat$X), "cols\n")
cat("W values: ", table(dat$W), "\n")
# should be roughly unequal since high-risk + far patients more likely treated

cat("Y range:  ", round(range(dat$Y), 2), "\n")
cat("tau range:", round(range(dat$tau), 2), "\n")
# tau should range from roughly -1.5 to +1.5
# negative = low digital literacy patients being harmed, positive = high-risk far-away patients benefiting
cat("e range:  ", round(range(dat$e), 2), "\n")
# should be 0.1 to 0.9 — moderate confounding, not extreme


# SIGN FLIP CHECK
# This is the defining feature of Setup 3 — unlike Setups 1 and 2 where tau is always positive, here tau changes sign.
# A method that predicts the average for everyone gets the direction wrong for a large fraction of patients.
cat("--- Sign flip check ---\n")
cat("% patients with POSITIVE tau (benefit):",
    round(mean(dat$tau > 0) * 100, 1), "%\n")
cat("% patients with NEGATIVE tau (harmed): ",
    round(mean(dat$tau < 0) * 100, 1), "%\n")
cat("% patients with ZERO tau:              ",
    round(mean(dat$tau == 0) * 100, 1), "%\n")
# should see meaningful mass on both sides
# roughly 38% positive, 62% negative based on our DGP


# TAU SHAPE CHECK
# tau has two drivers:
#   positive component: plogis(X1 + X3 - 1) — increases with X1 and X3
#   negative component: plogis(-X4 - 0.5)   — increases when X4 is low
# so tau should increase with X1, X3 and decrease with X4

plot(dat$X[,1] + dat$X[,3], dat$tau,
     xlab = "X1 + X3 (risk + distance)",
     ylab = "True tau",
     main = "tau vs X1+X3\nShould increase — high risk + far away patients benefit more",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.2))
abline(lm(dat$tau ~ I(dat$X[,1] + dat$X[,3])), col = "red", lwd = 2)

plot(dat$X[,4], dat$tau,
     xlab = "X4 (digital literacy)",
     ylab = "True tau",
     main = "tau vs X4\nShould increase — high literacy patients harmed less",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.2))
abline(lm(dat$tau ~ dat$X[,4]), col = "red", lwd = 2)


# CONFOUNDING CHECK
# Unlike Setup 2 (RCT), confounding is present here. High-risk (X1) and far-away (X3) patients are both more likely to be treated and have worse baseline outcomes.
# e(X) and b(X) should be correlated.

b_check <- -0.5 * dat$X[,1] + 0.3 * dat$X[,2] + 0.2 * dat$X[,5]

plot(dat$e, b_check,
     xlab = "Propensity score e(X) — how likely to be treated",
     ylab = "Baseline outcome b(X) — outcome ignoring treatment",
     main = "Confounding check: e(X) vs b(X)\nShould show a slope — moderate confounding",
     pch  = 16, col = adjustcolor("steelblue", alpha = 0.3))
abline(lm(b_check ~ dat$e), col = "red", lwd = 2)

cat("Correlation between e(X) and b(X):", round(cor(dat$e, b_check), 3), "\n")
# should be nonzero (confounding present) but weaker than Setup 1 (0.686)
# Setup 3 has moderate confounding, not severe
# But we see stronger than Setup A
# can chnage e in 3_dgp.R or can just defend this as well.
# Here is why: distance to clinic (X3) is the main reason for telemedicine enrolment, chronic risk (X1) plays a smaller role
# e <- plogis(0.1 * X[,1] + 0.4 * X[,3] - 0.1 * X[,2])


# NAIVE ESTIMATOR CHECK
# Because confounding is present, naive comparison is biased.
# But the direction of bias is different from Setup 1 — here treated patients are sicker (higher risk) so they tend to have worse outcomes, making the naive estimate understate or even reverse the true treatment effect.

naive_ate <- mean(dat$Y[dat$W == 1]) - mean(dat$Y[dat$W == 0])
true_ate  <- mean(dat$tau)

cat("True average treatment effect:    ", round(true_ate, 3), "\n")
cat("Naive estimate (treated-control): ", round(naive_ate, 3), "\n")
cat("Bias from confounding:            ", round(naive_ate - true_ate, 3), "\n")
# true ATE ≈ -0.11 (slightly negative)
# naive estimate will differ due to confounding


# HETEROGENEITY CHECK
cat("--- Treatment effect heterogeneity ---\n")
cat("Mean tau:   ", round(mean(dat$tau), 3), "\n")
cat("Std dev tau:", round(sd(dat$tau), 3), "\n")
cat("Min tau:    ", round(min(dat$tau), 3), "\n")
cat("Max tau:    ", round(max(dat$tau), 3), "\n")
# std dev should be around 0.45-0.50
# much higher variance than Setup 1 (0.207) — more heterogeneity


# SIGN FLIP VISUALISATION
# The most important plot for Setup 3.
# Unlike Setups 1 and 2 where the histogram is one-sided, this should show clear mass on both sides of zero.
# A learner that predicts the mean for everyone gets the direction wrong for everyone.

hist(dat$tau,
     breaks = 30,
     main   = "Distribution of true tau(X) in Setup 3\nMust show mass on BOTH sides of zero — this is the sign flip",
     xlab   = "True tau(X) for each patient",
     col    = "steelblue",
     border = "white")
abline(v = 0,            col = "red",    lwd = 2, lty = 2)
abline(v = mean(dat$tau), col = "orange", lwd = 2)
legend("topright",
       legend = c("zero", paste("mean =", round(mean(dat$tau), 2))),
       col    = c("red", "orange"),
       lty    = 2, lwd = 2)
