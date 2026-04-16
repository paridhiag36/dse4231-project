# =============================================================
# FILE: code/3_dgp.R
# PURPOSE: Setup 3 DGP — Sign-flip treatment heterogeneity

# This is our extension — not in the original paper.

# A real-world story eg:
# A hospital rolls out telemedicine follow-up for chronic patients. High-risk patients who live far from the clinic benefit a lot— they get care they otherwise wouldn't have access to.
# But low digital-literacy patients are mildly harmed— they struggle with the technology and miss nuances that an in-person visit would catch.

# The average treatment effect is near zero because benefits and harms roughly cancel out across the population.
# But there is v strong heterogeneity — tau(X) actually changes sign depending on who the patient is.

# This is also harder than setups 1 and 2:
# Setup 1: tau(X) always positive, just bigger for some people
# Setup 2: tau(X) nonlinear but always has clear direction
# Setup 3: tau(X) is +ve for some, -ve for others

# A method that predicts the average (near zero) for everyone looks okay on average MSE but is wrong for everyone.
# Need to actually find the heterogeneity to do well.

# Covariates:
#   X1 = chronic risk score  (higher = sicker)
#   X2 = age proxy
#   X3 = distance to clinic  (higher = farther away)
#   X4 = digital literacy    (higher = more tech comfortable)
#   X5-X10 = other patient characteristics (noise)

# Treatment Effect (the sign flip):
#   Positive: high-risk patients far from clinic benefit (+1.5)
#   Negative: low digital literacy patients are harmed (-1.0)
#   These partially cancel giving ATE ≈ 0

# Propensity:
#   High-risk patients far away more likely to be enrolled
#   There is moderate confounding — def not as severe as Setup 1

# In simple terms, why this creates confounding: 
# Sicker patients (high X1) are both more likely to get telemedicine and more likely to have worse health outcomes anyway. 
# So if we just compare treated vs untreated patients naively, we'd see treated patients doing worse — but that's because the sicker people were selected into treatment, not because telemedicine harmed them.
# =============================================================

gen_setup_3 <- function(n,        # number of patients
                        d  = 10,  # number of covariates
                        sigma = 1 # noise level
) {
  
  # covariates: standard normal
  # X1 = chronic risk, X2 = age, X3 = distance,
  # X4 = digital literacy, X5-X10 = other characteristics
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  
  # propensity: moderate confounding
  # high-risk (X1 high) and far away (X3 high) more likely to be enrolled in telemedicine programme
  # this creates confounding — sicker farther patients would have worse outcomes anyway, independent of treatment
  e <- plogis(0.3 * X[,1] + 0.3 * X[,3] - 0.2 * X[,2])
  e <- pmin(pmax(e, 0.1), 0.9)
  W <- rbinom(n, size = 1, prob = e)
  
  # baseline outcome
  # health improvement score — driven by risk and age
  b <- -0.5 * X[,1] + 0.3 * X[,2] + 0.2 * X[,5]
  
  # True treatment effect tau(X) [the sign flip]
  
  # +ve component:
  # high risk (X1) and far away (X3) benefit from telemedicine
  # plogis(X1 + X3) smoothly increases with both scaled to max contribution of +1.5
  positive_component <- 1.5 * plogis(X[,1] + X[,3] - 1)
  
  # -ve component:
  # low digital literacy (low X4) are harmed
  # plogis(-X4) is high when X4 is low scaled to max contribution of -1.5 (symmetric with positive)
  negative_component <- -1.5 * plogis(-X[,4] - 0.5)
  
  # combined treatment effect
  # roughly 25% of patients have positive tau (high risk + far)
  # roughly 50% have negative tau (low digital literacy)
  # some overlap so ATE is near zero overall
  tau <- positive_component + negative_component
  
  # observed outcome
  Y <- b + (W - 0.5) * tau + sigma * rnorm(n)
  
  list(X = X, W = W, Y = Y, tau = tau, e = e)
}