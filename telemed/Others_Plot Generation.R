library(ggplot2)
library(dplyr)
library(tidyr)
library(jsonlite)
library(scales)

# ============================================================
# LOAD DATA
# ============================================================

dgp_diag   <- read.csv("dgp_iter_diagnostics.csv")
dgp_summ   <- read.csv("telemed_dgp_summary.csv")
metrics    <- read.csv("telemed_metrics_long.csv")
table_data <- read.csv("telemed_table_ready.csv")
sim        <- fromJSON("telemed_simulation_200.json")

# ============================================================
# PLOT 6: SIGN RECOVERY RATES — SG1 vs SG2 (dot plot)
# ============================================================

learners <- names(sim$summary_by_learner)

sign_df <- do.call(rbind, lapply(learners, function(l) {
  d <- sim$summary_by_learner[[l]]
  data.frame(
    learner  = l,
    sg1_rate = d$sign_sg1_rate,
    sg2_rate = d$sign_sg2_rate,
    stringsAsFactors = FALSE
  )
}))

sign_df <- sign_df[sign_df$learner != "zero_pred", ]

sign_df <- sign_df %>%
  mutate(
    base_method = case_when(
      grepl("kern",  learner) ~ "Kernel",
      grepl("lasso", learner) ~ "Lasso",
      grepl("boost", learner) ~ "Boosting",
      learner == "ols_inter"  ~ "OLS"
    ),
    meta_type = case_when(
      grepl("^r", learner)   ~ "R-learner",
      grepl("^s", learner)   ~ "S-learner",
      grepl("^t", learner)   ~ "T-learner",
      grepl("^x", learner)   ~ "X-learner",
      learner == "ols_inter" ~ "OLS"
    )
  )

sign_long <- sign_df %>%
  pivot_longer(
    cols      = c(sg1_rate, sg2_rate),
    names_to  = "subgroup",
    values_to = "sign_rate"
  ) %>%
  mutate(
    subgroup = recode(subgroup,
                      sg1_rate = "SG1: Remote, digitally engaged\n(beneficial subgroup)",
                      sg2_rate = "SG2: Poorly controlled, multimorbid\n(harmful subgroup)"
    ),
    learner = factor(learner,
                     levels = sign_df %>% arrange(sg2_rate) %>% pull(learner)
    )
  )

meta_colours <- c(
  "R-learner" = "#2166AC",
  "S-learner" = "#4DAC26",
  "T-learner" = "#D01C8B",
  "X-learner" = "#E66101",
  "OLS"       = "#636363"
)

base_shapes <- c(
  "Kernel"   = 16,
  "Lasso"    = 17,
  "Boosting" = 15,
  "OLS"      = 18
)

p_sign <- ggplot(sign_long,
                 aes(x = sign_rate, y = learner,
                     colour = meta_type, shape = base_method)) +
  geom_vline(xintercept = 0.5, linetype = "dashed",
             colour = "grey60", linewidth = 0.4) +
  geom_point(size = 3.2, alpha = 0.9) +
  facet_wrap(~ subgroup, ncol = 2) +
  scale_x_continuous(
    limits = c(0.4, 1.02),
    breaks = seq(0.4, 1.0, 0.1),
    labels = percent_format(accuracy = 1)
  ) +
  scale_colour_manual(values = meta_colours, name = "Meta-learner") +
  scale_shape_manual(values = base_shapes,   name = "Base method") +
  labs(
    x       = "Sign recovery rate",
    y       = NULL,
    title   = "Sign Recovery Rate by Subgroup Across 184 Iterations",
    caption = "Dashed line at 50% (random chance). Learners ordered by SG2 sign recovery rate."
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
    panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.3),
    panel.grid.minor   = element_blank(),
    strip.background   = element_rect(fill = "grey95", colour = "grey70"),
    strip.text         = element_text(size = 9, face = "bold"),
    legend.position    = "bottom",
    legend.box         = "horizontal",
    plot.title         = element_text(size = 11, face = "bold", hjust = 0),
    plot.caption       = element_text(size = 8, colour = "grey50", hjust = 0),
    axis.text.y        = element_text(size = 9),
    axis.text.x        = element_text(size = 9)
  )

ggsave("sign_recovery_plot.png", p_sign,
       width = 8, height = 5.5, units = "in", dpi = 300)

cat("Sign recovery plot saved.\n")