# Load required libraries
library(jsonlite)
library(tidyverse)

# Read the JSON file
json_data <- fromJSON("preventive_200results.json", simplifyVector = FALSE)

# ============================================================
# 1. SUMMARY TABLE (Main results across all learners)
# Using MEDIAN for RMSE (converted from MSE)
# ============================================================

# First, extract detailed RMSE values per iteration for each learner
rmse_values_list <- list()

for (learner_name in names(json_data$summary_by_learner)) {
  rmse_values <- c()
  for (i in 1:length(json_data$iterations)) {
    iter_result <- json_data$iterations[[i]]$results[[learner_name]]
    if (!is.null(iter_result) && !iter_result$failed) {
      # Convert MSE to RMSE
      rmse_values <- c(rmse_values, sqrt(iter_result$mse))
    }
  }
  rmse_values_list[[learner_name]] <- rmse_values
}

summary_df <- do.call(rbind, lapply(names(json_data$summary_by_learner), function(learner_name) {
  s <- json_data$summary_by_learner[[learner_name]]
  
  # Calculate median RMSE from raw iteration values
  median_rmse <- median(rmse_values_list[[learner_name]], na.rm = TRUE)
  
  # Calculate mean RMSE for comparison
  mean_rmse <- mean(rmse_values_list[[learner_name]], na.rm = TRUE)
  sd_rmse <- sd(rmse_values_list[[learner_name]], na.rm = TRUE)
  
  data.frame(
    learner = learner_name,
    n_valid = s$n_valid,
    n_failed = s$n_failed,
    iqr_mean = s$iqr$mean,
    iqr_sd = s$iqr$sd,
    iqr_ci_lo = s$iqr$ci_lo,
    iqr_ci_hi = s$iqr$ci_hi,
    spread_sd_mean = s$spread_sd$mean,
    bias_mean = s$bias$mean,
    bias_sd = s$bias$sd,
    mse_median = median(s$mse$mean, na.rm = TRUE),  # Keep for reference
    rmse_median = median_rmse,
    rmse_mean = mean_rmse,
    rmse_sd = sd_rmse,
    pass_cal_rate = s$pass_cal_rate,
    fp_sg1_rate = s$fp_sg1_rate,
    fp_sg2_rate = s$fp_sg2_rate,
    fp_sg3_rate = s$fp_sg3_rate,
    stringsAsFactors = FALSE
  )
}))

# Write to CSV
write.csv(summary_df, "preventive_200results_summary_median_rmse.csv", row.names = FALSE)

# ============================================================
# 2. DETAILED PER-ITERATION RESULTS 
# ============================================================

iterations_list <- json_data$iterations

extract_iteration <- function(iter_data, iter_num) {
  results_list <- iter_data$results
  
  do.call(rbind, lapply(names(results_list), function(learner_name) {
    r <- results_list[[learner_name]]
    
    data.frame(
      iteration = iter_num,
      seed = iter_data$seed,
      skipped = iter_data$skipped,
      true_tau = iter_data$true_tau,
      prop_treated = iter_data$prop_treated,
      learner = learner_name,
      failed = r$failed,
      bias = r$bias,
      iqr = r$iqr,
      spread_sd = r$spread_sd,
      range_90 = r$range_90,
      mse = r$mse,
      rmse = sqrt(r$mse),  # Add RMSE column
      pass_cal = r$pass_cal,
      est_sg1 = r$est_sg1,
      fp_sg1 = r$fp_sg1,
      est_sg2 = r$est_sg2,
      fp_sg2 = r$fp_sg2,
      est_sg3 = r$est_sg3,
      fp_sg3 = r$fp_sg3,
      stringsAsFactors = FALSE
    )
  }))
}

detailed_df <- do.call(rbind, lapply(1:length(iterations_list), function(i) {
  extract_iteration(iterations_list[[i]], i)
}))

write.csv(detailed_df, "preventive_200results_detailed_rmse.csv", row.names = FALSE)

# ============================================================
# 3. LEARNER COMPARISON TABLE (using RMSE)
# ============================================================

comparison_table <- summary_df %>%
  select(learner, iqr_mean, iqr_ci_lo, iqr_ci_hi, 
         bias_mean, rmse_median, pass_cal_rate, 
         fp_sg1_rate, fp_sg2_rate, fp_sg3_rate) %>%
  mutate(
    iqr_display = sprintf("%.3f [%.3f, %.3f]", iqr_mean, iqr_ci_lo, iqr_ci_hi),
    bias_display = sprintf("%+.4f", bias_mean),
    rmse_median_display = sprintf("%.4f", rmse_median),
    pass_cal_display = sprintf("%.0f%%", pass_cal_rate * 100),
    fp_sg1_display = sprintf("%.0f%%", fp_sg1_rate * 100),
    fp_sg2_display = sprintf("%.0f%%", fp_sg2_rate * 100),
    fp_sg3_display = sprintf("%.0f%%", fp_sg3_rate * 100)
  ) %>%
  select(learner, iqr_display, bias_display, rmse_median_display, 
         pass_cal_display, fp_sg1_display, fp_sg2_display, fp_sg3_display)

write.csv(comparison_table, "preventive_200results_comparison_median_rmse.csv", row.names = FALSE)

# ============================================================
# 4. SUBSET FOR KEY LEARNERS 
# ============================================================

key_learners <- c("rlasso", "rboost", "rkern", 
                  "slasso", "sboost", "skern",
                  "tlasso", "tboost", "tkern",
                  "xlasso", "xboost", "xkern",
                  "const_pred", "zero_pred", "ols_inter")

key_summary <- summary_df %>%
  filter(learner %in% key_learners) %>%
  arrange(match(learner, key_learners)) %>%
  select(learner, iqr_mean, bias_mean, rmse_median, pass_cal_rate, 
         fp_sg1_rate, fp_sg2_rate, fp_sg3_rate)

write.csv(key_summary, "preventive_200results_key_learners_median_rmse.csv", row.names = FALSE)

# ============================================================
# 5. SUMMARY STATISTICS BY LEARNER TYPE 
# ============================================================

learner_type_summary <- summary_df %>%
  mutate(
    type = case_when(
      grepl("^r", learner) ~ "R-learner",
      grepl("^s", learner) ~ "S-learner", 
      grepl("^t", learner) ~ "T-learner",
      grepl("^x", learner) ~ "X-learner",
      TRUE ~ "Baseline"
    ),
    base = case_when(
      grepl("lasso$", learner) ~ "lasso",
      grepl("boost$", learner) ~ "boosting",
      grepl("kern$", learner) ~ "kernel",
      TRUE ~ "other"
    )
  ) %>%
  group_by(type, base) %>%
  summarise(
    n_learners = n(),
    iqr_mean = mean(iqr_mean),
    pass_rate_mean = mean(pass_cal_rate),
    fp_sg1_mean = mean(fp_sg1_rate),
    rmse_median = median(rmse_median),
    .groups = "drop"
  )

write.csv(learner_type_summary, "preventive_200results_by_type_median_rmse.csv", row.names = FALSE)

# ============================================================
# 6. MEDIAN RMSE RANKING TABLE
# ============================================================

rmse_ranking <- summary_df %>%
  filter(!learner %in% c("const_pred", "zero_pred")) %>%
  select(learner, rmse_median, rmse_mean, rmse_sd, iqr_mean, pass_cal_rate) %>%
  arrange(rmse_median) %>%
  mutate(
    rmse_rank = row_number(),
    rmse_median_display = sprintf("%.4f", rmse_median),
    rmse_mean_display = sprintf("%.4f", rmse_mean),
    rmse_cv_display = sprintf("%.3f", rmse_sd / rmse_median)  # Coefficient of variation
  )

write.csv(rmse_ranking, "preventive_200results_rmse_ranking_median.csv", row.names = FALSE)

# ============================================================
# 7. PRINT SUMMARY TO CONSOLE
# ============================================================

cat("\n===========================================\n")
cat("SMS CALIBRATION STUDY - RESULTS SUMMARY\n")
cat("(Using MEDIAN RMSE for robust estimation)\n")
cat("===========================================\n\n")

cat("Configuration:\n")
cat("  - Iterations:", json_data$simulation_config$n_completed, "\n")
cat("  - True tau: 0.01 (constant)\n")
cat("  - Calibration pass threshold: IQR < 0.20\n")
cat("  - False positive threshold: |est - 0.01| > 0.10\n\n")

cat("TOP 5 BEST CALIBRATED LEARNERS (lowest IQR):\n")
top_calibrated <- summary_df %>%
  filter(!learner %in% c("const_pred", "zero_pred")) %>%
  arrange(iqr_mean) %>%
  head(5)
for (i in 1:nrow(top_calibrated)) {
  cat(sprintf("  %d. %-12s: IQR = %.4f, Pass Rate = %.0f%%, Median RMSE = %.4f\n", 
              i, top_calibrated$learner[i], 
              top_calibrated$iqr_mean[i],
              top_calibrated$pass_cal_rate[i] * 100,
              top_calibrated$rmse_median[i]))
}

cat("\nTOP 5 LOWEST MEDIAN RMSE (most accurate):\n")
lowest_rmse <- summary_df %>%
  filter(!learner %in% c("const_pred", "zero_pred")) %>%
  arrange(rmse_median) %>%
  head(5)
for (i in 1:nrow(lowest_rmse)) {
  cat(sprintf("  %d. %-12s: Median RMSE = %.4f, IQR = %.4f\n", 
              i, lowest_rmse$learner[i], 
              lowest_rmse$rmse_median[i],
              lowest_rmse$iqr_mean[i]))
}

cat("\nCOMPARISON: MEAN vs MEDIAN RMSE (largest differences):\n")
diff_df <- summary_df %>%
  filter(!learner %in% c("const_pred", "zero_pred")) %>%
  mutate(diff = rmse_median - rmse_mean) %>%
  arrange(desc(abs(diff))) %>%
  head(8)
for (i in 1:nrow(diff_df)) {
  cat(sprintf("  %-12s: Mean = %.4f, Median = %.4f, Diff = %+.4f\n", 
              diff_df$learner[i], 
              diff_df$rmse_mean[i],
              diff_df$rmse_median[i],
              diff_df$diff[i]))
}

cat("\nFiles saved (using MEDIAN RMSE):\n")
cat("  1. preventive_200results_summary_median_rmse.csv - Main summary with median RMSE\n")
cat("  2. preventive_200results_detailed_rmse.csv - All iterations with RMSE column\n")
cat("  3. preventive_200results_comparison_median_rmse.csv - Formatted comparison table\n")
cat("  4. preventive_200results_key_learners_median_rmse.csv - Subset of key learners\n")
cat("  5. preventive_200results_by_type_median_rmse.csv - Aggregated by learner type\n")
cat("  6. preventive_200results_rmse_ranking_median.csv - Ranking by median RMSE\n")