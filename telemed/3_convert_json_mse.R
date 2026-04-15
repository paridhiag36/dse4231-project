# ============================================================
# convert_json_mse.R
# Adds raw_mse to existing simulation JSON files.
#
# The original files only contain norm_mse per learner per
# iteration. This script back-calculates:
#   raw_mse = norm_mse * tau_sd^2
# using the tau_sd stored in each iteration, and writes the
# updated JSON in place.
#
# Run ONCE after switching to raw MSE as the primary metric.
# ============================================================

library(jsonlite)

convert_json <- function(path) {
  cat("Converting:", path, "\n")
  raw <- fromJSON(path, simplifyVector = FALSE)

  n_iter <- length(raw$iterations)
  cat("  Iterations found:", n_iter, "\n")

  for (i in seq_along(raw$iterations)) {
    it <- raw$iterations[[i]]

    # Skip skipped iterations
    if (isTRUE(it$skipped[[1]])) next

    tau_sd  <- as.numeric(it$tau_sd[[1]])
    tau_var <- tau_sd^2

    for (lname in names(it$results)) {
      nm_raw <- it$results[[lname]]$norm_mse
      if (!is.null(nm_raw)) {
        nm <- as.numeric(nm_raw[[1]])
        raw$iterations[[i]]$results[[lname]]$raw_mse <- nm * tau_var
      } else {
        raw$iterations[[i]]$results[[lname]]$raw_mse <- NULL
      }
    }
  }

  write_json(raw, path, auto_unbox = TRUE, digits = 6)
  cat("  Done.\n")
}

# ============================================================
# Run conversion on both simulation files
# ============================================================

setwd(dirname(rstudioapi::getSourceEditorContext()$path))  # set to script dir

convert_json("telemed_simulation_100.json")
convert_json("telemed_simulation_200.json")

cat("\nConversion complete. Re-run json_analysis.R to regenerate the CSV.\n")
