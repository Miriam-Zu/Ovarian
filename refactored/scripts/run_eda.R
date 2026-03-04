library(ggplot2)

# source("R/data_processing.R")
# source("R/plots.R")
# source("R/statistics.R")

script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE))
source(file.path(script_dir, "../R/data_processing.R"))
source(file.path(script_dir, "../R/plots.R"))
source(file.path(script_dir, "../R/statistics.R"))

# Usage:
# Rscript scripts/run_eda.R /path/to/A_B/folder results/eda

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript scripts/run_eda.R /path/to/A_B/folder results/eda")
}
input_path <- args[1]
output_dir <- args[2]
if (!dir.exists(input_path)) {
  stop(paste("Input path does not exist:", input_path))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Load + Prepare Data
# -------------------------
message("[EDA] Loading data from: ", input_path)
data_list <- load_and_prepare_data(input_path)

sub <- data_list$sub
sub_h <- data_list$sub_h
sub_oc <- data_list$sub_oc

# this object will be used in the feature_filtering script as well.
saveRDS(sub, file.path(output_dir, "sub.Rdata"))
message("[EDA] Saving subsampled repertoire sub.Rdata to: ", output_dir)

stat_results <- list()

# -------------------------
# UNIQUE
# -------------------------
message("[EDA] performing EDA for unique clonotypes...")
uniq_h <- plot_unique(sub_h, "Unique clonotypes HD")
uniq_oc <- plot_unique(sub_oc, "Unique clonotypes OC")

ggsave(file.path(output_dir, "unique_hd.png"), uniq_h$plot)
ggsave(file.path(output_dir, "unique_oc.png"), uniq_oc$plot)

stat_results[[length(stat_results)+1]] <-
  run_t_test(uniq_h$values, uniq_oc$values, "unique_clonotypes")

# -------------------------
# GINI
# -------------------------
message("[EDA] performing EDA for Gini diversity...")
gini_h <- plot_diversity(sub_h, "gini", "Gini Diversity HD", "Gini")
gini_oc <- plot_diversity(sub_oc, "gini", "Gini Diversity OC", "Gini")

ggsave(file.path(output_dir, "gini_hd.png"), gini_h$plot)
ggsave(file.path(output_dir, "gini_oc.png"), gini_oc$plot)

stat_results[[length(stat_results)+1]] <-
  run_t_test(gini_h$values, gini_oc$values, "gini_diversity")

# -------------------------
# TRUE DIVERSITY
# -------------------------
message("[EDA] performing EDA for true diversity...")
trued_h <- plot_diversity(sub_h, "div", "True Diversity HD", "True Diversity")
trued_oc <- plot_diversity(sub_oc, "div", "True Diversity OC", "True Diversity")

ggsave(file.path(output_dir, "true_div_hd.png"), trued_h$plot)
ggsave(file.path(output_dir, "true_div_oc.png"), trued_oc$plot)

stat_results[[length(stat_results)+1]] <-
  run_t_test(trued_h$values, trued_oc$values, "true_diversity")

# -------------------------
# INVERSE SIMPSON
# -------------------------
message("[EDA] performing EDA for inverse Simpson clonotypes...")
inv_h <- plot_diversity(sub_h, "inv.simp", "Inverse Simpson HD", "Inverse Simpson")
inv_oc <- plot_diversity(sub_oc, "inv.simp", "Inverse Simpson OC", "Inverse Simpson")

ggsave(file.path(output_dir, "inv_simp_hd.png"), inv_h$plot)
ggsave(file.path(output_dir, "inv_simp_oc.png"), inv_oc$plot)

stat_results[[length(stat_results)+1]] <-
  run_t_test(inv_h$values, inv_oc$values, "inverse_simpson")

# -------------------------
# SAVE ALL STATISTICS
# -------------------------

stat_df <- do.call(rbind, stat_results)

write.csv(
  stat_df,
  file.path(output_dir, "statistical_tests_summary.csv"),
  row.names = FALSE
)
message("[EDA] Finished saving statistical tests summary.")

# -------------------------
# OVERLAP
# -------------------------

overlap_plot <- plot_overlap(sub)

htmlwidgets::saveWidget(
  overlap_plot,
  file.path(output_dir, "overlap_heatmap.html")
)

message("[EDA] Finished saving overlap heatmap.")

# -------------------------
# COMPLETION FLAG
# -------------------------
writeLines(as.character(Sys.time()), file.path(output_dir, "eda.done"))
message("[EDA] Stage complete.")