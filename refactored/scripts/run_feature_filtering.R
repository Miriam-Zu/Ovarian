script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE))
source(file.path(script_dir, "../R/feature_filtering.R"))

# Usage:
# Rscript scripts/run_feature_filtering.R results/eda results/features /path/to/db/folder method

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3 || length(args) > 4) {
  stop("Usage: Rscript scripts/run_feature_filtering.R <eda_dir> <output_dir> <db_dir> [method]
       method: 1, 2, or both (default: both)")
}

eda_dir    <- args[1]
output_dir <- args[2]
db_dir     <- args[3]
method     <- if (length(args) == 4) args[4] else "both"

if (!method %in% c("1", "2", "both")) {
  stop(paste("Invalid method:", method, "— must be 1, 2, or both"))
}
if (!dir.exists(eda_dir))    stop(paste("EDA directory not found:", eda_dir))
if (!dir.exists(db_dir))     stop(paste("Database directory not found:", db_dir))

sub_path <- file.path(eda_dir, "sub.Rdata")
if (!file.exists(sub_path))  stop(paste("sub.Rdata not found in:", eda_dir))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Load subsampled data
# -------------------------
message("[feature_filtering] Loading sub.Rdata from: ", eda_dir)
sub <- readRDS(sub_path)

# -------------------------
# Build feature matrix
# -------------------------
merged_t <- build_feature_matrix(sub)

# -------------------------
# Method 1: top 600
# -------------------------
if (method %in% c("1", "both")) {
  message("[feature_filtering] Running method 1: top-N shared clonotypes...")
  data_600 <- filter_top_n(merged_t, n = 600)
  write.csv(data_600, file.path(output_dir, "data_600_ab.csv"), row.names = TRUE)
  message("[feature_filtering] Saved data_600_ab.csv to: ", output_dir)
}

# -------------------------
# Method 2: database features
# -------------------------
if (method %in% c("2", "both")) {
  message("[feature_filtering] Running method 2: database-derived clonotypes...")
  data_16 <- filter_database_features(
    merged_t,
    vdjdb_path = file.path(db_dir, "SearchTable.tsv"),
    mcpas_path = file.path(db_dir, "mc.10.9.22.xlsx"),
    tcga_path  = file.path(db_dir, "trust4_tcga_deidentified_report.tsv")
  )
  write.csv(data_16, file.path(output_dir, "data_16_ab.csv"), row.names = TRUE)
  message("[feature_filtering] Saved data_16_ab.csv to: ", output_dir)
}

# -------------------------
# COMPLETION FLAG
# -------------------------
writeLines(as.character(Sys.time()), file.path(output_dir, "feature_filtering.done"))
message("[feature_filtering] Stage complete.")