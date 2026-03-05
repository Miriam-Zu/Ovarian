
build_feature_matrix <- function(sub) {
  message("[feature_filtering] Building public clonotype proportion matrix...")
  public_prop <- pubRep(sub, .quant = "prop", .verbose = FALSE, .col = "aa")
  pub_prop_df <- as.data.frame(public_prop)
  pub_prop_df[is.na(pub_prop_df)] <- 0
  pub_prop_df <- subset(pub_prop_df, select = -c(Samples))

  merged_t <- setNames(
    data.frame(t(pub_prop_df[, -1])),
    pub_prop_df[, 1]
  )
  message("[feature_filtering] Feature matrix dimensions: ",
          nrow(merged_t), " samples x ", ncol(merged_t), " features")
  merged_t
}

filter_top_n <- function(merged_t, n = 600) {
  message("[feature_filtering] Selecting top ", n, " most shared features...")
  data_m <- data.matrix(merged_t)
  samples_count <- colSums(data_m != 0)
  feature_selection <- order(samples_count, decreasing = TRUE)[1:n]
  selected <- merged_t[, feature_selection]
  message("[feature_filtering] Done. Selected ", ncol(selected), " features.")
  selected
}

filter_database_features <- function(merged_t, vdjdb_path, mcpas_path, tcga_path) {
  message("[feature_filtering] Loading database files...")

  # VDJdb
  if (!file.exists(vdjdb_path)) stop(paste("VDJdb file not found:", vdjdb_path))
  vdjdb <- read_tsv(vdjdb_path, show_col_types = FALSE)
  vdjdb <- unique(vdjdb[, -c(1, 13, 14, 15, 16)])
  top_vdjdb <- names(sort(table(vdjdb$CDR3), decreasing = TRUE)[1:10])

  # McPAS
  if (!file.exists(mcpas_path)) stop(paste("McPAS file not found:", mcpas_path))
  mcpas <- read_excel(mcpas_path)
  cdr3_mcpas <- sort(
    c(table(mcpas$CDR3.alpha.aa), table(mcpas$CDR3.beta.aa)),
    decreasing = TRUE
  )[-c(1, 2)]
  top_mcpas <- names(cdr3_mcpas[1:10])

  # TCGA Trust4
  if (!file.exists(tcga_path)) stop(paste("TCGA Trust4 file not found:", tcga_path))
  tcga <- read_tsv(tcga_path,
                   col_names = c("sample", "chain", "cdr3", "subchain", "count"),
                   show_col_types = FALSE)
  trust_tcr <- subset(tcga, chain %in% c("TRA", "TRB"))
  top_tcga <- names(sort(table(trust_tcr$cdr3), decreasing = TRUE)[2:11])

  # Intersect across databases for reporting
  overlap <- Reduce(intersect, list(top_vdjdb, top_mcpas, top_tcga))
  message("[feature_filtering] Overlap across all 3 databases: ",
          length(overlap), " clonotypes: ", paste(overlap, collapse = ", "))

  # Union, deduplicate, intersect with dataset
  all_tcrs <- unique(c(top_vdjdb, top_mcpas, top_tcga))
  message("[feature_filtering] Unique TCRs from all databases: ", length(all_tcrs))

  intersection <- intersect(all_tcrs, colnames(merged_t))
  message("[feature_filtering] Features present in dataset: ",
          length(intersection), " of ", length(all_tcrs))

  merged_t[, intersection]
}