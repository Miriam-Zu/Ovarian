
library("immunarch")
library("dplyr")


load_and_prepare_data <- function(input_path, min_clones = 24000, seed = 42) {
  immdata <- repLoad(input_path)

  clones <- repExplore(immdata$data, .method = "clones")
  blood_order <- arrange(clones, Clones)
  # get Sample names of samples with Clones < 24k
  small_clone_samples <- blood_order$Sample[blood_order$Clones < min_clones]
  remaining_clones <- clones$Clones[!clones$Sample %in% small_clone_samples]

  # Subsample
  message("[data_processing] Removing samples with fewer than ", min_clones, " clones: ", paste(small_clone_samples, collapse = ", "))
  message("[data_processing] Smallest clone count after removal: ", min(remaining_clones))

  # 1. Omit small samples (21_H, 49_H)
  immdata$data[which(names(immdata$data) %in% small_clone_samples)] <- NULL
  # 2. subsample
  set.seed(seed)
  message("[data_processing] Subsampling with seed: ", seed, "to ", min(remaining_clones), " clones per sample.")
  sub <- repSample(immdata$data, .method = "downsample")
  list(sub = sub,
       sub_h = sub[grepl("H", names(sub))],
       sub_oc = sub[grepl("OC", names(sub))])
}
