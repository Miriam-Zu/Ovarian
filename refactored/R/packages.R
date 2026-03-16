load_packages <- function(pkgs) {
  missing <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing) > 0) {
    stop("Missing packages: ", paste(missing, collapse = ", "))
  }
  
  invisible(lapply(pkgs, function(pkg) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }))
}

# List of packages used
required_packages <- c(
  "ggplot2",
  "immunarch",
  "dplyr",
  "stringr",
  "forcats",
  "reshape2",
  "purrr",
  "plotly",
  "data.table",
  "readr",
  "readxl"
)

