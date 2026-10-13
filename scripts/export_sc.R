#!/usr/bin/env Rscript
# Run once on Narval. Exports the real single-cell data for the Python/CUDA code.
#   Rscript scripts/export_sc.R
# Then on the same machine:  python scripts/mtx_to_npz.py data/
suppressPackageStartupMessages(library(Matrix))
DATA_DIR <- Sys.getenv("DATA_DIR", "/home/jthomo/projects/rrg-ubcxzh/SharedData/BALbenchmark")
OUT_DIR  <- Sys.getenv("OUT_DIR", "data")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
sc <- readRDS(file.path(DATA_DIR, "GEOdata/pair.sc_GEO.rds"))
writeMM(sc$cpm_matrix, file.path(OUT_DIR, "cpm.mtx"))           # genes x cells
writeLines(rownames(sc$cpm_matrix), file.path(OUT_DIR, "genes.txt"))
write.csv(sc$meta_ref[, c("SubjectName", "cell_type1", "cell_type2", "cell_type3")],
          file.path(OUT_DIR, "meta.csv"), row.names = FALSE)
cat("done\n")
