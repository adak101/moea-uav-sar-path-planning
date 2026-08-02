#!/usr/bin/env Rscript
library(irace)

tuning_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[
  grep("--file=", commandArgs(trailingOnly = FALSE))
]))
if (!nzchar(tuning_dir)) tuning_dir <- "tuning"

# W irace 4.x logFile (irace.log) jest plikiem RData
load(file.path(tuning_dir, "irace.log"))

cat("=== Wykonane eksperymenty ===\n")
cat(sum(!is.na(iraceResults$experiments)), "/", iraceResults$scenario$maxExperiments, "\n\n")

cat("=== Iteracje ===\n")
cat(length(iraceResults$allElites), "\n\n")

cat("=== Najlepsza konfiguracja (elite #1 ostatniej iteracji) ===\n")
best_id <- iraceResults$allElites[[length(iraceResults$allElites)]][1]
print(iraceResults$allConfigurations[best_id, ])

cat("\n=== Top elite ostatniej iteracji ===\n")
elite_ids <- iraceResults$allElites[[length(iraceResults$allElites)]]
print(iraceResults$allConfigurations[elite_ids, ])

if (!is.null(iraceResults$testing)) {
  cat("\n=== Wyniki na instancjach testowych (held-out) ===\n")
  print(iraceResults$testing$experiments)
}
