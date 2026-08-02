#!/usr/bin/env Rscript
library(irace)

tuning_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[
  grep("--file=", commandArgs(trailingOnly = FALSE))
]))
if (!nzchar(tuning_dir)) tuning_dir <- "tuning"
setwd(tuning_dir)

# irace_cmdline czyta scenario.txt bezpośrednio, bez pośrednich kroków
irace_cmdline("--scenario scenario.txt")
