scvis_map <- function(sce,
                      output_dir,
                      sce_assay = "logcounts",
                      use_reducedDim = FALSE,
                      reducedDim_name = NULL,
                      config_file = NULL,
                      data_label_file = NULL,
                      normalize = NULL) {

  if (!is(sce, "SingleCellExperiment")) {
    stop("The parameter sce must be a SingleCellExperiment")
  }

  if (use_reducedDim) {
    if (!reducedDim_name %in% SingleCellExperiment::reducedDimNames(sce)) {
      stop(paste("ReducedDim", reducedDim_name, "is not present in the assays associated with the input SingleCellExperiment."))
    }
    counts <- reducedDim(sce, reducedDim_name)
  } else {
    if (!sce_assay %in% names(SummarizedExperiment::assays(sce))) {
      stop(paste("Assay", sce_assay, "is not present in the assays associated with the input SingleCellExperiment."))
    }
    counts <- t(SummarizedExperiment::assay(sce, sce_assay))
  }

  data_matrix_file <- tempfile(pattern = "data_matrix_file", fileext = "tsv")
  write.table(counts, file = data_matrix_file, sep = "\t")

  curr_path <- get_current_directory()

  if (!is.null(config_file) && !file.exists(config_file)) {
    stop(paste("Configuration file", config_file, "does not exist."))
  }

  if (!is.null(data_label_file) && !file.exists(data_label_file)) {
    stop(paste("Data label file", data_label_file, "does not exist."))
  }

  checkpoint <- get_checkpoint(output_dir)

  # source_script()

  package_dir <- find.package("scvis")
  script <- file.path(package_dir, "python", "lib", "scvis", "run_noplot.py")

  reticulate::source_python(script)

  map_args <- reticulate::dict(data_matrix_file = data_matrix_file,
                               config_file = NULL,
                               out_dir = output_dir,
                               data_label_file = NULL,
                               pretrained_model_file = checkpoint,
                               normalize = NULL,
                               verbose = FALSE,
                               verbose_interval = 50,
                               show_plot = FALSE,
                               curr_path = curr_path)

  map(map_args)

  reducedDim_file <- list.files(output_dir, pattern = "*_map.tsv", full.names = TRUE)

  if (length(reducedDim_file) > 1) {
    stop("Multiple reduced dimension output files in the output directory.")
  }

  z_coordinates <- read.table(file = reducedDim_file, sep = "\t")
  # FIXME
  # SingleCellExperiment::reducedDim(sce, "scvis") <- z_coordinates

  sce
}
