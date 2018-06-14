#' Train a probabilistic parametric mapping
#'
#' @param sce SingleCellExperiment object
#' @param output_dir Path to the directory for outputs
#' @param sce_assay The assay of \code{sce} used to obtain the expression values for the calculations
#' @param use_reducedDim Whether reducedDim should be used instead of an assay
#' @param reducedDim_name The name of the reducedDim that should be used when \code{use_reducedDim = TRUE}
#' @param config_file Path to the configuration file (default: scvis/inst/python/lib/scvis/config/model_config.yaml)
#' @param data_label_file Path to a one column file (with column header) that provides the corresponding cluster information for each data point, just used for coloring scatter plots (optional)
#' @param normalize Positive float number for normalization of expression values (default: maximum expression value)
#' @export

scvis_train <- function(sce,
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

  if (!is.null(config_file) && !file.exists(config_file)) {
    stop(paste("Configuration file", config_file, "does not exist."))
  }

  if (!is.null(data_label_file) && !file.exists(data_label_file)) {
    stop(paste("Data label file", data_label_file, "does not exist."))
  }

  package_dir <- find.package("scvis")
  script <- file.path(package_dir, "python", "run.py")

  if (is.null(config_file)) {
    config_file <- file.path(package_dir, "python", "config", "model_config.yaml")
  }

  reticulate::source_python(script)

  train_args <- reticulate::dict(data_matrix_file = data_matrix_file,
                                 config_file = config_file,
                                 out_dir = output_dir,
                                 data_label_file = data_label_file,
                                 pretrained_model_file = NULL,
                                 normalize = normalize)

  train(train_args)

  reducedDim_file <- list.files(output_dir, pattern = "*iter_[0-9]+.tsv$", full.names = TRUE)

  if (length(reducedDim_file) > 1) {
    stop("Multiple reduced dimension output files in the output directory.")
  }

  z_coordinates <- as.matrix(read.table(file = reducedDim_file, sep = "\t", header = TRUE, row.names = 1))
  SingleCellExperiment::reducedDim(sce, "scvis") <- z_coordinates

  sce
}
