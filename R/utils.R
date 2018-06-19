get_checkpoint <- function(output_dir) {
  checkpoint_file <- file.path(output_dir, "model", "checkpoint")
  if (!file.exists(checkpoint_file)) {
    stop("Checkpoint file does not exist. Ensure that scvis_train has run and the output directory is correct.")
  }

  yaml::read_yaml(checkpoint_file)$model_checkpoint_path
}

source_script <- function() {
  package_dir <- find.package("scvis")
  script <- file.path(package_dir, "python", "run.py")

  reticulate::source_python(script)
}

