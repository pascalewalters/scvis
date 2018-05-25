get_current_directory <- function() {
  package_dir <- find.package("scvis")
  python_dir <- file.path(package_dir, "python")
  file.path(python_dir, "lib", "scvis")
}

get_checkpoint <- function(output_dir) {
  checkpoint_file <- file.path(output_dir, "model", "checkpoint")
  yaml::read_yaml(checkpoint_file)$model_checkpoint_path
}

source_script <- function() {
  package_dir <- find.package("scvis")
  script <- file.path(package_dir, "python", "lib", "scvis", "run_noplot.py")

  reticulate::source_python(script)
}
