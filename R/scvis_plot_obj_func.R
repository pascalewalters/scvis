scvis_plot_obj_func <- function(output_dir) {
  # Get objective function data .tsv
  obj_file <- list.files(output_dir, pattern = "*iter_[0-9]+_obj.tsv$", full.names = TRUE)

  if (length(obj_file) == 0) {
    stop("No objective function output file found in the output directory.")
  } else if (length(obj_file) > 1) {
    stop("Multiple objective function output files in the output directory.")
  }

  obj <- read.table(file = obj_file, sep = "\t", header = TRUE, row.names = 1)

  elbo <- obj$elbo
  tsne_cost <- obj$tsne_cost

  avg_elbo <- elbo - tsne_cost

  for(i in 2:length(elbo)) {
    avg_elbo[i] = (elbo[i] - tsne_cost[i]) / i + avg_elbo[i - 1] * (i - 1) / i
  }

  obj$avg_elbo <- avg_elbo
  obj$iteration <- 1:length(elbo)

  obj

}
