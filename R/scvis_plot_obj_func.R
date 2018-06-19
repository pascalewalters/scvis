scvis_plot_obj_func <- function(output_dir) {
  # Get objective function data .tsv
  obj_file <- list.files(output_dir, pattern = "*iter_[0-9]+_obj.tsv$", full.names = TRUE)

  if (length(obj_file) > 1) {
    stop("Multiple objective function output files in the output directory.")
  }

  obj <- read.table(file = obj_file, sep = "\t", header = TRUE, row.names = 1)

  elbo <- obj$elbo
  tsne_cost <- obj$tsne_cost

  avg_elbo <- elbo - tsne_cost

  for(i in 2:length(elbo)) {
    avg_elbo[i] = (elbo[i] - tsne_cost[i]) / i + avg_elbo[i - 1] * (i - 1) / i
  }

  iteration <- 1:length(elbo)

  p1 <- ggplot(df, aes(iteration, elbo)) + geom_line()
  p2 <- ggplot(df, aes(iteration, tsne_cost)) + geom_line()
  p3 <- ggplot(df, aes(iteration, avg_elbo)) + geom_line()

  # gridExtra::grid.arrange(p1, p2, p3)
  # FIXME
  cowplot::plot_grid(p1, p2, p3)

}
