#' Get a dataframe of objective function results for each iteration of scvis
#'
#' @param obj_file Path to the objective function results file after running scvis_train or scvis_map
#' @export

scvis_get_obj_func_results <- function(obj_file) {
  # Get objective function data .tsv
  if (!is.null(obj_file) && !file.exists(obj_file)) {
    stop(paste("Objective function results file", obj_file, "does not exist."))
  } 

  if (endsWith(obj_file, "tsv")) {
    obj <- read.table(file = obj_file, sep = "\t", header = TRUE, row.names = 1)
  } else {
    stop("Please make sure the file is in tsv format.")
  }

  obj <- read.table(file = obj_file, sep = "\t", header = TRUE, row.names = 1)

  if (is.null(obj$elbo) && is.null(obj$tsne_cost)) {
    stop("Please make sure the file has elbo and tsne_cost columns")
  }

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
