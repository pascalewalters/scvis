install_matplotlib <- function(method = "auto", conda = "auto") {
  reticulate::py_install("matplotlib", method = method, conda = conda)
}

install_pyyaml <- function(method = "auto", conda = "auto") {
  reticulate::py_install("pyyaml", method = method, conda = conda)
}

install_pandas <- function(method = "auto", conda = "auto") {
  reticulate::py_install("pandas", method = method, conda = conda)
}

install_numpy <- function(method = "auto", conda = "auto") {
  reticulate::py_install("numpy", method = method, conda = conda)
}
