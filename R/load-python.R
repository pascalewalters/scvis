.onLoad <- function(libname, pkgname) {
  package_dir <- find.package("scvis")
  script <- file.path(package_dir, "python", "lib", "scvis", "run.py")

  reticulate::source_python(script)
}
