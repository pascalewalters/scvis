# scvis

scvis is a wrapper for the scvis python package for dimension reduction of high-dimensional biological data, especially single-cell RNA-sequencing (scRNA-seq) data. It is adapted from https://bitbucket.org/jerry00/scvis-dev/.


## Dependencies:

  * tensorflow >= 1.1
  * PyYAML >= 3.11
  * matplotlib >= 1.5.1
  * numpy >= 1.11.1
  * pandas >= 0.19.1


## Running `scvis`

### 1, the `train` function

The `train` function can be used to learn a probabilistic parametric mapping (the exact directories of the input files should change based on their actual positions in the computer system):

```R
scvis_train(sce,
			output_dir,
			sce_assay = "logcounts",
			use_reducedDim = FALSE,
			reducedDim_name = NULL,
			config_file = NULL,
			data_label_file = NULL,
			normalize = NULL)
```

A trained model is saved in the folder `<output_dir>/model/`

In addition to the model file, the low-dimensional embedding and the log-likelihoods are also written to two files in `<output_dir>`,
(the log-likelihood file is named as `*_log_likelihood.tsv`).

The different components of the objective function are also saved to a file (`*_obj.tsv`).

By default, the desired data from `sce` is normalized by the maximum absolute value. If you want to provide a positive float number for normalization, you can set `normalize = your_number`.

Another important parameter is `config_file`, which allows you to set various parameters. If you want to use your own config file, you can pass it as a parameter. The default config file is in `scvis/inst/python/config/model_config.yaml`,  and you can use this file as a template to set parameters.  


### 2, the `map` function
After learning a probabilistic parametric mapping, the `map` function can be used to add new data to an existing embedding:

```R
scvis_map(sce,
			output_dir,
			sce_assay = "logcounts",
			use_reducedDim = FALSE,
			reducedDim_name = NULL,
			config_file = NULL,
			data_label_file = NULL,
			normalize = NULL)
```

As for calling the `scvis_train()` command, this command will also output the likelihood files and the low-dimensional embedding files, but without the model files and the objective function trace file and plots.

The data matrix files for calling both `train` and `map` should be normalized similarly, i.e., the parameters used to normalize the training data should be used to normalize the test data. This is the default setting. You can also pass a positive float number to normalize your data: `normalize = your_number`. 

For `map`, you can also pass the config file as a parameter: `config_file`. Notice that the `config_file` for `scvis_train()` and `scvis_map()` should be the same. 
