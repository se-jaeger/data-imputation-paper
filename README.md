# Source Code for the Paper: Benchmark for Data Imputation Methods

Check out the final paper at: https://www.frontiersin.org/articles/10.3389/fdata.2021.693674/full

## Disclaimer

This is research code and in no way ready for production usage!


## Citing us!

If you want to reference our paper or this code, please use the following BibTex:

```
@ARTICLE{imputation_benchmark_jaeger_2021,
	AUTHOR={Jäger, Sebastian and Allhorn, Arndt and Bießmann, Felix},
	TITLE={A Benchmark for Data Imputation Methods},
	JOURNAL={Frontiers in Big Data},
	VOLUME={4},
	PAGES={48},
	YEAR={2021},
	URL={https://www.frontiersin.org/article/10.3389/fdata.2021.693674},
	DOI={10.3389/fdata.2021.693674},
	ISSN={2624-909X},
	ABSTRACT={With the increasing importance and complexity of data pipelines, data quality became one of the key challenges in modern software applications. The importance of data quality has been recognized beyond the field of data engineering and database management systems (DBMSs). Also, for machine learning (ML) applications, high data quality standards are crucial to ensure robust predictive performance and responsible usage of automated decision making. One of the most frequent data quality problems is missing values. Incomplete datasets can break data pipelines and can have a devastating impact on downstream ML applications when not detected. While statisticians and, more recently, ML researchers have introduced a variety of approaches to impute missing values, comprehensive benchmarks comparing classical and modern imputation approaches under fair and realistic conditions are underrepresented. Here, we aim to fill this gap. We conduct a comprehensive suite of experiments on a large number of datasets with heterogeneous data and realistic missingness conditions, comparing both novel deep learning approaches and classical ML imputation methods when either only test or train and test data are affected by missing data. Each imputation method is evaluated regarding the imputation quality and the impact imputation has on a downstream ML task. Our results provide valuable insights into the performance of a variety of imputation methods under realistic conditions. We hope that our results help researchers and engineers to guide their data preprocessing method selection for automated data quality improvement.}
}
```


### Installation

In order to set up the necessary environment:

1. create an environment `data-imputation-paper` with the help of [conda],
   ```bash
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```bash
   conda activate data-imputation-paper
   ```
3. install `jenga` with
   ```bash
   cd src/jenga
   python setup.py develop
   ```
4. install `data-imputation-paper` with:
   ```bash
   cd ../..
   python setup.py develop # or `install`
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

Then take a look into the `scripts` and `notebooks` folders.


### Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n data-imputation-paper -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
### Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── data-imputation-paper      <- Actual Python package where the main functionality goes.
│   └── jenga               <- Jenga code, used to add data corruptions/missingness.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

### Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
