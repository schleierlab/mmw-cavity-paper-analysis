# Analysis software for _Optically accessible high-finesse millimeter-wave resonator for cavity quantum electrodynamics with atom arrays_

This repository contains analysis code for the paper entitled
_Optically accessible high-finesse millimeter-wave resonator for cavity quantum electrodynamics with atom arrays_
([arXiv:2506.05804](https://arxiv.org/abs/2506.05804)).
The code herein generates all the figures shown in the main text and supplementary material,
and additionally shows how to analyze the raw data.
We have tested this software with Python 3.12.

Please direct any questions to <txz@stanford.edu>


## Project status

Maintenance only.


## Quickstart

- Decide on a working environment. Choose from one of the following:
    1. Use some base python environment (highly discouraged)
    2. Create your own environment
    3. For maximum reproducibility.
        - Install `conda` (miniconda is okay) if you don't already have it.
        - Create an analysis environment closely approximating the one used for
          the analysis in the paper (see below for departures)
          ```shell
          conda env create -f environment.yml
          ```
        - Activate the environment:
          ```shell
          conda activate mmw-cavity-paper-analysis
          ```
- Clone and install the [suprtools repository, v0.1.0](https://doi.org/10.5281/zenodo.16923078) inside whatever environment you chose.
  This repository contains general analysis code.
- Download the files from Zenodo [[10.5281/zenodo.16907261](https://doi.org/10.5281/zenodo.16907261)]
  into a `data` directory at the same level as this README, and extract the `.zip` archive,
  producing a second data directory:
  ```
  mmw-cavity-paper-analysis/
  |-- analysis.ipynb
  |-- data/
  |    |-- README.md
  |    |-- data.zip
  |    +-- data/
  |         +-- ...
  |-- img/
  |    +-- ...
  |-- intermediate-data/
  |    +-- ...
  |-- paper_figures/
  |    +-- ...
  +-- README.md
  ```
- Now we can actually get to running the notebook.
    - If you have no local LaTeX installation, make the following modification
      in `analysis.ipynb` before runnning so that matplotlib does not try
      to use LaTeX for rendering figure text:
        ```python
        setup_paper_style(usetex=False)  # previously, usetex=True
        ```
    - Now run`analysis.ipynb`. This can take up to 10 minutes.

## Contents

- `analysis.ipynb`: main analysis notebook
- `img`: images used in the notebook
- `intermediate-data`: fit results that (used to) take some time to generate.
    - Presently these are in the form of `.csv` files that are already provided.
    - The `.csv` files may be regenerated from raw data in the `data` directory
      by running the entireties of the three analysis notebooks:
        - `caecilia-renalaysis.ipynb` (fast)
        - `cassia-reanalysis.ipynb` (>5 min)
        - `flaminia-reanalysis.ipynb` (>5 min)
    - The resulting `.csv` files may not be identical to those provided here,
      but will be numerically quite close.
- `paper_figures`: additional code that is imported in the main notebook (`analysis.ipynb`)
- `README.md`: this README

## Environment file

The environment file was exported as
```python
conda env export -f environment.yaml --no-builds --format=environment-yaml
```
with environment variables and irrelevant pip packages removed,
and with the following bumps:
- scikit-rf: 1.4.0 to 1.5.0 (in case of numpy install issues requiring numpy 2,
  this version bump prevents [this issue](https://github.com/scikit-rf/scikit-rf/issues/1199))
- libgfortran5: 13.2.0 to >=14.0.0 (to prevent [numpy install issues](https://github.com/conda-forge/numpy-feedstock/issues/347#issuecomment-2772248255))
