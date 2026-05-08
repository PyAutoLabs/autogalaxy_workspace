# PyAutoGalaxy Workspace

[![JOSS](https://joss.theoj.org/papers/10.21105/joss.02825/status.svg)](https://doi.org/10.21105/joss.02825)

[Installation Guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html) |
[readthedocs](https://pyautogalaxy.readthedocs.io/en/latest/index.html) |
[Introduction on Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.8.2/start_here.ipynb) |
[HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy)

Welcome to the **PyAutoGalaxy** Workspace!

## Getting Started

You can get set up on your personal computer by following the installation guide on
our [readthedocs](https://pyautogalaxy.readthedocs.io/).

Alternatively, you can try **PyAutoGalaxy** out in a web browser by going to
the [autogalaxy workspace on Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.8.2/start_here.ipynb).

## New Users

New users should read the `autogalaxy_workspace/start_here.ipynb` notebook, which will give you a concise
overview of **PyAutoGalaxy**'s core features and API.

This can be done via a web browser by going to the following Google Colab link:

https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.8.2/start_here.ipynb

Then checkout the [new user starting guide](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html)
to navigate the workspace for your science case.

## HowToGalaxy

If you are new to galaxy modeling or the statistical techniques it relies on, the **HowToGalaxy** lecture series
takes you from first principles through to modeling real galaxy imaging data. It now lives in its own repository:

https://github.com/PyAutoLabs/HowToGalaxy

## Workspace Structure

The workspace includes the following main directories:

- `notebooks`: **PyAutoGalaxy** examples written as Jupyter notebooks.
- `scripts`: **PyAutoGalaxy** examples written as Python scripts.
- `config`: Configuration files which customize **PyAutoGalaxy**'s behaviour.
- `dataset`: Where data is stored, including example datasets.
- `output`: Where **PyAutoGalaxy** analysis and visualization outputs are written.

The examples in the `notebooks` and `scripts` folders are structured as follows:

- `guides`: Guides which introduce the core features of **PyAutoGalaxy**, including the core galaxy modeling API.
- `imaging`: Examples for galaxy modeling using CCD imaging (e.g. Hubble, James Webb, Euclid).
- `interferometer`: Examples for galaxies observed with an interferometer (e.g. ALMA, JVLA).
- `multi`: Examples for modeling galaxies observed in multiple wavebands.

The dataset packages (e.g. `imaging`, `interferometer` and `multi`) include the following types of examples:

- `modeling`: Performing galaxy modeling using that type of data.
- `simulators`: Simulating galaxy images.
- `fit`: How to compute residuals, chi-squared maps, and likelihoods.
- `data_preparation`: Preparing real datasets for **PyAutoGalaxy** analysis.
- `features`: Advanced modeling features (e.g. Multi Gaussian Expansion, priors, constraints).
- `likelihood_function`: Step-by-step guides to the likelihood function.

The `guides` package contains important subpackages, including:

- `results`: How to load, inspect, and analyze results from many galaxy fits efficiently.
- `modeling`: Customizing galaxy models and building automated modeling pipelines.
- `plot`: How to visualize galaxy images, profiles, and residuals.

The `README.md` files distributed throughout the workspace describe what is in each folder.

## Community & Support

Support for **PyAutoGalaxy** is available via our Slack workspace, where the community shares updates, discusses
galaxy modeling and analysis, and helps troubleshoot problems.

Slack is invitation-only. If you'd like to join, please send an email requesting an invite.

For installation issues, bug reports, or feature requests, please raise an issue on the [GitHub issues page](https://github.com/Jammy2211/PyAutoGalaxy/issues).

## Contribution

To make changes in the tutorial notebooks, please make changes in the corresponding Python files (`.py`)
present in the `scripts` folder of each chapter. The marker `# %%` alternates between code cells and
markdown cells.

## Build Configuration

The `config/` directory contains two files used by the automated build and test system
(CI, smoke tests, and pre-release checks). These are not relevant to normal workspace usage.

- `config/build/no_run.yaml` — scripts to skip during automated runs. Each entry is a filename stem
  or path pattern with an inline comment explaining why it is skipped.
- `config/build/env_vars.yaml` — environment variables applied to each script during automated runs.
  Defines default values (e.g. test mode, small datasets) and per-script overrides for scripts
  that need different settings.
