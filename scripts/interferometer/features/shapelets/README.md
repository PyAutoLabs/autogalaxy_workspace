The `shapelets` folder contains example scripts showing how to fit `Interferometer` data using a
**shapelet decomposition** of the galaxy's light: a polar (Gauss-Hermite) basis whose `intensity` values
are solved for analytically via a linear inversion.

Shapelet fits to interferometer data were previously impractical because every iteration has to NUFFT
each basis component. With nufftax (https://github.com/GragasLab/nufftax) the full shapelet basis is
transformed inside the same jit/vmap pipeline as the rest of the model, so shapelets-on-visibilities is
now practical at any visibility count.

# Files

- `modeling`: Galaxy modeling of an `Interferometer` dataset with a polar shapelet bulge.
- `fit`: Fit a known-parameter shapelet galaxy and inspect the per-shapelet solved-for `intensity` values.

# Positive-Negative Solver

Unlike MGE or linear-light-profile-Sersic fits, shapelets **require** the positive-negative linear
algebra solver. Shapelets are a mathematical basis (not physically motivated profiles), and their ability
to decompose galaxy morphology depends on being able to mix positive and negative basis-function
amplitudes. The `Settings(use_positive_only_solver=False)` toggle is therefore set in the analysis.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts, see `autogalaxy_workspace/scripts/imaging/features/shapelets`.
