The `multi_gaussian_expansion` folder contains example scripts showing how to fit `Interferometer` data
using a **multi-Gaussian expansion (MGE)** — a decomposition of a galaxy's light into many linear Gaussian
components whose `intensity` values are solved for analytically via a linear inversion.

MGE fits to interferometer data were previously impractical because every likelihood evaluation has to
NUFFT each Gaussian in the basis, and prior NUFFT backends were not JAX-friendly. With nufftax
(https://github.com/GragasLab/nufftax) the full MGE basis is transformed inside the same jit/vmap pipeline
as the rest of the model, so MGE fits to visibilities are now routine even at ALMA-class visibility
counts.

# Files

- `modeling`: Galaxy modeling of an `Interferometer` dataset with an MGE bulge built from 30 linear
  `Gaussian` profiles arranged in two groups of 15 (each group shares a centre and ell_comps; sigmas are
  fixed to log-spaced values).
- `fit`: Fit a known-parameter MGE galaxy and inspect the per-Gaussian solved-for `intensity` values.
- `likelihood_function`: Step-by-step walkthrough of the visibility-plane MGE linear inversion — each
  Gaussian is one column of the real-space mapping matrix, NUFFT'd to the uv-plane, then the standard
  `D`/`F` solve over the joint complex visibility data.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts, see
`autogalaxy_workspace/scripts/imaging/features/multi_gaussian_expansion`.
