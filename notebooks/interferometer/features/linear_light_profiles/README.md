The `linear_light_profiles` folder contains example scripts showing how to fit `Interferometer` data using
**linear light profiles**, whose `intensity` is solved for analytically via a linear inversion instead of
being a free parameter of the non-linear search.

For interferometer data this is now practical thanks to the JAX-native NUFFT `nufftax`
(https://github.com/GragasLab/nufftax), which evaluates the image-to-uv Fourier transform of every basis
component inside the same jit/vmap pipeline as the rest of the model. Older interferometer guidance that
described light-profile fitting as "slow" pre-dates this change.

# Files

- `modeling`: Galaxy modeling of an `Interferometer` dataset with a linear `Sersic` bulge and linear
  `Exponential` disk.
- `fit`: Fit a linear-light-profile galaxy model to interferometer data and inspect the solved-for
  intensities.
- `likelihood_function`: A step-by-step walkthrough of the linear-light-profile interferometer likelihood
  function (NUFFT of each linear basis image, mapping/curvature matrices, $\chi^2$ in the visibility plane).

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts, see
`autogalaxy_workspace/scripts/imaging/features/linear_light_profiles`.
