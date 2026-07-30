The `extra_galaxies` folder contains example scripts showing how to add a lower tier of galaxies to a
multi-galaxy model — companions in the field whose light must be accounted for, but which are not themselves
subjects of the measurement.

The API is identical to the single-galaxy version in `autogalaxy_workspace/*/imaging/features/extra_galaxies`.
What is new here is that the model has **two tiers at once**: the co-equal galaxies of the blend, each with a
free light model and free centre, and the extra galaxies, each with a restricted light model and a fixed centre.

Both levers from the single-galaxy example carry over:

- **Noise scaling** — the extra galaxies' data values are scaled to zero and their noise inflated, so they
  contribute negligibly to the likelihood while their pixels stay in the fit. The model composition is
  untouched.
- **Modeling** — the extra galaxies are added to the model with fixed centres, as `SersicSph` profiles or as
  MGE bases.

At multi-galaxy scale the choice tilts towards modeling more often than for a single galaxy, because the
deliverable is a *decomposition*: light the noise scaling does not quite remove gets absorbed asymmetrically by
the two free light models, biasing the flux ratio between the pair, which is usually the quantity the analysis
exists to measure.

# Files

The following example scripts illustrate multi-galaxy modeling where:

- `modeling`: Multi-galaxy modeling using a model which includes extra galaxies, showing both levers.
- `simulator`: Simulating a blended pair with two fainter companions surrounding it.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/results`.
