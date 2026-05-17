The `extra_galaxies` folder contains example scripts showing how to fit `Interferometer` data containing
multiple galaxies in the same field of view — a main galaxy plus one or more nearby "extra" galaxies
(line-of-sight companions, group members, or resolved neighbours).

Unlike the lensing use case in `autolens_workspace/scripts/interferometer/features/extra_galaxies/` (where
extras contribute *mass* to the ray-tracing), autogalaxy fits the *light* of each galaxy in the field. The
main + extra galaxies all carry light profiles; there is no mass model anywhere.

Multi-galaxy interferometer fits were previously impractical because every iteration has to NUFFT every
galaxy's light profile into the uv-plane separately, and prior NUFFT backends were not JAX-friendly. With
nufftax (https://github.com/GragasLab/nufftax) the full set of light profiles is transformed inside the
same jit/vmap pipeline as the rest of the model, so multi-galaxy fits to visibility data are now routine
at any visibility count.

# Files

- `modeling`: Galaxy modeling of an `Interferometer` dataset with a main galaxy + 2 extra galaxies, each
  carrying a linear light profile. Extra galaxy centres are loaded from a `.json` file and fixed in the
  model (Option A: one `SersicSph` per extra; Option B in the prose: MGE bulges per extra).
- `simulator`: Simulate an interferometer dataset containing one main galaxy and 2 extra galaxies, plus
  the `extra_galaxies_centres.json` metadata used by `modeling.py`.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/guides/results`.

# Lensing Equivalent

For the strong-lensing version of this feature (extras contribute mass, not light), see
`autolens_workspace/scripts/interferometer/features/extra_galaxies/`.

# Imaging Equivalent

For the CCD-imaging autogalaxy version, see `autogalaxy_workspace/scripts/imaging/features/extra_galaxies/`.
