The `features` folder extends the multi-galaxy model of `multi_galaxy/start_here` and `multi_galaxy/modeling`,
which by design uses **only co-equal galaxies** — one free light model per blended galaxy, nothing else.

# Folders

- `extra_galaxies`: Add a lower tier of galaxies to the model — companions in the field which must be accounted
  for but are not themselves subjects of the measurement. They are either scaled out of the fit via a noise
  mask, or given a restricted light model with a fixed centre.

# Two Tiers

The `extra_galaxies` feature is what turns the multi-galaxy model into a two-tier model, and choosing the tier
for a given galaxy is the decision it exists to support:

- **Co-equal galaxies** get a full free light model with a free centre. Use this tier when a galaxy's light
  blends with the others and decomposing it is part of the science.
- **Extra galaxies** get a restricted light model with the centre fixed to the observed position. Use this tier
  when a galaxy has to be accounted for but no measurement of it is wanted.

The tiers describe intent, not brightness. A faint galaxy you want photometry for belongs in the first tier; a
bright galaxy you merely need to stop contaminating the fit belongs in the second.

# Single-Galaxy Features

Standard single-galaxy features — linear light profiles, MGE variations, sky handling — apply to multi-galaxy
systems unchanged. See `autogalaxy_workspace/*/imaging/features` and apply them per-galaxy via the `galaxy_0`,
`galaxy_1`, ... loop of this package.

# Scaling Relations (not applicable in autogalaxy)

The `autolens_workspace` ships a companion `scaling_relation` feature which ties the `einstein_radius` of many
faint companion galaxies to their luminosities via shared free parameters, so a large population costs only a
couple of parameters. It is the natural next step above the extra-galaxies tier at lensing multi-galaxy scale.

**That pattern does not transfer to autogalaxy**, for the reasons set out in full in
`autogalaxy_workspace/*/imaging/features/extra_galaxies/README.md`: it parameterises a *mass* profile parameter,
and autogalaxy is a light-only modeling library. The light-only analogues (Faber–Jackson, Tully–Fisher) tie
luminosity to velocity dispersion or rotation velocity, neither of which an autogalaxy light-profile model fits,
and linear light profiles already solve `intensity` by inversion so a luminosity-driven `intensity` scaling
would be degenerate.

When the number of extra galaxies grows past the point where one model each is practical, the autogalaxy answer
is the `cluster` package, which loads a member population from a CSV catalogue.

# Results

These scripts show how to perform modeling but only give a brief overview of how to analyse and interpret the
results of a model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/results`.
