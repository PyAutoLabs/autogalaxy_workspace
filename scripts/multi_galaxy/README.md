The `multi_galaxy` folder contains example scripts showing how to model multi-galaxy systems: two or more
galaxies whose light blends together on the sky, so all of them must be modeled simultaneously — every
galaxy a co-equal subject of the fit with its own free light model (interacting pairs, close projected
pairs, compact multiples).

# The Regime Ladder

PyAutoGalaxy organises galaxy-light modeling into a ladder of three regimes, mirroring the lensing
workspace's ladder:

| Regime | Model | Package |
|---|---|---|
| Single galaxy | One free light model; neighbours masked | `imaging` |
| Multi galaxy | One free light model per blended galaxy (list-based `galaxy_0`, `galaxy_1`, ... API) | `multi_galaxy` |
| Cluster | BCG(s) modeled individually + a member population loaded from a CSV catalogue | `cluster` |

**A note for lensing users:** `autolens_workspace` mirrors these package names. The one deliberate
divergence is at the top rung — the PyAutoGalaxy cluster workflow models the foreground galaxies' light
(that is its entire subject), whereas the PyAutoLens cluster workflow does not model lens light at all
(it fits point-source multiple-image positions).

# Start Here

New users should read the `start_here` example, which gives an overview of all examples in the folder.

# Files

- `start_here`: A simple example illustrating how to model a blended pair of galaxies.
- `modeling`: Detailed example of the modeling API, per-galaxy decomposition and photometry.
- `simulator`: How the example dataset was simulated.
- `plot`: How to plot the dataset and fits to it.

# Folders

- `features`: Extensions of the multi-galaxy model. Currently `extra_galaxies`, which adds a lower tier of
  galaxies — companions in the field which must be accounted for but are not themselves subjects of the
  measurement, either scaled out of the fit or modeled with a fixed centre.

Standard single-galaxy features (linear light profiles, MGE variations, sky subtraction — see
`imaging/features`) apply per-galaxy here unchanged.
