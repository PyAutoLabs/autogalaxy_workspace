The `cluster` folder contains example scripts showing how to model the light of a cluster's galaxy
population: a brightest cluster galaxy (BCG) modeled individually, plus tens-to-hundreds of member
galaxies driven by a **catalogue** — a CSV of centres and luminosities whose photometry pins the faint
members, leaving only shared normalizations free. Adding a member is a row append; the model
dimensionality does not grow with the population.

This is the top rung of PyAutoGalaxy's regime ladder (`imaging` → `multi_galaxy` → `cluster` — see
`multi_galaxy/README.md` for the ladder table).

# A Note for Lensing Users

`autolens_workspace` has a `cluster` package too, built on the same catalogue machinery (the
`scaling_galaxies.csv` schema is shared). The deliberate divergence: **here the foreground galaxies'
light is the entire subject and is always modeled; in PyAutoLens's cluster workflow lens light is not
modeled at all** (it fits point-source multiple-image positions of the lensed sources; lens-light
modeling will arrive there later as a feature).

# Start Here

New users should read the `start_here` example, which gives an overview of all examples in the folder.

# Files

- `start_here`: Fit a cluster field — free BCG MGE + a catalogue-driven member tier with one shared free
  normalization.
- `modeling`: Refinements — shared free member shapes, promoting bright members to their own free models,
  truth comparison.
- `simulator`: How the example field (1 BCG + 10 members) and its `scaling_galaxies.csv` were simulated.
