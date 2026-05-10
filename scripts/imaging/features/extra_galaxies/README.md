The `extra_galaxies` folder contains example scripts showing how to perform analysis using the extra galaxies
API, which includes extra galaxies surrounding a galaxy in the galaxy model, modeling their light and / or mass.

# Files

The following example scripts illustrating galaxy modeling where:

- `modeling`: galaxy modeling using a model which includes extra galaxies.
- `simulator`: Simulating galaxy data with extra galaxies surrounding it.
- `chaining`: Using non-linear search chaining to perform modeling where extra galaxies are fitted one-by-one initially.

# Results

These scripts only give a brief overview of how to analyse and interpret the results a galaxy model fit.

A full guide to result analysis is given at `autogalaxy_workspace/*/results`.

# Scaling Relations (not applicable in autogalaxy)

The `autolens_workspace` ships a companion `scaling_relation` feature that ties the `einstein_radius` of many
faint companion galaxies to their luminosities via two shared free parameters:

    einstein_radius = scaling_factor * (luminosity ** scaling_exponent)

**This pattern does not transfer to autogalaxy.** Two reasons:

1. **It's a mass relation.** The relation parameterises a mass profile parameter (`einstein_radius`).
   Autogalaxy is a light-only modeling library — there is no mass model to attach the relation to.

2. **The natural light-only analogues need quantities autogalaxy doesn't fit.** The classical light-only
   scaling relations (Faber–Jackson, Tully–Fisher) tie luminosity to velocity dispersion or rotation velocity.
   Neither is fit by an autogalaxy light-profile model. Linear light profiles (`ag.lp_linear.*`) already solve
   `intensity` via inversion, so layering a luminosity-driven `intensity` scaling on top would be degenerate —
   the linear inversion would absorb the relation and you'd recover the same fit either way.

If you arrived here looking for the autolens `scaling_relation` analogue, the relevant examples live at:

- `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — single-tier (one
  `extra_galaxies` collection mixing individually-modelled extras with a scaling-relation tier)
- `autolens_workspace/scripts/group/features/scaling_relation/modeling.py` — three-tier (main lens galaxies +
  individually-modelled extras + scaling galaxies on a shared relation)
- `autolens_workspace/scripts/group/features/scaling_relation/modeling_for_luminosities.py` — a standalone
  light-only fit (no mass, no source) that produces per-galaxy luminosities for the relation to consume

For the simpler "model each companion galaxy individually" pattern that does carry over to autogalaxy, see
`modeling.py` in this folder — both the `SersicSph` (Option A) and MGE (Option B) extras conventions are
demonstrated there.
