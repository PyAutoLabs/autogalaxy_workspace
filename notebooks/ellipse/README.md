The `ellipse` folder shows how to measure galaxy-light morphology with PyAutoGalaxy's isophote-fitting tools.
Fit the centre, ellipticity and position angle at different radii, then use multipoles to describe departures
from elliptical symmetry.

# Start Here

Begin with [`fit.ipynb`](fit.ipynb) to understand the image sampling, residuals and likelihood. Continue with
[`modeling.ipynb`](modeling.ipynb) to infer ellipse parameters using a non-linear search, and
[`multipoles.ipynb`](multipoles.ipynb) to fit more complex isophote shapes.

These examples work on the observed image without PSF convolution of a light model. Account for PSF effects
when interpreting small-radius structure, and mask neighbouring light where appropriate.

# Files

- `fit`: Inspect a fixed ellipse fit, residuals, chi-squared and likelihood.
- `modeling`: Fit ellipse parameters and repeat at multiple radii.
- `multipoles`: Extend the isophotes with multipole perturbations.
- `simulator`: Create galaxy imaging for the ellipse examples.
- `database`: Load and inspect saved fit results.
- `plot`: Plot ellipses and their fits.

# Results

The modeling example introduces the fit results. See the workspace's
[results guides](../guides/results/) for more ways to inspect and compare model fits.
