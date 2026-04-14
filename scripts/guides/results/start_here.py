"""
Results: Start Here
===================

After a model-fit completes, nearly everything a user could need is written to the `output/` folder. Most of it
can be loaded back into full Python objects with a single line of code, via either `.json` files (for model objects
like the `Galaxies`, `Model` and samples) or `.fits` files (for imaging products like the model image, residuals and
per-galaxy images).

This is the fast, simple way to inspect a single fit: loading is instant and the objects behave exactly like the
in-memory `Result` object returned by `search.fit()`. The trade-off is that everything you load sits in memory, so
if you need to analyse many fits together (e.g. a sample of hundreds of galaxies) you should use the **aggregator**
instead, which scrapes directories and uses generators to minimise memory use.

  - Simple loading (this file)       : fast, in-memory, one fit at a time
  - Aggregator (`aggregator/start_here.py`) : generator-based, memory-efficient, many fits

__Output Folder Layout__

Each completed fit lives at a path like::

    output/imaging/<dataset_name>/<search_name>/<unique_hash>/
        files/                     <- JSON + CSV: loadable Python objects
            galaxies.json          <- max log likelihood Galaxies collection
            model.json             <- fitted af.Collection model
            samples.csv            <- full Nautilus samples
            samples_summary.json   <- max log likelihood parameter values + errors
            samples_info.json      <- metadata about the samples
            search.json            <- non-linear search configuration
            settings.json          <- search settings
            cosmology.json         <- cosmology used for the fit
            covariance.csv         <- parameter covariance matrix
        image/                     <- FITS: imaging products
            dataset.fits           <- data, noise-map and PSF
            fit.fits               <- model image, residuals, chi-squared map
            model_galaxy_images.fits  <- per-galaxy model images
            galaxy_images.fits        <- per-galaxy images
            dataset.png, fit.png   <- visualisations
        model.info                 <- human-readable model summary
        model.results              <- human-readable fit summary
        search.summary             <- search run summary
        metadata                   <- run metadata

__Contents__

**Galaxies:** Load the maximum log likelihood `Galaxies` from `galaxies.json`.
**Model:** Load the fitted `af.Collection` model from `model.json`.
**Samples:** Load the non-linear search samples from `samples.csv` / `samples_summary.json`.
**FITS:** Load imaging products (data, fit, galaxy images) from the `image/` folder.
**Next:** Pointer to detailed aggregator examples for multi-fit workflows.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports
from autoconf.dictable import from_json

from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Result Path__

Set the path to the folder of the fit you want to load. Replace the values below with the path to your own fit.
"""
result_path = (
    Path("output")
    / "imaging"
    / "features"
    / "simple"
    / "start_here"
    / "<unique_hash>"  # The 32-character identifier for the specific fit.
)

files_path = result_path / "files"
image_path = result_path / "image"

"""
__Galaxies__

The maximum log likelihood `Galaxies` collection is saved to `files/galaxies.json` and can be loaded in one line.

It contains every `Galaxy` at its max log likelihood values (light and mass profiles included), so it can be used
directly to compute images, convergence maps and more — exactly as if it had been returned by `search.fit()`.
"""
galaxies = from_json(file_path=files_path / "galaxies.json")

print(galaxies)

"""
__Model__

The fitted `af.Collection` model is saved to `files/model.json`. This is the *prior* model (with free parameters),
not the max log likelihood instance — useful for inspecting the structure of what was fitted.
"""
model = from_json(file_path=files_path / "model.json")
print(model.info)

"""
__Samples__

The full set of non-linear search samples is saved to `files/samples.csv` and its summary to
`files/samples_summary.json`. Both can be loaded without re-running the search.
"""
samples = af.SamplesNest.from_csv(file_path=files_path / "samples.csv", model=model)
print(samples.max_log_likelihood())

"""
__FITS Files__

The `image/` folder contains the imaging products of the fit as `.fits` files. These load with the standard
`ag.Imaging` / `aa.Array2D` APIs and can be plotted with `aplt`.
"""
dataset = ag.Imaging.from_fits(
    data_path=image_path / "dataset.fits",
    noise_map_path=image_path / "dataset.fits",
    psf_path=image_path / "dataset.fits",
    data_hdu=0,
    noise_map_hdu=1,
    psf_hdu=2,
    pixel_scales=0.1,
)

galaxy_images = ag.Array2D.from_fits(
    file_path=image_path / "galaxy_images.fits", hdu=0, pixel_scales=0.1
)

"""
__Next__

For scripts that analyse many fits together using the aggregator — loading samples, galaxies and fits lazily via
Python generators to keep memory use low — see:

  `autogalaxy_workspace/*/guides/results/aggregator/start_here.py`

That script is the entry point for building sample-level scientific workflows. Individual topics (samples, galaxies,
data_fitting, queries, interferometer, models) are covered by sibling files inside `aggregator/`.
"""
