"""
Results: Galaxies and Fits
===========================

This tutorial inspects an inferred model using galaxies inferred by the non-linear search.
This allows us to visualize and interpret its results.

The galaxies and fit API is described fully in the guides:

 - `autogalaxy_workspace/*/guides/galaxies.ipynb`
 - `autogalaxy_workspace/*/guides/fit.ipynb`

This result example only explains specific functionality for using a `Result` object to inspect galaxies or a fit
and therefore you should read these guides in detail first.

__Contents__

- **Model Fit:** Perform a model-fit to create a Result object for inspecting galaxies and fits.
- **Max Likelihood Galaxies:** Visualize the maximum likelihood galaxies from the result.
- **Samples:** Extract individual galaxy components from the samples and plot their images.
- **Refitting:** Refit the data using different samples to inspect how the fit varies.

__Units__

In this example, all quantities are **PyAutoGalaxy**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoGalaxy** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autogalaxy_workspace/*/guides/data_structure.ipynb` guide.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Quick Fit Auto-Trigger__

If a previous fit has not been run yet, the shared helper ``_quick_fit.py`` is invoked to produce one.
The helper writes a capped Nautilus fit to ``output/results_folder/`` so this tutorial (and its siblings
in this folder) have results to work with. When that folder already exists the helper exits immediately,
so re-running this tutorial is cheap.
"""
results_path = Path("output") / "results_folder"
if not results_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/guides/results/_quick_fit.py"],
        check=True,
    )

"""
__Model Fit__

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

The code below performs a model-fit using nautilus. The helper above already wrote a completed fit to
``output/results_folder/``, so the ``search.fit(...)`` call below resumes from that checkpoint and
returns the in-memory ``Result`` object without redoing the search.

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )


dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=Path("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_batch=50,
    n_live=100,
    n_like_max=300,
)

analysis = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Galaxies__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_galaxies` which we can visualize.
"""
galaxies = result.max_log_likelihood_galaxies

aplt.subplot_galaxies(galaxies=galaxies, grid=mask.derive_grid.all_false)

"""
This ensures that when interpreting results, the intensities are adjusted to reflect the true non-linear values.

__Samples__

In the first results tutorial, we used `Samples` objects to inspect the results of a model.

We saw how these samples created instances, which include a `galaxies` property that mains the API of the `Model`
creates above (e.g. `galaxies.galaxy.bulge`). 

We can also use this instance to extract individual components of the model.
"""
samples = result.samples

ml_instance = samples.max_log_likelihood()

# Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
fit = ag.FitImaging(dataset=dataset, galaxies=ml_instance.galaxies)
galaxies = fit.galaxies_linear_light_profiles_to_light_profiles

bulge = galaxies[0].bulge

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
print(bulge_image_2d.slim[0])

aplt.plot_array(array=bulge.image_2d_from(grid=dataset.grid), title="Image")


"""
__Refitting__

Using the API introduced in the `samples.py` tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-1)

galaxies = ag.Galaxies(galaxies=instance.galaxies)

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

aplt.subplot_fit_imaging(fit=fit)

"""
Fin.
"""
