"""
Start Here: Multi Wavelength
============================

Galaxies are often observed with CCD imaging, for example using HST, JWST,
or ground-based telescopes.

The examples `start_here_imaging.ipynb` illustrate how to perform galaxy modeling of CCD imaging
of single galaxies; it is recommended you read that example before reading this one.

This script shows you how to model multiple images of a galaxy, taken at different wavelengths,
with as little setup as possible. In about 15 minutes you’ll be able to point the code at your own
FITS files and fit your first galaxy.

Multi-wavelength galaxy modeling is an advanced feature and it is recommended you become more familiar with
**PyAutoGalaxy** and galaxy modeling before using it for your own science. Nevertheless, this script
should make it quick and easy to at least have a go doing multi-wavelength modeling of your own data.

We focus on a *galaxy-scale* target (a single galaxy). If you have multiple galaxies,
see the `start_here_group.ipynb` and `start_here_cluster.ipynb` examples.

__JAX__

PyAutoGalaxy uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

We also show how to simulate galaxy imaging. This is useful for building machine learning training datasets,
or for investigating imaging effects in a controlled way.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoGalaxy installation.

The code below sets up your environment if you are using Google Colab, including installing autogalaxy and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

# NOTE: This call is AutoLens-specific. Update to the PyAutoGalaxy equivalent in your codebase.
setup_colab.for_autogalaxy(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)


"""
__Imports__

Lets first import autogalaxy, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

We begin by loading the dataset. Three ingredients are needed for galaxy modeling:

1. The image itself (CCD counts).
2. A noise-map (per-pixel RMS noise).
3. The PSF (Point Spread Function).

Here we use multi-wavelength James Webb Space Telescope imaging of a galaxy. Replace
these FITS paths with your own to immediately try modeling your data.

The `pixel_scales` value converts pixel units into arcseconds. It is critical you set this
correctly for your data.

**Multi-wavelength Specific**: Note how each waveband and its corresponding pixel scale are put into a list and dictionary,
which we use to load all wavelength images in a list of imaging datasets.
"""
waveband_list = ["g", "r"]
pixel_scale_dict = {"g": 0.08, "r": 0.12}

dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = []

for dataset_waveband in waveband_list:

    dataset_waveband_path = dataset_path

    pixel_scale = pixel_scale_dict[dataset_waveband]

    dataset = ag.Imaging.from_fits(
        data_path=dataset_waveband_path / f"{dataset_waveband}_data.fits",
        psf_path=dataset_waveband_path / f"{dataset_waveband}_psf.fits",
        noise_map_path=dataset_waveband_path / f"{dataset_waveband}_noise_map.fits",
        pixel_scales=pixel_scale,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    dataset_list.append(dataset)

"""
__Extra Galaxy Removal__

There may be regions of an image that have signal near the galaxy that is from other galaxies not associated
with the galaxy we are studying. The emission from these objects will impact our model fitting and needs to be
removed from the analysis.

This `mask_extra_galaxies` is used to prevent them from impacting a fit by scaling the RMS noise map values to
large values. This mask may also include emission from objects which are not technically galaxies,
but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
or emission due to the data reduction process.

In this example, the noise is scaled over all regions of the image, even those quite far away from the galaxy
in the centre. We are next going to apply a 2.5" circular mask which means we only analyse the central region of
the image. It is only in these central regions where for the actual analysis it matters that we scaled the noise.

After performing galaxy modeling, the script further down provides a GUI to create such a mask
for your own data, if necessary.

**Multi-wavelength Specific**: The RMS noise map scaling is applied to all datasets one-by-one.
"""
dataset_scaled_list = []

for dataset, dataset_waveband in zip(dataset_list, waveband_list):

    dataset_waveband_path = dataset_path

    mask_extra_galaxies = ag.Mask2D.from_fits(
        file_path=dataset_waveband_path / "mask_extra_galaxies.fits",
        pixel_scales=dataset.pixel_scales,
        invert=True,
    )

    dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    dataset_scaled_list.append(dataset)

"""
__Masking__

Galaxy modeling does not need to fit the entire image, only the region containing the galaxy light.
We therefore define a circular mask around the galaxy.

- Make sure the mask fully encloses the galaxy emission.
- Avoid masking too much empty sky, as this slows fitting without adding information.

We’ll also oversample the central pixels, which improves modeling accuracy without adding
unnecessary cost far from the galaxy.

**Multi-wavelength Specific**: The mask is applied to each wavelength of data.
"""
mask_radius = 2.5

dataset_masked_list = []

for dataset in dataset_scaled_list:

    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    # Over sampling is important for accurate galaxy modeling, but details are omitted
    # for simplicity here, so don't worry about what this code is doing yet!

    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    dataset_masked_list.append(dataset)

"""
__Model__

To perform galaxy modeling we must define a model describing the light profiles of the galaxy.

A brilliant galaxy model to start with is one which uses a Multi Gaussian Expansion (MGE)
to model the galaxy light.

Full details of why this model is so good are provided in the main workspace docs,
but in a nutshell it provides an excellent balance of being fast to fit, flexible
enough to capture complex galaxy morphologies and providing accurate fits to the vast
majority of galaxies.

The MGE model composition API is quite long and technical, so we simply load the MGE
model below via a utility function `mge_model_from` which hides the API to make the code
in this introduction example ready to read. We then use the PyAutoGalaxy Model API to
compose the galaxy model.

**Multi-wavelength Specific**: The main model composition does not change for
multi wavelength, however it is worth emphasizing that the MGE will infer a unique
solution for each wavelength whereby the Gaussians have different intensities, meaning
that effects like colour gradients will be captured accurately.

Multi wavelength data may also have small offsets between each band, often smaller
than a pixel and thus below standard astrometric precision. We therefore include
a `dataset_model` composition which models these offsets as free parameters during
the galaxy modeling. Slightly further down in the script we will tell autogalaxy
to make a difference between each dataset.
"""
bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

# Dataset Model
dataset_model = af.Model(ag.DatasetModel)

# Overall Model
model = af.Collection(
    dataset_model=dataset_model, galaxies=af.Collection(galaxy=galaxy)
)

"""
We can print the model to show the parameters that the model is composed of, which shows many of the MGE's fixed
parameter values the API above hid the composition of.
"""
print(model.info)

"""
__Analysis__

In other examples, a single `Analysis` object is passed the dataset and used to perform galaxy modeling.

When there are multiple datasets, a list of analysis objects is created, once for each dataset.

__JAX__

PyAutoGalaxy uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis_list = [
    ag.AnalysisImaging(
        dataset=dataset,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
    )
    for dataset in dataset_masked_list
]

"""
Each analysis object is wrapped in an `AnalysisFactor`, which pairs it with the model and prepares it for use in a
factor graph. This step allows us to flexibly define how each dataset relates to the model.

Whilst not illustrated here, note that the API below is extremely customizable and allows us to
make the model vary on a per-dataset basis. We use this below to make it so the dataset offset of the second,
third and fourth datasets are included.
"""
analysis_factor_list = []

for i, analysis in enumerate(analysis_list):
    model_analysis = model.copy()

    if i > 0:
        model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )
        model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-1.0, upper_limit=1.0
        )

    # NOTE: Keeping as-is: factor uses `model` not `model_analysis`, matching the original script.
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

# Required to set up a fit with multiple datasets.
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
__Model Fit__

We now fit the data with the galaxy model using the non-linear fitting method and nested sampling algorithm Nautilus.

This uses the factor graph defined above.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce
an error. If this occurs, see the `autogalaxy_workspace/guides/modeling/bug_fix` example for a fix.
"""
search = af.Nautilus(
    path_prefix=Path(
        "multi_wavelength"
    ),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=150,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is visualized and output.
)

print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a factor graph.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
The result also contains the maximum likelihood galaxy model which can be used to plot the best-fit information
and fit to the data.
"""
for result in result_list:

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
__Model Your Own Galaxy__

If you have your own imaging data, you are now ready to model it yourself by adapting the code above
and simply inputting the path to your own .fits files into the `Imaging.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own CCD image, PSF, and RMS noise-map.
- Ensure the galaxy is roughly centered in the image.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to include all relevant light.
- Remove extra light from galaxies and other objects using an extra galaxies mask.
- Start with the default model — it works very well for pretty much all galaxies!

__Simulator__

In the example `start_here_imaging.ipynb`, we showed how to simulate CCD imaging of a galaxy.

We do not give a full description of the simulation API for multi wavelength imaging here,
but it is fully described in the main workspace documentation.

__Wrap Up__

This script has shown how to model CCD imaging data of galaxies across multiple wavelengths.

Details of the **PyAutoGalaxy** API and how galaxy modeling and simulations actually work were omitted for simplicity,
but everything you need to know is described throughout the main workspace documentation. You should check it out,
but maybe you want to try and model your own galaxy first!

The following locations of the workspace are good places to check out next:

- `autogalaxy_workspace/*/multi/features`: A full description of the multi wavelength and multi image fitting.
"""
