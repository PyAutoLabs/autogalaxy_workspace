"""
Overview: Pixelizations
-----------------------

Pixelizations reconstruct a galaxy's light on a pixel-grid.

Unlike `LightProfile`'s, they are able to reconstruct the light of non-symmetric and irregular galaxies.

To reconstruct the galaxy using a `Pixelization`, we impose a prior on the smoothness of the reconstructed
source, called the `Regularization`. The more we regularize the galaxy, the smoother the reconstruction.

The process of reconstructing a `Galaxy`'s light using a `Pixelization`  is called an `Inversion`,
and the term `inversion` is used throughout the **PyAutoGalaxy** example scripts to signify that their analysis
reconstructs the galaxy's light on a pixel-grid.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the `Imaging` data that we'll reconstruct the galaxy's light of using a pixelization.

Note how complex the galaxyed source galaxy looks, with multiple clumps of light - this would be very difficult to 
represent using `LightProfile`'s!
"""
dataset_name = "complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We are going to fit this data, so we must create `Mask2D` and `Imaging` objects.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)

dataset = dataset.apply_mask(mask=mask)

"""
__Mesh + Regularization__

To reconstruct the galaxy on a pixel-grid, called a mesh, we simply pass it the `Mesh` class we want to reconstruct its 
light on as well as the `Regularization` scheme describing how we smooth the source reconstruction. 

We use a `Rectangular` mesh with resolution 40 x 40 and a `Constant` regularizaton scheme with a regularization
coefficient of 1.0. The higher this coefficient, the more our source reconstruction is smoothed.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Fit__

Now that our galaxy has a `Pixelization`, we are able to fit the data using it in the same way as before, by simply 
passing the galaxy to a `Plane` and using this `Plane` to create a `FitImaging` object.
"""
plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

"""
__Pixelization__

The fit has been performed using an `Inversion` for the galaxy.

We can see that the `model_image` of the fit subplot shows a reconstruction of the observed galaxy that is close 
to the data.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, model_image=True)

"""
__Positive Only Solver__

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these 
unphysical solutions, which can degrade the results of lens model in general.

__Why Use Pixelizations?__

From the perspective of a scientific analysis, it may be unclear what the benefits of using an inversion to 
reconstruct a complex galaxy are.

When I fit a galaxy with light profiles, I learn about its brightness (`intensity`), size (`effective_radius`), 
compactness (`sersic_index`), etc.

What did I learn about the galaxy I reconstructed? Not a lot, perhaps.

Inversions are most useful when combined with light profiles. For the complex galaxy above, we can fit it with light 
profiles to quantify the properties of its `bulge` and `disk` components, whilst simultaneously fitting the clumps 
with the inversion so as to ensure they do not impact the fit.

The workspace contains examples of how to do this, as well as other uses for pixelizations.

__Wrap Up__

This script gives a brief overview of pixelizations. 

However, there is a lot more to using *Inversions* then presented here. 

In the `autogalaxy_workspace/*/modeling` folder you will find example scripts of how to fit a model to a 
galaxy using an `Inversion`. 

In chapter 4 of the **HowToGalaxy** lectures we fully cover all details of  *Inversions*, specifically:

 - How the inversion's reconstruction determines the flux-values of the galaxy it reconstructs.
 - The Bayesian framework employed to choose the appropriate level of `Regularization` and avoid overfitting noise.
 - Unphysical model solutions that often arise when using an `Inversion`.
"""
