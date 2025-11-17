"""
Model Cookbook
==============

The model cookbook provides a concise reference to model composition tools, specifically the `Model` and
`Collection` objects.

Examples using different PyAutoGalaxy API’s for model composition are provided, which produce more concise and
readable code for different use-cases.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autogalaxy as ag

"""
__Simple Model__

A simple model we can compose has a galaxy with a Sersic light profile:
"""

bulge = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

"""
The model `total_free_parameters` tells us the total number of free parameters (which are fitted for via a 
non-linear search), which in this case is 7.
"""
print(f"Model Total Free Parameters = {model.total_free_parameters}")

"""
If we print the `info` attribute of the model we get information on all of the parameters and their priors.
"""
print(model.info)

"""
__More Complex Models__

The API above can be easily extended to compose models where each galaxy has multiple light or mass profiles:
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bar = af.Model(ag.lp_linear.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk, bar=bar)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

"""
The use of the words `bulge`, `disk` and `bar` above are arbitrary. They can be replaced with any name you
like, e.g. `bulge_0`, `bulge_1`, `star_clump`, and the model will still behave in the same way.

The API can also be extended to compose models where there are multiple galaxies:
"""
bulge = af.Model(ag.lp_linear.Sersic)

galaxy_0 = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=bulge,
)

bulge = af.Model(ag.lp_linear.Sersic)

galaxy_1 = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=bulge,
)

model = af.Collection(
    galaxies=af.Collection(
        galaxy_0=galaxy_0,
        galaxy_1=galaxy_1,
    )
)

print(model.info)

"""
__Concise API__

If a light profile is passed directly to the `af.Model` of a galaxy, it is automatically assigned to be a `af.Model` 
component of the galaxy.

This means we can write the model above comprising multiple light profiles more concisely as follows:
"""
galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp_linear.Sersic,
    disk=ag.lp_linear.Exponential,
    bar=ag.lp_linear.Sersic,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

"""
__Prior Customization__

We can customize the priors of the model component individual parameters as follows:
"""
bulge = af.Model(ag.lp_linear.Sersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=4.0, sigma=1.0, lower_limit=1.0, upper_limit=8.0
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=bulge,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

"""
__Model Customization__

We can customize the model parameters in a number of different ways, as shown below:
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

# Parameter Pairing: Pair the centre of the bulge and disk together, reducing
# the complexity of non-linear parameter space by N = 2

bulge.centre = disk.centre

# Parameter Fixing: Fix the sersic_index of the bulge to a value of 4, reducing
# the complexity of non-linear parameter space by N = 1

bulge.sersic_index = 4.0

# Parameter Offsets: Make the bulge effective_radius parameters the same value as
# the disk but with an offset.

bulge.effective_radius = disk.effective_radius + 0.1

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

# Assert that the effective radius of the bulge is larger than that of the disk.
# (Assertions can only be added at the end of model composition, after all components
# have been bright together in a `Collection`.
model.add_assertion(
    model.galaxies.galaxy.bulge.effective_radius
    > model.galaxies.galaxy.disk.effective_radius
)

# Assert that the bulge effetive radius is below 3.0":
model.add_assertion(model.galaxies.galaxy.bulge.effective_radius < 3.0)

print(model.info)

"""
__Available Model Components__

The light profiles, mass profiles and other components that can be used for galaxy modeling are given at the following
API documentation pages:

 - https://pyautogalaxy.readthedocs.io/en/latest/api/light.html
 - https://pyautogalaxy.readthedocs.io/en/latest/api/mass.html
 - https://pyautogalaxy.readthedocs.io/en/latest/api/pixelization.html
 
 __JSon Outputs__
 
 After a model is composed, it can easily be output to a .json file on hard-disk in a readable structure:
"""
import os
import json

model_path = Path("path", "to", "model", "json")

os.makedirs(model_path, exist_ok=True)

model_file = Path(model_path, "model.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file.
"""
model = af.Model.from_json(file=model_file)

print(model.info)

"""
This means in **PyAutoGalaxy** one can write a model in a script, save it to hard disk and load it elsewhere, as well
as manually customize it in the .json file directory.

__Many Profile Models (Advanced)__

Features such as the Multi Gaussian Expansion (MGE) and shapelets compose models consisting of 50 - 500+ light
profiles.

The following example notebooks show how to compose and fit these models:

https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/modeling/imaging/features/multi_gaussian_expansion.ipynb
https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/modeling/imaging/features/shapelets.ipynb

__Model Linking (Advanced)__

When performing non-linear search chaining, the inferred model of one phase can be linked to the model.

The following example notebooks show how to compose and fit these models:

https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/imaging/advanced/guides/modeling/chaining.ipynb

__Across Datasets (Advanced)__

When fitting multiple datasets, model can be composed where the same model component are used across the datasets
but certain parameters are free to vary across the datasets.

The following example notebooks show how to compose and fit these models:

https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/multi/modeling/start_here.ipynb

__Relations (Advanced)__

We can compose models where the free parameter(s) vary according to a user-specified function 
(e.g. y = mx +c -> effective_radius = (m * wavelength) + c across the datasets.

The following example notebooks show how to compose and fit these models:

https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/multi/modeling/features/wavelength_dependence.ipynb

__PyAutoFit API__

**PyAutoFit** is a general model composition library which offers even more ways to compose models not
detailed in this cookbook.

The **PyAutoFit** model composition cookbooks detail this API in more detail:

https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html
https://pyautofit.readthedocs.io/en/latest/cookbooks/multi_level_model.html

__Wrap Up__

This cookbook shows how to compose simple models using the `af.Model()` and `af.Collection()` objects.
"""
