The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoGalaxy** analysis:

Files (Advanced)
----------------

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``mask``: Choosing a mask for the analysis.
- ``light_centre``: Masking the centre of the galaxy(s) light to help compose the model.
- ``extra_galaxies_centres``: Adding additional extra galaxy centres, which add extra light profiles to a model.
- ``mask_extra_galaxies``: Removing unwanted light from interpolator galaxies and stars in an image.
- ``info``: Adding information to the dataset (e.g. redshifts) to aid analysis after modeling.