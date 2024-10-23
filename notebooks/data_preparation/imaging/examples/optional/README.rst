The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoGalaxy** analysis:

Files (Advanced)
----------------

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``mask.py``: Choosing a mask for the analysis.
- ``light_centre.py``: Masking the centre of the galaxy(s) light to help compose the model.
- ``extra_galaxies_centres.py``: Adding additional extra galaxy centres, which add extra light profiles to a model.
- ``mask_extra_galaxies.py``: Removing unwanted light from interpolator galaxies and stars in an image.
- ``info.py``: Adding information to the dataset (e.g. redshifts) to aid analysis after modeling.