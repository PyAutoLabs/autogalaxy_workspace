The `interferometer/data_preparation` package provides tools for preparing an interferometer
dataset (e.g. Hubble Space Telescope) before **PyAutoGalaxy** analysis:

# Files


# Files (Beginner / In Imaging)

The following scripts are used to prepare components of an interferometer dataset, however they are used in an
identical fashion for dataset datasets.

Therefore, they are not located in the `interferometer/data_preparation` package, but instead in the
`imaging/data_preparation` package, so refer there for a description of their usage.

Note that in order to perform some tasks (e.g. mark on the image where the source is), you will need to use an image
of the interferometer data even though visibilities are used for the analysis.

- `light_centre`: Masking the centre of the galaxy(s) light to help compose the model.
- `extra_galaxies_centres`: Adding additional extra galaxy centres, which add extra light and mass profiles to a model.
- `info`: Adding information to the dataset (e.g. redshifts) to aid analysis after modeling.
