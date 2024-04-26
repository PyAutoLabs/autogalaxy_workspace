"""
__Log Likelihood Function: Pixelization__

The `pixelization` script accompanying this one provides a step-by-step guide of
the **PyAutoGalaxy** `log_likelihood_function used to fit `Imaging` data with a pixelization.

This script provides a contributor guide, that gives links to every part of the source-code that performs each step
of the likelihood evaluation.

This gives contributors a linear run through of what functions, modules and packages in the source code are called
when the likelihood is evaluated, and should help them navigate the source code.


__Source Code__

The likelihood evaluation is spread over the following two GitHub repositories:

**PyAutoArray**: https://github.com/Jammy2211/PyAutoArray
**PyAutoGalaxy**: https://github.com/Jammy2211/PyAutoGalaxy


__LH Setup: Light Profiles (Setup)__

To see examples of all light profiles in **PyAutoGalaxy** checkout  the `light_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/light

Each light profile has an `image_2d_from` method that returns the image of the profile, which was used in the
likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the light profile's geometry.


__LH Setup: Galaxy__

The galaxy package and module contains the `Galaxy` object:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/galaxy

The `Galaxy` object also has an `image_2d_from` method that returns the image of the galaxy, which calls the
`image_2d_from` functiuon of each light profile and sums them.


__LH Setup: Pixelization and Regularization__

To see examples of all pixelizations and regularization schemes checkout the pixelization packages in **PyAutoArray**:

 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/pixelization
 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/regularization


__LH Step 1: Galaxy Image__

This step calls the `image_2d_from` method of the `Galaxy` object, as described above.

The input to this function is the masked dataset's `Grid2D` object, which is computed here:

- See the method `grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/abstract/dataset.py

The calculation also used a `blurring_grid` to evaluate light which the PSF convolution blurred into the mask,
which is computed here:

- See the method `blurring_grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/imaging/dataset.py


__LH Step 2: Galaxy Light Convolution__

Convlution uses the `Convolver` object and its method `convolve_image`

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 7: Image-Source Mapping__

 Checkout the modules below for a full description of a `Mapper` and the `mapping_matrix`:

 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/mappers
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/voronoi.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 8: Mapping Matrix__

 `Mapper.__init__`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
 `mapping_matrix_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 9: Blurred Mapping Matrix (f)__

 This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix` (it here where our
 use of real space convolution can exploit sparcity to speed up the convolution compared to an FFT):

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 10: Data Vector (D)__

The calculation is performed by the method `data_vector_via_blurred_mapping_matrix_from` at:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

This function is called by `LEqMapping.data_vector_from()` to make the `data_vector`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/imaging.py


__LH Step 11: Curvature Matrix (F)__

 The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

 This function is called by `LEqMapping.curvature_matrix` to make the `curvature_matrix`:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/imaging.py


__LH Step 12: Regularization Matrix (H)__

 A complete description of regularization is at the link below.

 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/regularization
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/abstract.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/constant.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/regularization_util.py

 An `Inversion` object has a property `regularization_matrix` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 13: F + Lamdba H__

 An `Inversion` object has a property `curvature_reg_matrix` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py


__LH Step 14: Source Reconstruction (S)__

 An `Inversion` object has a property `reconstruction` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py


__LH Step 15 Image Reconstruction__

 The calculation is performed by the method `mapped_reconstructed_data_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

 This function is called by `AbstractInversion.mapped_reconstructed_data`:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 18: Regularization Term__

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 19: Complexity Terms__

 An `Inversion` object has a property `log_det_curvature_reg_matrix_term` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py

 An `Inversion` object has a property `log_det_regularization_matrix_term` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 20: Likelihood Function__

The `model_image` computed previously was subtracted from the observed image, and the residuals, chi-squared
and log likelihood computed.

This is performed in the `FitImaging` object which is spread over **PyAutoArray** and **PyAutoGalaxy**:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/imaging/fit_imaging.py

The following methods are relevant in this module:

`blurred_image`: Computes the blurred image via convolution described above.
`model_data`: This is the blurred image, but the variable is renamed as for more advanced fits it is extended.

The steps of subtracting the model image from the observed image and computing the residuals, chi-squared and
log likelihood are performed in the `FitDataset` and `FitImaging` object of **PyAutoArray**:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_dataset.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_imaging.py

Many calculations occur in `fit_util.py`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_util.py

These modules also ihnclude the calcualtion of the `noise_normalization`.
"""
