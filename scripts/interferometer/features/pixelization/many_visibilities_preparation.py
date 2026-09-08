"""
Pixelization: Many Visibilities Preparation
===========================================

To perform many visibility modeling, a matrix called `nufft_precision_operator` is created and used, which encodes information
and symmetries into the Fourier transform operation performed when modeling interferometer datasets, in a way that
exploits the sparsity of the inversion — pixelizations, linear light profiles, or both — and means a very small
amount of memory or VRAM is used.

The details can be found in the source code, but you do not need to know them to do science with the code,
nevertheless this ultimately means datasets exceeding millions of visibilities can be modeled in under an hour on a GPU.

As of the next `autoarray` release this matrix is built as a type-1 (adjoint) NUFFT, which takes seconds. It was
previously computed by brute force, which took minutes to hours on large datasets: for an ALMA dataset with 1
million visibilities the build went from around 35 minutes to under 10 seconds, and 5 million visibilities now takes
around 20 seconds. These are CPU times, so you no longer need a GPU, or a preparation step like this one, to compute
the matrix at the start of every model-fit.

Saving the matrix to hard-disk is still supported, and this example still shows how, but it is no longer necessary
for run time. Matrices saved by earlier versions remain valid, because the array itself is unchanged.

What does still matter is memory. Building the matrix in one shot at millions of visibilities needs tens of GB, so
the calculation is chunked over visibilities. The chunk size is taken from the transformer's own `chunk_size` input,
so there is nothing to set by hand unless you want a lower memory ceiling.

This example therefore creates the `nufft_precision_operator` matrix using independent Python code and saves it to hard-disk
for modeling. The `cpu_fast_modeling` example loads this matrix from hard-disk if it is available,
and computes it from scratch if not.

__Contents__

- **High Resolution Dataset:** Information on downloading high-resolution ALMA uv-wavelength data.
- **Dataset + Masking:** Loading the interferometer dataset and defining the real-space mask.
- **Profiling Dataset:** Profiling run times for different dataset sizes to plan computation.
- **Curvature Preload:** Computing the sparse NUFFT operator matrix for fast pixelized modeling.
- **Curvature Preload Output:** Saving and loading the precomputed matrix to and from hard-disk.
- **Wrap Up:** Summary of the preparation workflow for many-visibility datasets.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autogalaxy_workspace` repository:

https://github.com/PyAutoLabs/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autogalaxy_workspace/dataset/interferometer/alma`

You can then compute the `nufft_precision_operator` matrix for this dataset by uncommenting
the line `dataset_name = "alma"` below.
"""

from autogalaxy import jax_wrapper  # Sets JAX environment before other imports

# from autogalaxy import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import time

import autogalaxy as ag

"""
__Dataset + Masking__

Load the `Interferometer` data, define the visibility and real-space masks.
"""
mask_radius = 3.0

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=mask_radius
)

dataset_name = "simple"
# dataset_name = "alma"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = ag.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerNUFFT,
)

"""
__Profiling Dataset__

The code above loads a dataset with very few visibilities and a low resolution real space mask, so the
`nufft_precision_operator` computation is near instant.

Real datasets often have 100,000+ visibilities, and a high resolution real space mask, which takes longer -- though
seconds, not the minutes to hours it used to.

It may therefore still be useful to profile the run times and memory use for different dataset sizes using the code
below, which overwrites the dataset above. This lets you check ahead of time that the computation fits in your
machine's memory for your dataset.

This code is commented out by default, so your dataset is used instead, but you can uncomment it to run the profiling.
"""
# ### Key run time parameters ###
#
# mask_radius = 3.0
# total_visibilities = 1000000
#
# ### Setup Data ###
#
# real_space_mask = ag.Mask2D.circular(
#     shape_native=(800, 800), pixel_scales=0.05, radius=mask_radius
# )
#
# data = ag.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
#     loc=0.0, scale=1.0, size=total_visibilities
# ))
#
# noise_map = ag.VisibilitiesNoiseMap(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))
#
# uv_wavelengths = np.random.uniform(
#     low=-300.0, high=300.0, size=(total_visibilities, 2)
# )
#
# dataset = ag.Interferometer(
#     data=data,
#     noise_map=noise_map,
#     uv_wavelengths=uv_wavelengths,
#     real_space_mask=real_space_mask,
#     transformer_class=ag.TransformerNUFFT,
# )

"""
__Curvature Preload__

Pixelized galaxy modeling requires dense linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **sparse linear algebra formalism**.

You do not need to understand the full details of the method, but the key point is:

- It exploits the **sparsity** of the matrices used in pixelized galaxy reconstruction.
- This leads to a **significant speed-up on GPU or CPU**, using JAX to perform the linear algebra calculations.

To enable this feature, we call `apply_sparse_operator()` on the dataset. This computes and stores a NUFFT operator 
matrix.

As discussed above, this matrix is built as a type-1 NUFFT and takes seconds, even for datasets with many
visibilities and high resolution real-space masks.

The code has the following inputs:

- `use_jax`: The NUFFT builder always runs via JAX, so this input does not change which builder is used. It selects
  between the two reference brute-force builders, and is kept here so this script also works on `autoarray`
  releases which predate the NUFFT builder.

- `chunk_k`: The chunk size of visibilities used by the brute-force builders. The NUFFT builder chunks at the
  transformer's own `chunk_size` instead, which is what keeps memory bounded at millions of visibilities.

- `show_progress`: Whether to output a progress bar to the terminal. This applies to the brute-force builders; the
  NUFFT builder finishes too quickly to need one.

- `show_memory`: Whether to output memory usage to the terminal, which is useful to ensure your system has enough
  memory to complete the computation.
"""
dataset = dataset.apply_sparse_operator(
    use_jax=True,
    chunk_k=2048,
    show_progress=True,
    show_memory=True,
)

"""
__Curvature Preload Output__

We now output the `nufft_precision_operator` object to hard-disk, so it can be loaded quickly in the 
`cpu_fast_modeling` example. This is optional now the matrix builds in seconds, but it still saves recomputing it
for every fit.

We save it using a numpy `npy` file, which compresses the data to save hard-disk space, and put it in the 
dataset folder so it can be easily found. 
"""
nufft_precision_operator = dataset.psf_precision_operator_from(
    use_jax=True,
    chunk_k=2048,
    show_progress=True,
    show_memory=True,
)

np.save(
    file=dataset_path / f"nufft_precision_operator_{mask_radius}.npy",
    arr=nufft_precision_operator,
    allow_pickle=False,
)
"""
To load the `nufft_precision_operator` matrix from hard-disk in your model-fit, you can use the code:
"""
nufft_precision_operator = np.load(
    file=dataset_path / f"nufft_precision_operator_{mask_radius}.npy",
    allow_pickle=False,
)

"""
__Wrap Up__

This example has demonstrated how to set up the linear algebra to perform fast pixelized galaxy modeling on
interferometer datasets with many visibilities, and how to save the `nufft_precision_operator` matrix to hard-disk.
Since the matrix now builds in seconds, saving it is a convenience rather than a requirement.
"""
