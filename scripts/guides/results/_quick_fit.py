"""
Results: Quick Fit Helper
=========================

Internal helper invoked via subprocess from the tutorials in this folder
(``start_here.py`` and everything under ``aggregator/``). Produces a fast,
capped Nautilus fit at ``output/results_folder/`` so the aggregator examples
have a populated results directory to read from.

Idempotent: exits immediately if ``output/results_folder/`` already exists,
so concurrent or repeated invocations are cheap.

Not a tutorial. The model and dataset mirror those used in ``start_here.py``
(simple imaging, single galaxy, Sersic bulge + Exponential disk), but the
search is hard-capped at ``n_like_max=300`` likelihood evaluations rather
than running to convergence. This produces a shallow but valid posterior
fast enough to fit inside the per-script CI timeout.
"""

import sys
from pathlib import Path

results_path = Path("output") / "results_folder"
if results_path.exists():
    sys.exit(0)

import autofit as af
import autogalaxy as ag

dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

if not dataset_path.exists():
    import subprocess

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

search.fit(model=model, analysis=analysis)
