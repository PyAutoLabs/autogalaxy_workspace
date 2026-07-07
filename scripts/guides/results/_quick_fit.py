"""
Results: Quick Fit Helper
=========================

Internal helper invoked via subprocess from the tutorials in this folder.
Produces two fast, capped Nautilus fits at ``output/results_folder/`` so the
aggregator and workflow examples have a populated results directory to read
from.

Idempotent: exits immediately if ``output/results_folder/`` already contains
the two completed imaging fits, so concurrent or repeated invocations are
cheap.

Not a tutorial. The model and dataset mirror those used in ``start_here.py``
(simple imaging, single galaxy, Sersic bulge + Exponential disk), but the
search is hard-capped at ``n_like_max=300`` likelihood evaluations rather
than running to convergence. This produces a shallow but valid posterior
fast enough to fit inside the per-script CI timeout.
"""

import shutil
import sys
from pathlib import Path

from autoconf.test_mode import with_test_mode_segment

results_path = with_test_mode_segment(Path("output")) / "results_folder"
if (
    len(list(results_path.glob("**/image/dataset.fits"))) >= 2
    and len(list(results_path.glob("**/files/latent/latent_summary.json"))) >= 2
    and len(list(results_path.glob("**/image/fit.png"))) >= 2
    and len(list(results_path.glob("**/image/fit.fits"))) >= 2
):
    sys.exit(0)

if results_path.exists():
    shutil.rmtree(results_path)

import os

# The aggregator tutorials that invoke this helper read image/dataset.fits via
# fit.value("dataset"). Smoke-mode env vars (PYAUTO_TEST_MODE>=2,
# PYAUTO_SKIP_VISUALIZATION) suppress the visualizer that writes that file, so
# neutralize them here.
mode = os.environ.get("PYAUTO_TEST_MODE", "0")
if mode in ("2", "3"):
    os.environ["PYAUTO_TEST_MODE"] = "1"
os.environ.pop("PYAUTO_SKIP_VISUALIZATION", None)
os.environ.pop("PYAUTO_SKIP_FIT_OUTPUT", None)
os.environ.pop("PYAUTO_FAST_PLOTS", None)

import autofit as af
import autogalaxy as ag
from autoconf import conf

# This deliberately shallow helper must retain its exploratory samples because
# the results tutorials demonstrate indexed sample access.
conf.instance["output"]["samples_weight_threshold"] = None

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


class LatentSersicIndex(ag.Latent):
    """
    Custom latent catalogue reporting a derived Sersic-index quantity for the
    workflow CSV example.
    """

    @staticmethod
    def keys(analysis):
        return ["galaxies.galaxy.bulge.sersic_index_x2"]

    @staticmethod
    def variables(analysis, parameters, model):
        instance = model.instance_from_vector(vector=parameters)

        return (instance.galaxies.galaxy.bulge.sersic_index * 2.0,)


class AnalysisLatent(ag.AnalysisImaging):
    Latent = LatentSersicIndex


analysis = AnalysisLatent(dataset=dataset, use_jax=True)

for i in range(2):
    search = af.Nautilus(
        path_prefix=Path("results_folder"),
        name="results",
        unique_tag=f"{dataset_name}_{i}",
        n_batch=50,
        n_live=100,
        n_like_max=300,
    )

    search.fit(model=model, analysis=analysis)
