"""
Using JAX
=========

**PyAutoGalaxy** runs on either **NumPy** (the default) or **JAX** (Google's array library with GPU support and
just-in-time compilation). JAX makes galaxy model-fitting 10-100x faster on large grids — sometimes more on GPU —
so the library is built to use it automatically wherever it helps.

The `start_here.py` introduction covers the one thing every user needs to know: install the JAX extra
(`pip install autogalaxy[jax]` on Python 3.11+) and model-fitting uses JAX automatically. This guide covers the
technical detail behind that, and the situations where you interact with JAX directly.

__Contents__

- **Auto-Enabled Modeling:** What "JAX is used automatically" means under the hood for model-fits.
- **Disabling JAX:** Forcing the NumPy path per-analysis or globally, and why that helps when debugging.
- **Writing @jax.jit Yourself:** Custom simulations rendered fast with the simulator's JAX mode.
- **JIT-ing Library Methods:** The advanced path wrapping library methods like `galaxy.image_2d_from` directly.
- **Return-Type Contract:** What `jax.Array` data inside results means for plotting, saving and arithmetic.

__Auto-Enabled Modeling__

If JAX is installed, the `AnalysisImaging` and `AnalysisInterferometer` classes default to `use_jax=True`. The
non-linear search driver (Nautilus, dynesty, ...) batches parameter vectors and evaluates the likelihood through
`jax.vmap(jax.jit(...))` internally. You'll see a one-time log line like
`JAX: Applying vmap and jit to likelihood function -- may take a few seconds.` the first time a search starts;
that's the JIT compile kicking in, after which evaluations re-use the compiled trace.

If JAX is not installed, the analysis warns once and falls back to NumPy automatically.

__Disabling JAX__

You can force the NumPy path explicitly with `ag.AnalysisImaging(dataset=dataset, use_jax=False)`, or globally by
setting the environment variable `PYAUTO_DISABLE_JAX=1`.

This is useful when debugging: NumPy stack traces are easier to read than JAX traces, and you can drop a debugger
or `print` statement into code that JAX would otherwise trace and compile.

__Writing @jax.jit Yourself__

Pass `use_jax=True` to a simulator constructor and wrap your call in `@jax.jit` when you want to render many
datasets fast — parameter sweeps, mock-data studies, batch figure generation:

```python
import jax

simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, use_jax=True
)

@jax.jit
def simulate(galaxies):
    return simulator.via_galaxies_from(galaxies=galaxies, grid=grid)
```

The simulator handles pytree registration internally, so you write nothing JAX-specific beyond the decorator.
Note that eager `simulator.via_galaxies_from(galaxies, grid)` (no `@jax.jit`) already runs on JAX and is
sufficient for one-off simulations — the `@jax.jit` wrap only pays off when you call the function many times.

The per-dataset-type `simulator.py` scripts (`scripts/imaging/simulator.py`,
`scripts/interferometer/simulator.py`) each show this pattern in their `__JAX Variant__` section.

__JIT-ing Library Methods__

The advanced path is JIT-ing library methods directly (`galaxy.image_2d_from`, etc.) without going through a
`Simulator` or `Analysis`. A library method called inside `@jax.jit` must be passed `xp=jnp` so its internal
arithmetic stays on the JAX path.

The `data_structures.py` guide (`scripts/guides/data_structures.py`) covers what you need for this — when the
backing array of a data structure becomes a `jax.Array`, and the "not-pytree rule" for what your `@jax.jit`
function may return.

__Return-Type Contract__

When `use_jax=True`, the data structures you get back (`Imaging`, `FitImaging`, `Galaxies.image_2d_from(...)`
results, ...) carry `jax.Array` data inside instead of `numpy.ndarray`. For nearly everything you'd do in a
workspace — plotting, saving to `.fits`, comparing fit residuals — this is transparent: the plotters and FITS
writers call `numpy.asarray()` internally and you see the same images and numbers you would on the NumPy path.

What changes:

- Arithmetic on JAX arrays stays on the JAX path. Direct calls into NumPy (`np.sqrt(fit.residual_map.array)`)
  will host-transfer the array off the GPU; not wrong, but slower than `jnp.sqrt(...)` if you're inside a hot
  loop. For one-off analysis code, don't worry about it.
- The `.array` property of `aa.Array2D` etc. is the raw backing array — a `numpy.ndarray` on the NumPy path, a
  `jax.Array` on the JAX path.

The `data_structures.py` guide (`scripts/guides/data_structures.py`) covers the wrapper-vs-raw-array distinction
in detail.
"""
