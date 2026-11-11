"""
Features: Point Source
======================

An unresolved source has no measured intrinsic spatial extent at the resolution of the observation. Its
observed appearance is set by the instrumental response and its location and flux.

__Separate the Compact and Extended Emission__

An unresolved nuclear component can sit inside an extended host galaxy. Their light is combined before
comparison with the observed image, so the PSF and host-light model both matter to their separation.

An intrinsic point source and an operated profile are related but different descriptions: the latter already
includes the relevant instrumental operation. Avoid applying that response twice.

__What This Folder Provides__

This feature currently supplies a simulator showing an extended host and an unresolved point component. Use it
to understand the model and the resulting data. It is not presented as a complete inference example; follow
the listed scripts for the functionality actually demonstrated.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
galaxy light is modeled in the observed field; there is no lens-source plane mapping. Retain the particular
light-component choices stated in each recipe.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
