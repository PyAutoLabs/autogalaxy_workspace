"""
Features: Point Source
======================

Explore how an unresolved nucleus contributes to a galaxy image and affects measurements of its extended host. The point-source component supplies a position and flux; the instrumental response determines its observed footprint.

__Scientific Use Cases__

- Simulate a compact nucleus embedded in extended galaxy light.
- Study how the PSF blends unresolved and resolved emission.
- Prepare controlled examples for testing host–nucleus decomposition.

__This Example__

This folder currently provides a simulator, not a complete inference workflow. It combines an extended
host with an intrinsic point component and produces an imaging dataset. It does not measure the nucleus's
intrinsic size. Read the operated-profile example for the related, already-PSF-blurred description.

![Point Source](../../../../images/features/point_source.png)

Conceptual illustration of added unresolved nuclear emission. The bright footprint is set by the response; it does not resolve the intrinsic nucleus.

__Separate the Compact and Extended Emission__

An unresolved nuclear component can sit inside an extended host galaxy. Their light is combined before
comparison with the observed image, so the PSF and host-light model both matter to their separation.

An intrinsic point source and an operated profile are related but different descriptions: the latter already
includes the relevant instrumental operation. Avoid applying that response twice.

__What This Folder Provides__

This feature currently supplies a simulator showing an extended host and an unresolved point component. Use it
to understand the model and the resulting data. It is not presented as a complete inference example; follow
the listed scripts for the functionality actually demonstrated.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
