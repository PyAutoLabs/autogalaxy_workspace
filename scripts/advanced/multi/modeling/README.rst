The ``modeling`` folder contains example scripts showing how to fit a galaxy model to multiple imaging datasets:

Notes
-----

The ``multi`` package extends the ``imaging`` package and readers should refer to the ``imaging`` package for
descriptions of how to customize the non-linear search, the fit settings, etc.

These scripts show how to perform galaxy modeling but only give a brief overview of how to analyse
and interpret the results a galaxy model fit. A full guide to result analysis is given at ``autogalaxy_workspace/*/results``.

Files (Beginner)
----------------

- ``start_here.py``: A simple example illustrating how to fit a galaxy model to multiple datasets.

Folders (Beginner)
------------------

- ``examples``: Example modeling scripts for multiple datasets for users familiar.
- ``customize``: Customize aspects of a model-fit, (e.g. priors, the imaging mask).
- ``searches``: Using other non-linear searches (E.g. MCMC, maximum likelihood estimators).
