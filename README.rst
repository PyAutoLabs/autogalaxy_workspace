PyAutoGalaxy Workspace
=====================

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

|JOSS|

`Installation Guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Colab <https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace/blob/release/start_here.ipynb>`_
`HowToGalaxy <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_

Welcome to the **PyAutoGalaxy** Workspace!

Getting Started
---------------

You can get set up on your personal computer by following the installation guide on
our `readthedocs <https://pyautogalaxy.readthedocs.io/>`_.

Alternatively, you can try **PyAutoGalaxy** out in a web browser by going to
the `autogalaxy workspace on Colab <https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace>`_.

New Users
---------

New users should read the ``autogalaxy_workspace/start_here.ipynb`` notebook, which will give you a concise
overview of **PyAutoGalaxy**’s core features and API.

This can be done via a web browser by going to the following Google Colab link:

https://colab.research.google.com/github/Jammy2211/autogalaxy_workspace/blob/release/start_here.ipynb

Then checkout the `new user starting guide <https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html>`_
to navigate the workspace for your science case.

HowToGalaxy
-----------

For experienced scientists, the run through above will have been a breeze. Concepts surrounding galaxy structure and
morphology were already familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToGalaxy** Jupyter Notebook lectures are provide exactly this They are a 3+ chapter guide which thoroughly
take you through the core concepts of galaxy light profiles, teach you the principles ofthe  statistical techniques
used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

If this sounds like it suits you, checkout the `autogalaxy_workspace/notebooks/howtogalaxy` package now, its it
recommended you go here before anywhere else!

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoGalaxy** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoGalaxy** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoGalaxy**’s behaviour.
- ``dataset``: Where data is stored, including example datasets.
- ``output``: Where **PyAutoGalaxy** analysis and visualization outputs are written.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``guides``: Guides which introduce the core features of **PyAutoGalaxy**, including the core galaxy modeling API.
- ``imaging``: Examples for galaxy modeling using CCD imaging (e.g. Hubble, James Webb, Euclid).
- ``interferometer``: Examples for galaxies observed with an interferometer (e.g. ALMA, JVLA).
- ``multi``: Examples for modeling galaxies observed in multiple wavebands.
- ``howtogalaxy``: Conceptual tutorials introducing galaxy modeling and inference with **PyAutoGalaxy**.

The dataset packages (e.g. ``imaging``, ``interferometer`` and ``multi``) include the following types of examples:

- ``modeling``: Performing galaxy modeling using that type of data.
- ``simulators``: Simulating galaxy images.
- ``fit``: How to compute residuals, chi-squared maps, and likelihoods.
- ``data_preparation``: Preparing real datasets for **PyAutoGalaxy** analysis.
- ``features``: Advanced modeling features (e.g. Multi Gaussian Expansion, priors, constraints).
- ``likelihood_function``: Step-by-step guides to the likelihood function.

The ``guides`` package contains important subpackages, including:

- ``results``: How to load, inspect, and analyze results from many galaxy fits efficiently.
- ``modeling``: Customizing galaxy models and building automated modeling pipelines.
- ``plot``: How to visualize galaxy images, profiles, and residuals.

The files ``README.rst`` distributed throughout the workspace describe what is in each folder.

Contribution
------------

To make changes in the tutorial notebooks, please make changes in the corresponding Python files (``.py``)
present in the ``scripts`` folder of each chapter. The marker ``# %%`` alternates between code cells and
markdown cells.

Support
-------

Support for installation issues, help with galaxy modeling, and using **PyAutoGalaxy** is available by
`raising an issue on the autogalaxy_workspace GitHub page <https://github.com/Jammy2211/autogalaxy_workspace/issues>`_
or by joining the **PyAutoGalaxy** Slack channel, where we also provide the latest updates.

Slack is invitation-only, so if you'd like to join send an email via
`GitHub <https://github.com/Jammy2211>`_ requesting an invite.
