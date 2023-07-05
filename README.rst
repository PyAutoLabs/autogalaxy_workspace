PyAutoGalaxy Workspace
======================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02825/status.svg
   :target: https://doi.org/10.21105/joss.02825

|binder| |JOSS|

`Installation Guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/release?filepath=introduction.ipynb>`_ |
`HowToGalaxy <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_

Welcome to the **PyAutoGalaxy** Workspace. You can get started right away by going to the `autogalaxy workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautogalaxy.readthedocs.io/>`_.

Getting Started
---------------

We recommend new users begin by looking at the following notebooks: 

- ``notebooks/overview``: Examples giving an overview of **PyAutoGalaxy**'s core features.
- ``notebooks/howtogalaxy``: Detailed step-by-step Jupyter notebook lectures on how to use **PyAutoGalaxy**.

Installation
------------

If you haven't already, install `PyAutoGalaxy via pip or conda <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_.

Next, clone the ``autogalaxy workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autogalaxy_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autogalaxy_workspace
   git clone https://github.com/Jammy2211/autogalaxy_workspace --depth 1
   cd autogalaxy_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoGalaxy** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoGalaxy** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoGalaxy**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed.
- ``output``: Where the **PyAutoGalaxy** analysis and visualization are output.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``overview``: Examples giving an overview of **PyAutoGalaxy**'s core features.
- ``howtogalaxy``: Detailed step-by-step Jupyter notebook lectures on how to use **PyAutoGalaxy**.

- ``imaging``: Examples for analysing and simulating CCD imaging data (e.g. Hubble, Euclid).
- ``interferometer``: Examples for analysing and simulating interferometer datasets (e.g. ALMA, JVLA).
- ``multi``: Modeling multiple datasets simultaneously (E.g. multi-wavelength imaging, imaging and interferometry).

- ``plot``: An API reference guide for **PyAutoGalaxy**'s plotting tools.
- ``misc``: Miscellaneous scripts for specific galaxy analysis.

Inside these packages are packages titled ``advanced`` which only users familiar **PyAutoGalaxy** should check out.

In the ``imaging``, ``interferometer``, and ``multi`` folders you'll find the following packages:

- ``modeling``: Examples of how to fit a galaxy model to data via a non-linear search.
- ``simulators``: Scripts for simulating realistic imaging and interferometer data of strong galaxyes.
- ``data_preparation``: Tools to preprocess ``data`` before an analysis (e.g. convert units, create masks).
- ``results``: Examples using the results of a model-fit.
- ``advanced``: Advanced modeling scripts which use **PyAutoGalaxy**'s advanced features.


The files ``README.rst`` distributed throughout the workspace describe what is in each folder.

Getting Started
---------------

We recommend new users begin with the example notebooks / scripts in the *overview* folder and the **HowToGalaxy**
tutorials.

Workspace Version
-----------------

This version of the workspace is built and tested for using **PyAutoGalaxy v2023.6.18.3**.

HowToGalaxy
-----------

Included is the ``HowToGalaxy`` lecture series, which provides an introduction to strong gravitational
galaxy modeling. It can be found in the workspace & consists of 5 chapters:

- ``Introduction``: An introduction to galaxy morphology & **PyAutoGalaxy**.
- ``Galaxy Modeling``: How to model strong galaxyes, including a primer on Bayesian analysis and model-fitting via a non-linear search .
- ``Search Chaining``: Chaining non-linear searches together to build model-fitting pipelines & tailor them to your own science case.
- ``Pixelizations``: How to perform pixelized reconstructions of a galaxy.

Contribution
------------
To make changes in the tutorial notebooks, please make changes in the corresponding python files(.py) present in the
``scripts`` folder of each chapter. Please note that  marker ``# %%`` alternates between code cells and markdown cells.

Support
-------

Support for installation issues, help with galaxy modeling and using **PyAutoGalaxy** is available by
`raising an issue on the autogalaxy_workspace GitHub page <https://github.com/Jammy2211/autogalaxy_workspace/issues>`_. or
joining the **PyAutoGalaxy** `Slack channel <https://pyautogalaxy.slack.com/>`_, where we also provide the latest updates on
**PyAutoGalaxy**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
