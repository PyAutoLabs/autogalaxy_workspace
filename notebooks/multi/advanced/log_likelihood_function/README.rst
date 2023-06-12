The ``log_likelihood_function`` folder contains example scripts showing step-by-step visual guides to how likelihoods
are evaluated in **PyAutoGalaxy**.

The likelihood function for fitting multiple datasets simply evaluates the likelihood function of each dataset
individually and sum the likelihoods together.

Therefore, no specific likelihood function description is given in the ``multi``` package and readers should instead
refer to:

- ``autogalaxy_workspace/*/imaging/log_likelihood_function``.
- ``autogalaxy_workspace/*/interferometer/log_likelihood_function``.