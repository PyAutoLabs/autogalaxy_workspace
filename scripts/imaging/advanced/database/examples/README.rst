The ``database`` folder contains tutorials explaining how to use a SQLite3 database for managing large
suits of modeling results.

Files (Advanced)
----------------

- ``samples.py``: Loads the non-linear search results from the SQLite3 database and inspect the samples (e.g. parameter estimates, posterior).
- ``queries.py``: Query the database to get certain modeling results (e.g. all models where `effective_radius > 1.0`).
- ``models.py``: Inspect the models in the database (e.g. visualize the galaxy images).
- ``data_fitting.py``: Inspect the data-fitting results in the database (e.g. visualize the residuals).

