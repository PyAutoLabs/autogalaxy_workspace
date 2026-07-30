The `plot` folder contains scripts which show how to use **PyAutoGalaxy** built-in visualization tools.

The API provides a simple interface with matplotlib for making plots that does not require the user to
write any matplotlib code themselves.

This is illustrated in the `start_here` file, where new users should begin.

Dataset and fit plotting (e.g. subplots of imaging data, fits and inversions) lives in each dataset package's
own `plot.py`, for example `scripts/imaging/plot.py`.

# Files

- `start_here`: An introduction to plotting and visualization (RECOMMENDED READ).
- `plotters`: Object-by-object plotting figures (e.g. galaxies, light profiles, grids).
- `searches`: Visualization of non-linear search results (e.g. cornerplots, walker trajectories).
- `visuals`: Overlaying visuals (e.g. positions, grids) on plots.
- `simulator`: Simulates the `sersic_x2` dataset used by the plot guides.
