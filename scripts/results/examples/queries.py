"""
Database: Queries
=================

Suppose we have the results of many fits and we only wanted to load and inspect a specific set
of model-fits (e.g. the results of `start_here.py`). We can use querying tools to only load the results we are
interested in.

This includes support for advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset)
can be loaded.

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autogalaxy as ag

"""
__Loading From Hard-disk__

Results can be loaded from hard disk using the `Aggregator` object (see the `start_here.py` script for a description of
what the `Aggregator` does if you have not seen it!).
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=Path("output") / "results_folder",
)

"""
__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results. 

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.

__Unique Tag__

We can use the `Aggregator` to query the results and return only specific fits that we are interested in. We first 
do this using the `unique_tag` which we can query to load the results of a specific `dataset_name` string we 
input into the model-fit's search. 

By querying using the string `simple__1` the model-fit to only the second dataset is returned:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "simple")
samples_gen = agg_query.values("samples")

"""
As expected, this list now has only 1 `SamplesNest` corresponding to the second dataset.
"""
print("Directory Filtered DynestySampler Samples: \n")
print("Total Samples Objects via unique tag = ", len(list(samples_gen)), "\n\n")

"""
If we query using an incorrect dataset name we get no results:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

"""
__Search Name__

We can also use the `name` of the search used to fit to the model as a query. 

In this example, all three fits used the same search, which had the `name` `results`. Thus, using it as a 
query in this example is somewhat pointless. However, querying based on the search name is very useful for model-fits
which use search chaining (see chapter 3 **HowToGalaxy**), where the results of a particular fit in the chain can be
instantly loaded.

As expected, this query contains all 3 results.
"""
name = agg.search.name
agg_query = agg.query(name == "results")
print("Total Queried Results via search name = ", len(agg_query), "\n\n")

"""
__Model Queries__

We can also query based on the model fitted. 

For example, we can load all results which fitted an `Sersic` model-component, which in this simple example is 
all 3 model-fits.

The ability to query via the model is extremely powerful. It enables a user to fit many models to large samples 
of galaxies efficiently load and inspect the results. 

[Note: the code `agg.model.galaxies.galaxy.bulge` corresponds to the fact that in the `Model` we named the model 
components `galaxies`, `galaxy` and  `bulge`. If the `Model` had used a different name the code below would change 
correspondingly. Models with multiple galaxies are therefore easily accessed.]
"""
galaxy = agg.model.galaxies.galaxy
agg_query = agg.query(galaxy.bulge == ag.lp_linear.Sersic)
print("Total Samples Objects via `Sersic` model query = ", len(agg_query), "\n")

"""
We can also query the model on whether a component is None, which is not the case for model we created in tutorial 1
but can be useful for more complex model fitting. 

When performing model comparison with search-chaining pipelines, it is common for certain components to be included or 
omitted via a `None`. Querying via `None` therefore allows us to load the results of different model-fits.
"""
source = agg.model.galaxies.galaxy
agg_query = agg.query(galaxy.disk == None)
print("Total Samples Objects via `Sersic` model query = ", len(agg_query), "\n")

"""
Queries using the results of model-fitting are also supported. Below, we query to find all fits where the 
inferred value of `sersic_index` for the `Sersic` of the source's bulge is less than 3.0 (which returns only 
the first of the three model-fits).
"""
bulge = agg.model.galaxies.galaxy.bulge
agg_query = agg.query(bulge.sersic_index < 3.0)
print(
    "Total Samples Objects In Query `galaxy.bulge.sersic_index < 3.0` = ",
    len(agg_query),
    "\n",
)

"""
__Logic__

Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted an `Sersic` bulge model AND (using the & symbol) inferred a value of effective radius of 
greater than 3.0 for the bulge. 

The OR logical clause is also supported via the symbol |.
"""
bulge = agg.model.galaxies.galaxy.bulge
agg_query = agg.query((bulge == ag.lp_linear.Sersic) & (bulge.effective_radius > 3.0))
print(
    "Total Samples Objects In Query `Sersic and effective_radius > 3.0` = ",
    len(agg_query),
    "\n",
)

"""
Finished.
"""
