input(
    "############################################\n"
    "### WELCOME TO THE AUTOGALAXY WORKSPACE ###\n"
    "############################################\n\n"
    "This script runs a few checks to ensure PyAutoGalaxy is set up correctly.\n"
    ""
    "Once they pass, you should read through the autogalaxy_workspace/start_here.ipynb notebook "
    "(or autogalaxy_workspace/start_here.py script if you prefer Python scripts) to get a full overview of PyAutoGalaxy.\n\n"
    "\n"
    "############################################\n"
    "### AUTOGALAXY WORKSPACE WORKING DIRECTORY ###\n"
    "############################################\n\n"
    """
    PyAutoGalaxy assumes that the `autogalaxy_workspace` directory is the Python working directory. 
    This means that, when you run an example script, you should run it from the `autogalaxy_workspace` 
    as follows:
    
    
    cd path/to/autogalaxy_workspace (if you are not already in the autogalaxy_workspace).
    python3 scripts/modeling/imaging/light_parametric.py


    The reasons for this are so that PyAutoGalaxy can:
     
    - Load configuration settings from config files in the `autogalaxy_workspace/config` folder.
    - Load example data from the `autogalaxy_workspace/dataset` folder.
    - Output the results of models fits to your hard-disk to the `autogalaxy/output` folder. 

    If you have any errors relating to importing modules, loading data or outputting results it is likely because you
    are not running the script with the `autogalaxy_workspace` as the working directory!
    
    [Press Enter to continue]"""
)

input(
    "\n"
    "###############################\n"
    "##### MATPLOTLIB BACKEND ######\n"
    "###############################\n\n"
    """
    We`re now going to plot an image in PyAutoGalaxy using Matplotlib, using the backend specified in the following
    config file (the backend tells Matplotlib where to render the plot)"


    autogalaxy_workspace/config/visualize/generag.yaml -> [general] -> `backend`


    The default entry for this is `default` (check the config file now). This uses the default Matplotlib backend
    on your computer. For most users, pushing Enter now will show the figure without error.

    However, we have had reports that if the backend is set up incorrectly on your system this plot can either
    raise an error or cause the `welcome.py` script to crash without a message. If this occurs after you
    push Enter, the error is because the Matplotlib backend on your computer is set up incorrectly.

    To fix this in PyAutoGalaxy, try changing the backend entry in the config file to one of the following values:"

    backend=TKAgg
    backend=Qt5Agg
    backeknd=Qt4Agg

    NOTE: If a matplotlib figure window appears, you may need to close it via the X button and then press 
    enter to continue the script.

    [Press Enter to continue]
    """
)

import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(
    shape_native=(50, 50),
    pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

sersic_light_profile = ag.lp.Exponential(
    centre=(0.3, 0.2), ell_comps=(0.2, 0.0), intensity=0.05, effective_radius=1.0
)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.figures_2d(image=True)

input(
    "\n"
    "#####################\n"
    "## LIGHT PROFILES ###\n"
    "#####################\n\n"
    """
    The image displayed on your screen shows a `LightProfile`, the object PyAutoGalaxy uses to represent the 
    luminous emission of galaxies. 

    [Press Enter to continue]
    """
)

input(
    "\n"
    "###########################\n"
    "##### WORKSPACE TOUR ######\n"
    "###########################\n\n"
    """
    PyAutoGalaxy is now set up and you can begin exploring the workspace. 
    
    We recommend new users begin by following the 'start_here.ipynb' notebook, which gives an overview 
    of **PyAutoGalaxy** and the workspace. 
    
    This will also guide you through where to go next in the workspace depending on your scientific interests.
    
    A full description of the workspace can be found in the `autogalaxy_workspace/README.rst` file and on the 
    PyAutoGalaxy readthedocs website.
    
    [Press Enter to continue]
    """
)
