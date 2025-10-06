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

try:
    import numba
except ModuleNotFoundError:
    input(
        "##################\n"
        "##### NUMBA ######\n"
        "##################\n\n"
        """
        Numba is not currently installed.
        
        Numba is a library which makes PyAutoGalaxy run a lot faster. Certain functionality is disabled without numba
        and will raise an exception if it is used.
        
        If you have not tried installing numba, I recommend you try and do so now by running the following 
        commands in your command line / bash terminal now:
        
        pip install --upgrade pip
        pip install numba
        
        If your numba installation raises an error and fails, you should go ahead and use PyAutoGalaxy without numba to 
        decide if it is the right software for you. If it is, you should then commit time to bug-fixing the numba
        installation. Feel free to raise an issue on GitHub for support with installing numba.

        A warning will crop up throughout your *PyAutoGalaxy** use until you install numba, to remind you to do so.
        
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
    PyAutoGalaxy is now set up and you can begin exploring the workspace. New users should follow the
    'start_here.ipynb' notebook, which gives an overview of **PyAutoGalaxy** and the workspace.
    
    Examples are provided as both Jupyter notebooks in the 'notebooks' folder and Python scripts in the 'scripts'
    folder. It is up to you how you would prefer to use PyAutoGalaxy. With these folders, you can find the following
    packages:
    
    - howtogalaxy: Jupyter notebook lectures introducing beginners to strong gravitational galaxying, describing how to
     perform scientific analysis of galaxy data and detailing the PyAutoGalaxy API. A great starting point for new users!
    
    - overview: An overview of all PyAutoGalaxy's main features.
    
    - imaging: Examples for analysing and simulating CCD imaging data of a strong galaxy.

    - interferometer: Examples for analysing and simulating interferometer data of a strong galaxy.
     
     - plot: An API reference guide of all of PyAutoGalaxy's plotting and visualization tools.
     
     - results: Tutorials on how to use PyAutoGalaxy's results after fitting a galaxy.
     
     - misc: Miscellaneous scripts for specific galaxy analysis.
     
    The `chaining` folders are for experienced users. The example scripts and HowToGalaxy lectures will guide new users 
    to these modules when they have sufficient experience and familiarity with PyAutoGalaxy.
    
    [Press Enter to continue]
    """
)
