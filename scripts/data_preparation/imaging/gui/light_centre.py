"""
GUI Preprocessing: Lens Light Centre
====================================

This tool allows one to input the galaxy light centre(s) of a galaxy(es) via a GUI, which can be used as a fixed
value in pipelines.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt
from matplotlib import pyplot as plt

"""
__Dataset__

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/simple__sersic`.
"""
dataset_name = "simple__sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the image which we will use to mark the galaxy light centre.
"""
data = ag.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=pixel_scales
)

"""
__Search Box__

When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""
search_box_size = 5

"""
__Clicker__

Set up the `Clicker` object from the `clicker.py` module, which monitors your mouse clicks in order to determine
the galaxy light centres.
"""
clicker = ag.Clicker(
    image=data, pixel_scales=pixel_scales, search_box_size=search_box_size
)

"""
Set up the clicker canvas and load the GUI which you can now click on to mark the galaxy light centres.
"""
n_y, n_x = data.shape_native
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(data.native, cmap="jet", extent=ext)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", clicker.onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

"""
Use the results of the Clicker GUI to create the list of lens light centres.
"""
light_centres = ag.Grid2DIrregular(values=clicker.click_list)

"""
__Output__

Now lets plot the image and lens light centres, so we can check that the centre overlaps the brightest pixel in the
lens light.
"""
visuals = aplt.Visuals2D(mass_profile_centres=light_centres)

array_2d_plotter = aplt.Array2DPlotter(
    array=data, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D()
)
array_2d_plotter.figure_2d()

"""
Output this image of the galaxy light centres to a .png file in the dataset folder for future reference.
"""
array_2d_plotter = aplt.Array2DPlotter(
    array=data,
    visuals_2d=visuals,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, filename="light_centres", format="png")
    ),
)
array_2d_plotter.figure_2d()

"""
Output the galaxy light centres to a .json file in the dataset folder, so we can load them in modeling scripts.
"""
ag.output_to_json(
    obj=light_centres,
    file_path=path.join(dataset_path, "light_centre.json"),
)

"""
Finish.
"""
