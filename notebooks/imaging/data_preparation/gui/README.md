The `imaging/data_preparation/gui` package holds interactive tools for marking up an imaging dataset by hand:
painting masks with the mouse (`Scribbler`) and clicking positions (`Clicker`). None of them are required; each
writes one product into the dataset's folder that the modeling scripts can then load.

# Recommended Order

1. `mask_extra_galaxies`: paint the CONTAMINANTS to remove (neighbouring galaxies, stars, artefacts). Do this
   first, so that later steps can show the mask and a contaminant is not mistaken for part of the galaxy.
2. `mask`: paint the region of the image to fit, if the default circular mask is not suitable.

`light_centre` and `extra_galaxies_centres` are independent of that order.

Every mask tool can be reopened on its own product to refine it rather than redraw it (`refine_existing = True`
in each script).

# Where Files Belong

Every GUI reads the dataset from, and writes its product back into, that dataset's own folder under `dataset/`, next
to the `data.fits` / `noise_map.fits` / `psf.fits` it describes. Nothing goes in `output/`, which is reserved for
model-fit results.

```
dataset/imaging/<dataset_name>/
    data.fits                      image                              (data_preparation/examples)
    noise_map.fits                 noise-map                          (data_preparation/examples)
    psf.fits                       PSF                                (data_preparation/examples)
    mask_extra_galaxies.fits       contaminants, True = masked        (gui/mask_extra_galaxies)      step 1
    mask_gui.fits                  region to fit, True = masked       (gui/mask)                     step 2
    light_centre.json              galaxy light centre                (gui/light_centre)
    extra_galaxies_centres.json    extra galaxy centres               (gui/extra_galaxies_centres)
    info.json                      redshift etc.                      (by hand)
```

Every `.fits` mask is stored in `Mask2D`'s own convention (`True` = excluded from the fit) and loads with
`ag.Mask2D.from_fits(file_path=..., pixel_scales=...)`; the scripts note where a painted region is inverted before
saving. Centres load with `ag.from_json`.

# Files

- `mask_extra_galaxies`: paint the extra galaxies / contaminants to remove (step 1).
- `mask`: paint the region of the image to fit (step 2).
- `light_centre`: click the galaxy's light centre.
- `extra_galaxies_centres`: click the centres of the extra galaxies.

# Keys

All painting GUIs share the same keys: `1` white brush adds, `2` black brush erases, `=` / `-` grow / shrink the
brush, `z` undoes the last stroke, `Esc` finishes.
