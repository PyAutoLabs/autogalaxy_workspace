# PyAutoGalaxy Workspace

Example scripts and notebooks for modeling galaxy structure and morphology with **PyAutoGalaxy**. Start with an introductory example, then explore the tools for your own data.

## Getting Started

### Explore It Yourself

- **[Try it in your browser](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.13.1/start_here.ipynb)** — run the introductory notebook on Google Colab without installing anything.
- **[Run it locally](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html)** — follow the installation guide, then open `start_here.ipynb`.
- **[Browse examples with images](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/markdown/README.md)** — read worked examples and view their results directly on GitHub.

See [Find Your Examples](#find-your-examples) below to choose where to start.

### Explore with AI

The [**PyAutoGalaxy AI Assistant**](https://github.com/PyAutoLabs/autogalaxy_assistant) helps you learn galaxy modeling, run examples and analyse your own data through conversation. Visit its README for setup instructions and a starting prompt.

## Find Your Examples

Examples are available as Python scripts in `scripts/` and Jupyter notebooks in `notebooks/`.

### Galaxy Systems

![A single galaxy, blended galaxies and a cluster field.](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_2/galaxy_systems.png)

For a single galaxy, choose your data type below. To fit galaxies together, start with [multi_galaxy/](scripts/multi_galaxy) for blended light or [cluster/](scripts/cluster) for a cluster field.

### Dataset Type

![Galaxy imaging, interferometer sampling and multi-band imaging.](https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_2/dataset_types.png)

Match your observations to [imaging/](scripts/imaging), [interferometer/](scripts/interferometer) or [multi_dataset/](scripts/multi_dataset) for multiple wavebands. Each has a `start_here` example to get you going.

*AI-generated illustrations, not observations. Galaxy panels use different angular scales. See the [new user guide](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) for details.*

### Browse All Examples

| I want to… | Start here |
|---|---|
| Learn the API, customise models or inspect results | [Guides](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/guides) |
| Model galaxies in imaging data | [Imaging](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/imaging) |
| Model interferometer observations | [Interferometry](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/interferometer) |
| Fit observations in multiple wavebands | [Multiple datasets](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/multi_dataset) |
| Fit several blended galaxies together | [Multiple galaxies](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/multi_galaxy) |
| Measure galaxy structure by fitting ellipses | [Ellipse fitting](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/ellipse) |
| Model the light of galaxies in a cluster | [Clusters](https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/scripts/cluster) |

Each folder has a README explaining its examples. You’ll find sample data in `dataset/`; analysis results are written to `output/`.

## Learn Galaxy Modeling

[**HowToGalaxy**](https://github.com/PyAutoLabs/HowToGalaxy) teaches galaxy modeling and Bayesian inference through worked notebooks. Start here if you’re new to the subject or would like more explanation alongside the workspace examples.

## Community and Contributing

Ask questions, suggest improvements and share your work in [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions), or chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

To contribute examples or explore community projects, visit the [PyAutoGalaxy community page](https://pyautogalaxy.readthedocs.io/en/latest/general/community.html).

## Citations

If you use PyAutoGalaxy in your research, see the [citation guide](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/CITATIONS.md).

## License

See the [PyAutoGalaxy license](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/LICENSE).
