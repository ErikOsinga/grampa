# Gaussian Random-fields Approximation for Magnetic-fields with Python Algorithms (GRAMPA)

## Overview
GRAMPA is a Python-based tool for modeling magnetic fields as Gaussian random fields following a specified power spectrum and electron density profile, as described in Murgia et al. (2004) and originally proposed by Tribble (1991). This framework is designed primarily for analyzing Faraday rotation experiments in galaxy clusters but can be applied more generally to other magnetized plasmas, depending on the chosen parameters.

## Getting Started
To begin using GRAMPA, refer to the Jupyter notebook `examples/example_cluster_Bfield.ipynb`, which provides an example workflow to help users understand the implementation and features of the code.

## Citation
If you use this code in your research, please cite the following works:
- **Osinga et al. 2022**: [A&A, 665, A71](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..71O/abstract)
- **Osinga et al. 2025**: [A&A, 694, A44](https://ui.adsabs.harvard.edu/abs/2025A%26A...694A..44O/abstract)

If your work involves lognormal density fluctuations, please also cite:
- **Khadir et al. 2025 (in prep.)**
- **PyFC**: [PyFC on PiWheels](https://www.piwheels.org/project/pyfc/)

## License
This software is provided under an open-source license. Please check the GitHub repository for details.

## Contact
For questions or feedback, please reach out to the authors or open an issue on the [Github repository](https://github.com/ErikOsinga/magneticfields/)


## Developing

Pull requests are welcome for users that want to add features!

GRAMPA uses a set of developer tools that can be installed with

`git clone git@github.com:ErikOsinga/grampa.git`
`cd grampa`
`pip install '.[dev]'`

These tools can run upon git commit by using

`pre-commit install`

## Versioning

GRAMPA will follow the [Conventional Commit Message format:](https://www.conventionalcommits.org/en/v1.0.0/) for versioning

- For a feature (MINOR VERSION UPDATE): (e.g. git commit -m "feat(optional scope): description")
- For a bugfix (PATCH VERSION UPDATE): (e.g. git commit -m "fix(optional scope): description")
- For a breaking change (!) (MAJOR VERSION UPDATE): (e.g. git commit -m "featorfix!(optional scope): description")
