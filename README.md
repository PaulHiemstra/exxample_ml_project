# Introduction
The goal of this repository is to provide an example of how an ML repo can be organised. The focus is to make any experiment reproducible. The following files can be found in the repo template:

- `docker` this folder contains the docker file to build the container needed to run the experiment.
- `.devcontainer.json` Visual Studio Code will pick up this file when loading the repo locally using VS Code. This will automatically load the appropriate VS Code packages, launch the docker container and start Jupyter Notebook. 

# TODO
- Integrate MLFlow for experiment tracking