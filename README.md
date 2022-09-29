# Introduction
The goal of this repository is to provide an example of how an ML repo can be organised. The focus is to make any experiment reproducible. The following files can be found in the repo template:

- `docker` this folder contains the docker file to build the container needed to run the experiment.
- `.devcontainer.json` Visual Studio Code will pick up this file when loading the repo locally using VS Code. This will automatically load the appropriate VS Code packages, launch the docker container and start Jupyter Notebook. 

# Work with the repo
### Locally using VS Code
The use case here is more interactive use as the local GPU is bound to be less powerful than the remote one. 

Just clone the folder and open with VS Code. Assuming Docker Desktop is running, this should launch the container. Do remember to switch the Jupyter kernel from local to the kernel running in the container. 

### Remotely via CLI on a server
Run the following docker commands to build the container and run the model into it:

    docker build -t example_ml docker
    docker run --gpus all -v ${PWD}:/tmp example_ml python /tmp/train.py

where:
- `-t` is the tag we use to refer to the container. Mind that if you run multiple of these simulteously, you should take care to use a unique tag. t
- `--gpus all` ensures the container has access to the gpu. 
- `-v ${PWD}:/tmp` mounts the local working folder inside the container at /tmp. This makes it possible to call the script using `/tmp/train.py`. 
- If you built the container earlier, just running docker run should suffice. 

Note that if you fire up the training script, be sure to launch this in tmux and detach. Otherwise, closing the ssh connection will kill the training proces. 

# TODO
- Integrate MLFlow for experiment tracking
- Look at running everything inside the container. 
	- Means we could ditch tmux
- Do we want to use a container registry? 
