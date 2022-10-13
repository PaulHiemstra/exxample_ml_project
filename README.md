# Introduction
The goal of this repository is to provide an example of how an ML repo can be organised. The focus is to make any experiment a) reproducible and b) make it convinient to acces earlier results. The following aspects influence the reproducibility:

- Data. Using different data for training and/or validation leads to a different result. Using a version control system such as Git is only feasible for very small datasets, for large datasets a good database that support versioning is needed. Alternatively, for a relatively static dataset a simpler scheme using for example different folders might be sufficient. Fixing the data problem is not in scope of this particular repo, we assume that the data is correct and static from the pov of the repo. 
- Software. Different versions of say Keras will lead to different outcomes. The repo fixes this by using a docker container to run the software in. In the Dockerfile we can exactly specify which versions of the software should be loaded. This makes reproducing the exact software environment feasible as long as the base containers and libraries can be loaded. For really long term reproducibility you could opt to build the container and store this explicity. Although you might want to go for an actual container registry system at that stage. 
- Training configuration. Different settings for number of layers, number of neurons, hyperparameter settings, etc will influence the outcome you get. Fortunately, these settings are stored in code. So if you store that code correctly (e.g. in `train.py`), this should ensure reproducibility in combination with the environment we already have from the container. 

So, given the data the experiement can be repeated using a system that runs docker using the dockerfile and the train.py file. Changes over time can be saved using git commits, making it possible to trace the changes over time. However, this is probably not a very good solution when the number of trial-and-error gets really big. You could lose track of which things you already tried, especially when you transfer the experiment to someone else to work on. Here, a system such as MLflow probably makes a lot of sense. 

The following files can be found in the repo template:

- `docker` this folder contains the docker file to build the container needed to run the experiment.
- `.devcontainer.json` Visual Studio Code will pick up this file when loading the repo locally using VS Code. This will automatically load the appropriate VS Code packages, launch the docker container and start Jupyter Notebook. 
- `train.py` the training script that is used to train the model when running from the CLI when running on a server. 

# Work with the repo
- Focus is that the repo contains all that is needed to run the analysis:
    - Docker file to run the analysis from the CLI on a server. 
    - .devcontainer.json file that also uses this dockerfile to run the same analysis locally. 

### Locally using VS Code
The use case here is more interactive use as the local GPU is bound to be less powerful than the remote one. 

Just clone the folder and open with VS Code. Assuming Docker Desktop is running, this should launch the container. Do remember to switch the Jupyter kernel from local to the kernel running in the container. 

### Remotely via CLI on a server
Run the following docker commands to build the container and run the model into it:

    docker build -t example_ml docker
    docker run --gpus all -v ${PWD}:/tmp example_ml python train.py

where:
- `-t` is the tag we use to refer to the container. Mind that if you run multiple of these simulteously, you should take care to use a unique tag. t
- `--gpus all` ensures the container has access to the gpu. 
- `-v ${PWD}:/tmp` mounts the local working folder inside the container at /tmp. This makes it possible to call the script using `/tmp/train.py`. 
- If you built the container earlier, just running docker run should suffice. 

Note that if you fire up the training script, be sure to launch this in tmux and detach. Otherwise, closing the ssh connection will kill the training proces. A helper script, `run_script.bash`, builds the container, starts the container and runs the script you pass to it inside the container. Usage is:

    > run_script.bash <commands to run inside the container>

for example:

    > run_script.bash python train_mlflow.py

runs the example mlflow training. 

# MLFlow
### MLFlow Philosophy
TODO: How do we use MLFlow and why. 

### Exploring the MLFlow results
MLFlow is included in the container, and the UI can be started using the following command issued inside the container (using the VS Code terminal for example):

    > bash run_mlflow_ui.bash

Note that this does not simply run `mlflow ui`, but also [fixes the artifact path](https://github.com/mlflow/mlflow/issues/3144#issuecomment-782681919) that results from transporting the mlflow results between machines. After running the command you can open your browser and point it to `localhost:5000`. 

Note that the `mlruns` directory is not stored in git as it balloons in size quite quickly. If you want to transport the mlflow results between machines you can use tools like `rsync` to simply sync between multiple locations. The typical setup would be that you run the experiments on the server, but visualise them using `mlflow` locally. 

# TODO
- Integrate MLFlow for experiment tracking. This makes it a lot easier to track progress over time. When for each experiment you create a new commit in git you could reproduce all the results, this does not make it easy to quickly browse through earlier results. 
    - How much data does MLflow store? Is it feasible to save this is a github repo? **DONE: for realistic models it takes quite some space. So storing this in the github repo is not realistic.**
    - Does MLFlow integrate with frameworks as Keras tuner? 
    - Add other artifacts such as accuracy plots. **DONE: took some additional work as the artifact path was absolute inside the container where the experiment was run, not the container where the mlflow ui runs.**
    - Write a function that summarizes the architecture of a Keras model, and dump this as MLFlow information. Make smart choices how to do this. 
    - Look at MLFlow models, is it worth it to expand to that or just stick to the tracking API? 
- Look at running everything inside the container. 
	- Means we could ditch tmux
- Do we want to use a container registry? 
- Fix requirements.txt problem, I currently hardcode the packages in the docker file. 
- Kijk naar KerasTuner en andere optim frameworks. 
    - Keras Tuner en MLFlow?
