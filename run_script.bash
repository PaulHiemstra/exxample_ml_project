docker build -t example_ml docker
docker run --gpus all -v ${PWD}:/tmp example_ml $@
