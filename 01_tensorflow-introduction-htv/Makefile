# Makefile for ML Engineering Track (WSL + GPU + TF Container)

WSL_ENV_NAME = wsl-gpu-env
PROJECT_PATH = /mnt/c/Users/hvanl/Documents/ml-engineering-track
TF_CONTAINER_SCRIPT = $(HOME)/tf_tf_container.sh

.PHONY: help setup wsl-jupyter tf-container tf-jupyter

help:
	@echo "Available targets:"
	@echo "  make setup        - Run setup_env.sh (create WSL env, kernel, TF container script)"
	@echo "  make wsl-jupyter  - Start Jupyter Lab in WSL GPU env on port 8888"
	@echo "  make tf-container - Start TensorFlow GPU container (shell only)"
	@echo "  make tf-jupyter   - Start TensorFlow GPU container + Jupyter Lab"

setup:
	chmod +x setup_env.sh
	./setup_env.sh

wsl-jupyter:
	@echo "Activating WSL GPU env and starting Jupyter Lab on port 8888..."
	@source $(HOME)/$(WSL_ENV_NAME)/bin/activate && \
	jupyter lab --ip=0.0.0.0 --no-browser --port=8888

tf-container:
	@echo "Starting TensorFlow GPU container shell..."
	@$(TF_CONTAINER_SCRIPT)

tf-jupyter:
	@echo "Starting TensorFlow GPU container and Jupyter Lab..."
	@$(TF_CONTAINER_SCRIPT) -c "jupyter lab --ip=0.0.0.0 --no-browser --allow-root"
