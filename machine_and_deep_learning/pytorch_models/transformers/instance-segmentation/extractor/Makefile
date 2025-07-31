# Name of your docker image
IMAGE_NAME=prime
# Docker container name
CONTAINER_NAME=prime-workspace

# Dockerfile location
DOCKERFILE_PATH=./Dockerfile

# Port mapping for the docker container
PORT_MAP=1127:80

# Base url to download content
BASE_URL="https://staging-api-central.jebi.ai/api/v1/jaid/get-object"

# Target: build
# Description: Builds a Docker image using the specified IMAGE_NAME and DOCKERFILE_PATH.
.SILENT: build
build:
	@echo "Building Docker image $(IMAGE_NAME)"
	chmod u+x ./scripts/download.sh
	./scripts/download.sh
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE_PATH) .

