# Model

This folder contains the code for the model component of the project.

## Description

The Model folder houses the implementation of the machine learning model used in the project. It includes the training and prediction logic.

## Dependencies

The following dependencies are required to run the model:

-transformers 4.29.2  
-lightning 2.0.2  
-torch 2.0.1  
-pandas 2.0.2  
-redis 4.5.5  
-gdown 4.6.0

# Usage

## Docker Compose

To run the model component along with its dependencies (Redis and API), you can use Docker Compose. The provided `docker-compose.yml` file defines the necessary services and configurations.

To start the services, navigate to the root directory of the project and run the following command:

    docker-compose up --build -d

The `--build` flag ensures that the Docker images are rebuilt if any changes have been made to the code or dependencies.

The services defined in the `docker-compose.yml` file are as follows:

- api: This service runs the API component of the project. It builds the FastAPI image using the Dockerfile in the api folder and exposes `port 80`. It depends on the Redis and model services.

- redis: This service uses the official Redis image and exposes `port 6379`. It is a dependency for the API and model services.

- model: This service runs the ML model component. It builds the ML service image using the Dockerfile in the model folder. It depends on the Redis service.

After running docker-compose up, the services will be started in the background. You can access the API at http://localhost:80.

To stop the services, use the following command:

    docker-compose down

Remember to run these commands from the root directory of the project where the docker-compose.yml file is located.
