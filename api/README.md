# API

This folder contains the code for the API component of the project.

## Description

The API folder contains the implementation of the API that serves as the interface for interacting with the model. It handles incoming requests, processes the data, and returns the predicted results. The API acts as a bridge between the client applications and the model.

## Dependencies

The following dependencies are required to run the API:

- redis 4.5.5
- fastapi 0.88.0
- uvicorn 0.22.0

## Installation

1.  Clone the repository

2.  Navigate to the api folder:

    ```bash
    cd api

    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the API, run the following command:

    python app.py

The API will be accessible at http://localhost:80 by default. You can send HTTP requests to the API endpoints to interact with the model.

## Docker Compose

To run the API component along with Redis and Model, you can use Docker Compose. The provided `docker-compose.yml` file defines the necessary services and configurations.

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
