# Automated Product Categorization for E-commerce with AI

Automated Product Categorization for E-commerce with AI is a project aimed at automatically categorizing products in an e-commerce dataset using AI and natural language processing techniques.
Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Structure](#structure)
- [Usage](#usage)

## Project Overview

Automated product categorization is a critical task in e-commerce to enhance product search, recommendation systems, and user experience. This project leverages AI and NLP techniques to automatically categorize products based on their names and descriptions. It utilizes a BERT-based model for text encoding and employs seven classifiers for the hierarchical categorization levels.

The project consists of the following components:

- **Data Preprocessing**: The dataset is loaded from an external source and processed to filter categories and assign a "Other" category to low-frequency categories.
- **Model Training**: The BERT model is trained on the preprocessed dataset, incorporating seven classifiers for the hierarchy levels.
- **API Service**: A RESTful API service is provided to receive product text inputs, perform categorization using the trained model, and return the predicted categories.
- **Dockerization**: The project can be deployed using Docker containers, ensuring easy setup and portability.

## Installation

To install and set up the project environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/daidaragona/ecommerce.git
   ```

2. Navigate to the project directory:

   ```bash
   cd automated-product-categorization
   ```

## Structure

### Folders structure:

- ### api

  Contains the API service code.

- ### model

  Trained model, data and pre-preocessing needed.

- ### notebooks

  Jupyter notebooks for exploration, analysis and training.

- ### reports

  Generated analysis as HTML, PDF, or other formats.

### files:

- `.env:` Defines environment variables for the project (UID and GID)
- `.gitignore:` List of files and directories to ignore in version control
- `requirements.txt:` Python dependencies for the project
- `docker-compose.yml:` docker compose file to start api, model and redis containers.

### Folders diagram

```
├── api
├── model
│   └── weights
├── notebooks
├── reports
├── .env
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Usage

To use the project, follow these steps:

- Start the services:

      docker-compose up --build -d

Remember to run this command from the root directory of the project where the docker-compose.yml file is located.

Send HTTP requests to the API endpoint with the product text data to receive predicted categories.

Example API endpoint: http://localhost:80/predict

Example request body:

    {
      "text": "Product name and description"
    }

The API will return the predicted categories for the given product text.
