## Census Income Prediction API

This project provides an end-to-end machine learning workflow for predicting whether an individual's income exceeds $50K per year using U.S. Census data. It includes data preprocessing, model training, slice-based evaluation, automated testing, CI/CD, and deployment of a FastAPI inference service.

## Project Overview

The repository includes:

- data preprocessing and model training code
- serialized model artifacts for inference
- slice-based model performance evaluation
- a FastAPI application for online predictions
- unit tests for model and API functionality
- GitHub Actions for continuous integration
- cloud deployment configuration using Render

## Environment Setup

Working in a command-line environment is recommended for easier use with Git, DVC, and deployment tooling. On Windows, WSL is recommended for a smoother development experience.

# Option 1: Using pip and venv

Ensure Python 3.13 is installed, then run:

bash

python3.13 -m venv .venv
source .venv/bin/activate
pip install -r starter/requirements.txt
On Windows, activate the environment with:

applescript

.venv\Scripts\activate
# Option 2: Using conda

Create and activate a conda environment:

conda create -n census-api "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge
conda activate census-api
If needed, install Git separately through conda or your system package manager.

## Data

The model is trained on the Census Income dataset. Data versioning is managed with DVC to support reproducibility and traceability.

## Model Training

The training pipeline:

- loads and cleans the census dataset
- processes categorical and numerical features
- trains a classification model
- evaluates model performance
- saves model artifacts for inference
To train the model, run:

python starter/starter/train_model.py

Generated model artifacts are stored in:

starter/model/

These artifacts are used by the FastAPI application during inference.

## Model Evaluation

The project includes:

- overall model performance metrics 
- slice-based evaluation across categorical features
- a model card documenting intended use, assumptions, and limitations
Slice-based metrics are saved to:

starter/slice_output.txt

## API

The application exposes a FastAPI service for inference.

# Endpoints

GET /
Returns a welcome or health-check message.

POST /predict
Accepts a single census record and returns an income prediction.

# The API uses:

- type hints
- a Pydantic request schema
- serialized model artifacts for prediction

# Running the API Locally

Start the API server with:

cd starter
uvicorn main:app --reload
If needed on Windows, you can specify a port explicitly:

uvicorn main:app --reload --port 8001

Once running, open the interactive API docs at:

http://127.0.0.1:8000/docs
Or, if using port 8001:

http://127.0.0.1:8001/docs

## Testing

The project includes unit tests for both the model code and API behavior.

Run all tests with:

pytest
Run lint checks with:

flake8

## CI/CD

GitHub Actions is used for continuous integration. On each push, the workflow validates the codebase by running:

- lint checks
- model tests
- model artifact generation
- API tests
The application is deployed on Render with automatic deployment enabled after CI checks pass.

## Deployment

The production API is hosted on Render.

# Live API URL:
https://nd0821-c3-starter-code-war4.onrender.com

# Example Inference Request

You can test the deployed API using Python and the requests library.

''' 
import requests

url = "https://nd0821-c3-starter-code-war4.onrender.com/predict"

payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
'''

## Repository Structure

starter/
├── main.py
├── requirements.txt
├── model/
├── slice_output.txt
├── starter/
│   ├── data.py
│   ├── model.py
│   ├── train_model.py
│   └── ...

## Links

GitHub Repository: https://github.com/lochan1209/nd0821-c3-starter-code.git
Live API: https://nd0821-c3-starter-code-war4.onrender.com