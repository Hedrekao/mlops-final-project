# Fake vs Real Job Postings – MLOps Project

## Project Description

This repository contains the exam project for the course **02476 - Machine Learning Operations** at DTU. The purpose of the project is to apply the tools and concepts taught throughout the course to a self-chosen machine learning problem, with a strong focus on MLOps practices rather than model performance.

## Overall Goal of the Project
The overall goal of this project is to use **natural language processing (NLP)** to solve a binary classification task, where the objective is to predict whether a given job posting is **real or fake**. Fraudulent job advertisements are a common issue on online job platforms, and this project aims to simulate how such a problem could be handled in a real-world production environment.

The main focus of the project is not to build the most accurate classifier, but to design and implement a robust and reproducible MLOps pipeline. This includes data handling, model training, experiment tracking, automation, deployment, and basic monitoring.

## Framework
For this project, we use the **Transformers** framework, as the task is centered around text classification. The framework provides access to many state-of-the-art pre-trained NLP models and integrates well with PyTorch, which is the base framework used in the course.

## How the Framework Is Used
We plan to take advantage of pre-trained transformer models as a starting point. Initially, we will evaluate how a pre-trained model performs on our dataset. If time allows, we will fine-tune the model on the job posting data. Using pre-trained models allows us to focus more on the MLOps aspects of the project rather than spending unreasonable time on model development.

## Data
The dataset used in this project is the **[Fake vs Real Job Postings (Synthetic NLP Dataset)](https://www.kaggle.com/datasets/khushikyad001/fake-vs-real-job-postings-synthetic-nlp-dataset/data)** from Kaggle. It consists of approximately 3,000 synthetic job postings with a total of 25 features. Each sample includes text fields such as job descriptions and requirements, as well as categorical and numerical features. The dataset also contains missing values.

## Models
We expect to work mainly with pre-trained transformer-based models for text classification. Depending on time constraints, we may experiment with lighter or more optimized variants to reduce training and inference time.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Data Management

This project uses **DVC (Data Version Control)** with Google Cloud Storage. After cloning the repository, pull the data:

```bash
uv run dvc pull
```

For detailed instructions on working with data, see [docs/source/data_management.md](docs/source/data_management.md).

## Running tests and coverage

Use the project's `uv` wrapper to run tests and print a coverage report.

- Run the test suite for the `tests/` folder:

```powershell
uv run pytest tests/
```

- Generate a coverage report (requires `coverage` to be installed in the environment):

```powershell
uv run coverage report -m
```
