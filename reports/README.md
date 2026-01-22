# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [x] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

103


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s250695, s253811, s250778, s250779

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We didn't really used any other libraries than the ones covered in the course.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used uv for managing our dependencies. The list of dependencies was auto-generated using pyproject.toml . To get a complete copy of our development environment, one would have to run the following commands*
>
> Answer: uv sync

We decided to use the uv to manage our dependencies. This was motivated by a great developer experience that comes from using uv as well as its speed when it comes to resolving dependencies. The version of python that should be used with the project is stored in `.python-version` file.
Dependencies and their versions are stored automatically in `pyproject.toml` if added through `uv add <dep-nam>`, developers are also able to modify `pyproject.toml` themselves. A new team meber to get an exact copy of dev environment would just need to simply run (given of course that they
have uv already installed on their machine) `uv sync`.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We have used the template showcased during the lecture that also includes all the files related to agentic programming. We have used most of the folders except the `notebooks` as we started working on actual python files from the start. Additionally, we have added a `scripts` directory.
This directory contains the python scripts that should be suppose to run from command line, without being core of actual application, specifically a script that is located there is being called during CI to generate a comment in each pull request showcasing various data statistics.
The overall structure of the product is centered around `src` containing the actual code of application, `data` containing both raw and processed data as well as dvc info, `models` containing output of training runs, `dockerfiles` with Dockefiles, `.github` with all the CI related files and `tests` containing all the test

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used ruff for both linting and formatting code style. Type hints are enforced throughout the codebase using Python's typing module, with pyright configured in pre-commit hooks to catch type errors before commits. Documentation follows Google-style docstrings for all functions and classes, with mkdocs configured to build documentation from markdown files.

These concepts are critical in larger projects. Code formatting ensures consistency across team members, reducing cognitive load during code reviews. Linting catches common errors (unused imports, undefined variables) automatically. Typing prevents runtime errors by validating function signatures at development time.

In our project, these tools are enforced in CI/CD via uv run ruff check . --fix and uv run pre-commit run --all-files, ensuring every commit meets quality standards before merging to main. This scales development: new contributors can understand APIs quickly, and bugs are caught earlier when code is uniform and type-safe.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**

> Answer:

The test suite contains five focused tests covering data, model, and API. Data tests validate the dataset and datamodule load preprocessed tensors, correct lengths, tensor shapes, and dtypes. Model tests verify model construction, a forward pass, and a single optimization step to ensure training paths work. API tests exercise key endpoints (responses and status codes) to catch integration regressions. Together they provide smoke-level coverage of critical pipelines.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

The total code coverage of code is 58%, which includes all our source code apart from tests folder.
We are far from 100% coverage of our code, but even 100% test coverage does not guarantee a bug free code, it is only as good as the tests.
Code coverage measures how much code is executed during testing, not whether that code is correct.
A test can run a line of code without actually validating its behavior.
For example, a test could execute a function that returns an incorrect value but still count toward coverage if the assertion doesn't catch it.
Additionally, coverage metrics often miss edge cases, integration failures between components, and production-specific issues like concurrency bugs or memory leaks that only manifest under load.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests throughout our project, following a feature branch workflow to maintain code quality and facilitate collaboration.
Each major task or feature (e.g., CI/CD setup, API development, monitoring implementation) had its own dedicated branch created from main.
Team members worked independently on their branches, committing regularly without blocking others.

When a feature was complete, we opened a pull request to main. Every PR required at least one approval and had to pass all automated checks: unit tests, linting. This gating mechanism prevented broken or non-compliant code from reaching production.
Code reviews caught logic errors, style violations, and architectural issues before merge, improving overall code quality and knowledge sharing across the team.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did make of use of DVC in the following way: instead of storing raw dataset (before preprocessing) we use data versioning for it. The actual dataset is stored in bucket in Google Cloud. In our actual case we only had single version of dataset as it was a simple synthethic dataset from kaggle.
But DVC was still a nice improvement of life, as all the team members could simply pull the data when running project. Additionally, if we were to train the model in the cloud we would also not have to care about making sure the data is the we could simply pull it. The versioning of data itself would be more useful
if we would version actual preprocessed datasets, we would have different versions containing different feature preprocessing and that way we would be able to very easily swap between version.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We run continuous integration on GitHub Actions and have separated responsibilities across jobs to keep feedback fast and actionable. CI covers: unit testing (uv run pytest tests/), linting and formatting (uv run ruff check . --fix and uv run ruff format .), and pre-commit hooks (uv run pre-commit run --all-files). Workflows are triggered on pull requests and pushes to main; branch-protection requires passing CI and at least one review before merge. Tests and linters run in a matrix across operating systems (ubuntu-latest, windows-latest, macos-latest) and multiple Python versions (3.9–3.11) to catch platform-specific issues.
Dependency installation uses the project’s uv-managed environment (uv sync / uv install) so CI mirrors local developer environments; the uv.lock file is used to pin versions. We enable caching (actions/cache) for the pip/venv cache keyed by python-version and the uv.lock hash to speed runs and reduce network overhead. Build artifacts such as test coverage reports are uploaded as job artifacts; coverage is produced during pytest runs and used for monitoring test health over time.
Formatting and linting are enforced in CI (fail on ruff errors) and pre-commit is executed to keep commits clean. Additional jobs include docs build checks (uv run mkdocs build) and an optional Docker image build for deployment testing. See .github/workflows/ci.yml for the CI implementation and workflow details.


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used Hydra for experiment configuration management. Configuration files are stored in configs/ with subdirectories for model/, trainer/, and data/ settings. To run an experiment with default config:

uv run python src/postings_classifier/train.py

To override specific parameters from the command line:

uv run python src/postings_classifier/train.py trainer.lr=1e-4 trainer.batch_size=32 trainer.max_epochs=20

Hydra automatically merges configs, validates values, and logs all parameters to outputs/ for full reproducibility.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Frankly speaking, given the problem and dataset and size of transformers picked, our model managed to achieve very good test performance already on first experiment and therefore we haven't experimented much.
Even though we made sure that the infrastructure for such experiments is in place. We use hydra for setting reproducible configuration (including data, model config and even seed), logging to monitor a process of training and finally logging
actual model in wandb. If one would like to reproduce the experiment they would have to use the same version of dataset (achievable thanks to DVC) and make sure that hydra configuration have the same values as during that experiment and of course
be on the same git commit as when experiment was run.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We developed three Docker images: one for API/inference deployment, one for training.
To run the API image locally:
`docker build -f dockerfiles/api.dockerfile -t postings-api:latest . && docker run -p 8000:8000 postings-api:latest`.
The training image (`train.dockerfile`) installs all dependencies including GPU support (PyTorch, Lightning) and can be invoked with `docker run -v $(pwd)/data:/app/data postings-train:latest`. Both images are built automatically via
Cloud Build in our CI/CD pipeline (`cloudbuild_containers.yaml`) and pushed to Artifact Registry.
Additionally of pushing API image to the registry, there is a trigger that deployes it automatically to CloudRun
`docker-compose.yml` enables local multi-container orchestration for development.

Link to serving image: [here](https://github.com/Hedrekao/mlops-final-project/blob/main/dockerfiles/api.dockerfile)

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We used print statements and logging with `loguru` to trace execution flow, or some team members used VS Code's built-in debugger during development. We extensively used PyTorch Lightning's logging and validation callbacks to catch issues early during training. We did implement profiling via PyTorch Profiler (configured in `configs/trainer/default.yaml`) to identify performance bottlenecks. The profiler tracks CPU/GPU time and memory allocation per operation. Additionally, we used `loguru` for structured logging across all modules (training, inference, API) to surface errors and warnings. For API debugging, we used FastAPI's built-in documentation (`/docs`) and manual curl requests. Cloud deployment bugs were debugged via Cloud Logging and local Docker testing before pushing to production.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services: Bucket, Cloudbuild, CloudRun, Artifact Registry. Bucket is used for storing logs, monitoring and actual data referenced by DVC.
Cloudbuild automatically builds containers using Dockerfiles and stores them in Artifact Registry. We used CloudRun to deploy our inference server and it runs the API Docker Image.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

Because we did not train our model in the cloud but rather locally and used the CloudRun for hosting the inference server, the only moment we had briefly encountered Compute engine
was when building docker images in the CloudBuild as it spins up a vm to run the building process. If we were to fully utilize the Compute engine we would most likely try to get quota for the GPU virtual machines
as that would allow us to run even faster training. Also many Google Cloud services try to abstract the fact that they use Compute Engine under the hood providing more user friendly interfaces.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Buckets overview](figures/bucket1.png)
![Specific bucket](figures/bucket2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Artifact registry](figures/registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![CloudBuild](figures/cloudbuild.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We were planning to train our model in the cloud, however in the end we decided not to for a few following reasons. First and foremost, the problem and dataset that we selected for this project (natural langugage processing task)
were not problem for a pretrained transformer from hugging face with the classifier head attached on top of it. Because of that our first experiment result in a model that aced through benchmarks both on train and test sets. Because of that we had no reason to do multiple
of experiments which would probably force to use cloud for time efficiency. Moreover, the actual training time was also not a terrible one, with around 30 minutes. Training in the cloud would be required if the project involved automatic/on-demand retraining of the model
using new data, however in our simplified case we just used a static dataset from kaggle.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We built a FastAPI service (`postings_classifier.api`) with endpoints `/`, `/health`, `/predict`, `/monitoring/stats`, `/monitoring/report`, and `/metrics` (Prometheus). On first use it lazily loads a DistilBERT checkpoint from `models/checkpoints/` or a GCS-mounted path (envs: `MODEL_CHECKPOINT`, `HF_MODEL_PATH`, `HF_HOME`, and `TRANSFORMERS_OFFLINE=1`); if nothing is available it falls back to a simple rule-based predictor so the API still responds. `/predict` returns label + score and logs each request in the background to a GCS bucket for monitoring/drift checks. We added Prometheus counters/histograms for request counts, errors, latency, and input length. Health reports whether the model is loaded and the last load error. Monitoring endpoints read recent predictions (GCS first, local CSV fallback) and render a lightweight HTML report for quick drift/volume inspection.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes. Locally we run `uv run uvicorn postings_classifier.api:app --reload` for dev. For cloud we containerized the FastAPI app with `dockerfiles/api.dockerfile`, built via Cloud Build, pushed to Artifact Registry, and deployed to Cloud Run (europe-west1, CPU, autoscaling). The service mounts our GCS bucket `jop-postings-mlops-data` at `/gcs/jop-postings-mlops-data` and reads env vars: `MODEL_CHECKPOINT` (pointing to checkpoint in GCS), `HF_MODEL_PATH`, `HF_HOME`, and `TRANSFORMERS_OFFLINE=1` to force local-only HuggingFace loads. To invoke the deployed service we POST JSON to `/predict`:

curl -X POST "https://postings-classifier-api-948592557572.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "We are hiring a data scientist with Python and NLP experience."}'

Returns `{"label":"fake","score":0.799...}`. Health, monitoring stats/report, and Prometheus metrics are on `/health`, `/monitoring/stats`, `/monitoring/report`, `/metrics`.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit testing we used pytest with FastAPI's TestClient. We wrote 20 tests in `tests/test_api.py` covering: root/health/predict endpoints (response structure, status codes), edge cases (empty text, whitespace, unicode, special characters, long text), valid label/score ranges, and integration scenarios. All 20 tests passed successfully.

For load testing we used Locust against the deployed Cloud Run service. We simulated 10 concurrent users with weighted endpoints: `/predict` (weight 10), `/` (weight 2), `/health` (weight 1), `/monitoring/stats` (weight 1), `/monitoring/report` variants (weight 1). Over 3 minutes, we generated 664 total requests with 0 failures. Key results: `/predict` handled 447 requests at avg 86.94ms (50th percentile 77ms, 95th percentile 150ms), achieving 2.58 RPS. Root and health endpoints averaged 33ms and 30ms respectively. Throughput reached 3.83 RPS peak. The API never crashed and gracefully handled all concurrent traffic, though `/monitoring/stats` is slower (7271ms avg) due to GCS reads. Overall the deployment is stable and responsive for the core prediction workload.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we implemented two-layer monitoring. **System metrics**: FastAPI `/predict` endpoint includes Prometheus instrumentation tracking request count, error rate, latency distribution, and input text length. These metrics are exposed at `/metrics` and scraped by Cloud Run's Managed Service for Prometheus sidecar every 30 seconds, pushing to Google Cloud Monitoring for real-time dashboards and alerting. **Prediction monitoring**: The `/monitoring/report` endpoint displays HTML dashboards showing label distributions, confidence scores, and recent predictions. All predictions are logged to GCS (`predictions/prediction_*.json`) with timestamps for historical analysis. Together, these layers continuously measure system health, track inference performance, maintain prediction audit trails, and enable detection of anomalies that could indicate model degradation or data distribution shifts.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

As it currently stands the total cloud cost up-to-date is 13.5$, where the actually the majority of that (over 13$) comes from container registry vulnerability scanning feature that we enabled for test.
The second most expensive service is the CloudRun used for hosting inference server, which for now only costed 0.4$. In general working in the cloud went pretty smoothly, thanks to the guides that we had available in the course.
But unfortunately, the UI and UX of GoogleCloud is horrible. Things are hidden, slow, unintuitive, unresponsive, costs are also not that easy to estimate right away. Additionally, debugging issues that only appear in the cloud
is not fun at all, but this is the reality we live in and we have to deal with it :)


### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We did not add any extra features beyond the required scope. Instead, we focused on making sure the core parts of the project were solid and worked well end to end. We chose to skip optional additions like a frontend or more advanced infrastructure, as our main priority was building a stable and reliable pipeline rather than adding extra components.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![MLOPS Architecture](figures/MLOPS_diagram.png)

Our system architecture follows a complete MLOps pipeline from local development to cloud deployment and monitoring.

**Local development**: Team members clone the repository, set up the environment with `uv sync`, pull data using DVC and develop code.
Tests (pytest) validate data loading, model training, and API functionality locally before pushing. Additionally there is pre-commit hook to check linting before even pushing the commit.

**Data versioning**: Training data (`data/raw/fake_real_job_postings_3000x25.csv`) is versioned using DVC with remote storage configured at `gs://jop-postings-mlops-data/dvc-store`. This allows team members to track dataset changes while keeping the Git repository lightweight.

**CI/CD pipeline**: When code is pushed to GitHub, GitHub Actions triggers automated workflows: unit tests, linting, data statistics workflow (to check quality of data). If all checks pass and the PR is approved, code merges to main,
automatically triggering Cloud Build to construct Docker images (training, API) and push them to Google Artifact Registry, serving image is also automatically deployed to CloudRun.

**Cloud deployment**: The API image is deployed to Cloud Run, which auto-scales based on traffic. The Prometheus sidecar is configured on the Cloud Run service for Google Cloud Monitoring for real-time dashboards and alerting.

**Data flow**: Input requests hit the `/predict` endpoint, which loads the PyTorch model and returns predictions. Every prediction is logged as JSON to Google Cloud Storage (`predictions/prediction_*.json`).
There is also a dashboard available to can be used to check against data drift


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

// Please add all of you here your contributions

**Student s253811**  responsible for implementing profiling (PyTorch Profiler integration), logging infrastructure (loguru setup across modules), and the complete monitoring system (Prometheus metrics instrumentation, `/monitoring/report` endpoint, Cloud Monitoring integration, prediction logging to GCS). Also some necessary fixes to GitHub Actions workflows and other code maintenance tasks throughout the project.

**Student s250695**  was responsible for implementing and maintaining the continuous integration pipeline. This included setting up GitHub Actions workflows with multi-OS and multi-Python version testing, configuring pytest with coverage reporting, integrating ruff linting and formatting checks, and setting up pre-commit hooks to enforce code quality standards before commits. Additionally, implemented distributed data loading optimization following the DTU MLOps M29 module, including multi-worker PyTorch DataLoader configuration with GPU memory optimization , performance benchmarking across different worker counts, and dataloader best practices in the data.py module to improve training pipeline performance.

**Student s250778** was responsible for creating the project description in the main README.md. Designed and implemented the FastAPI inference service, including model loading from cloud storage with fallback logic. Led the most cloud deployment pipeline: set up Cloud Run deployment, containerized the application with Docker, and tested the build process end-to-end. Modified the model and tokenizer loading to work efficiently in a cloud environment with GCS bucket integration and set up environment variables for cloud storage paths. Uploaded model checkpoints and HuggingFace model files to the GCS bucket. Implemented API testing with pytest and performed load testing using Locust.

**Student 250779** @Me

All members contributed to code reviews, testing, and documentation.
We used GitHub Copilot for code completion and ChatGPT/GitHub Copilot Chat for debugging, explaining error messages, and generating code.
