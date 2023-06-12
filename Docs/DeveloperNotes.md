# ML-04-ThyroidPrediction

## Step 1: Create Conda Environment

```
conda create -p venv python==3.10 -y
```

## Step 2: Create following
            a. .github/workflows/main.yaml
            b. .dockerignore
            c. .Dockerfile
            d.  app.py
            e.  requirements.txt

## Step 3: Add The content to files
            a. Add CI/CD file for Gihub Actions as main.yaml in .github/workflows/main.yaml
            b. Add the file names are not be added in Docker Image
            c. Add Content to Dockerfile
            d. Creat flask app

## Step 4: Build Docker Image

*Build Docker Image*
```
docker build -t <image-name>:<tag-nam> <location-of-docker-file for curren directory just add dot (.)>
```
<span style="color:red">Note: Docker Image name must be lowrcase</span>

*To Check List of Docker Images*
```
docker images
```

*To Run Docker Image*
```
docker run -p 5000:5000 -e PORT=5000 <Image-ID>
```

*To Check Running Containers in docker*
```
docke ps
```

*To Stop Docker Container*

```
docker stop <container_id>
```

## Step 5: Poject Structure Creation
### Step 5.1: Create Folder With Project Name <ThyroidPrediction> in root directory

a. Add a init file

        `_init__.py`

b. Inside ThyroidPrediction folder Create following folders with `__init__.py` file:

* ThyroidPrediction
    - `__init__.py`
    - exception / `__init__.py`
    - logger / `__init__.py`
    - component / `__init__.py`
    - pipeline / `__init__.py`
    - entity / `__init__.py`
    - config / `__init__.py`
    - constant / `__init__.py`

### Step 5.2: Create setup.py file in root directory

### Step 5.3: Distribution Package Building
For Python < 3.10
```
python setup.py install
```

For Python >= 3.10
```
pip install build

```

```
python -m build
```

> if you run python -m build --wheel, it will create a .whl file in the dist directory. Similarly, if you run python -m build --sdist, it will create a source distribution package (a .tar.gz or .zip file) in the dist directory.

> If python -m build or python -m build --wheel is not working then just add '-e .' in requirements.txt file and remove it when reading it in setup.py file.

Run pip install -r -e .
Run pip install -r requirements.txt


## Step 5: Start working on `logger` and `exception`

## Step 5: Start working on `constant/__init__.py`

## Step 5: Create Files in entity folder and component folder

* ThyroidPrediction
    - `__init__.py`
    - entity

        -   `__init__.py`
        -   artifact_entity.py
        -   config_enity.py
        -   experiment_entity.py
        -   model_factory.py
        -   thyroid_predictor.py
    - component
        -   `__init__.py`
        -   data_ingestion.py
        -   data_validation.py
        -   data_transformation.py
        -   model_evaluation.py
        -   model_pusher.py
        -   model_trainer.py


# Check AWS S3 Bucket and AZURE before terminating the service

# PROJECT TREE


```
ML-04-ThyroidPrediction
├─ .dockerignore
├─ .git
├─ .github
│  └─ workflows
│     └─ main.yaml
├─ .gitignore
├─ app.py
├─ Dockerfile
├─ LICENSE
├─ logs
├─ README.md
├─ requirements.txt
├─ setup.py
└─ ThyroidPrediction
   ├─ component
   │  ├─ data_ingestion.py
   │  ├─ data_transformation.py
   │  ├─ data_validation.py
   │  ├─ model_evaluation.py
   │  ├─ model_pusher.py
   │  ├─ model_trainer.py
   │  └─ __init__.py
   ├─ config
   │  └─ __init__.py
   ├─ constant
   │  └─ __init__.py
   ├─ entity
   │  ├─ artifact_entity.py
   │  ├─ config_entity.py
   │  ├─ experiment.py
   │  ├─ model_factory.py
   │  ├─ thyroid_predictor.py
   │  └─ __init__.py
   ├─ exception
   │  └─ __init__.py
   ├─ logger
   │  └─ __init__.py
   ├─ pipeline
   │  ├─ pipeline.py
   │  └─ __init__.py
   └─ __init__.py

```


# Deployment: Elastic Beanstalk
-   .ebextension/python.config folder and file should be there
-   github/workflows/amin.yaml should not b there

# For Azure Deployment 
-    Remov github/workflows folder

*******************************************************

# End to End MAchine Learning Project

1. Docker Build checked
2. Github Workflow
3. Iam User In AWS

*   Create I am user with permissions to AmazonEC2FullAccess and AmazonFullAccess
*   Create ECR repository
*   Create instance EC2 Instance and connect.


## Docker Setup In EC2 commands to be Executed

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:

## Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app