# Graph Deep Learning Postprocessing Project @USI

Project designed for the [Graph Deep Learning](https://search.usi.ch/en/courses/35270698/graph-deep-learning) course at Università della Svizzera italiana (USI), focusing on postprocessing wind NWP forecasts.

<img src="./imgs/wind_stations.png" alt="Description" width="600">

## Quickstart

1. **Install the dependencies:**
```sh 
poetry install
```

2. **Activate the environment:**
```sh
cd spatiotemporal_postprocessing
poetry shell
```

3. **Train**

Define the folder with the training data:

```sh
export DATA_BASE_FOLDER=<FOLDER>
```

Define the MLFlow tracking URI (defaults to a local folder called `mlruns`):

```sh
export MLFLOW_TRACKING_URI=<URI>
```

Train with default settings:
```sh
python train.py
```

Train with a different config:
```sh
python train.py --config-name <CFG>
```

Overwrite (if existing) or append (if not existing) a config value, such as the optimizer:

```sh
python train.py ++training.optim.algo=SDG
```

4. **Check the logs on MLflow:**

```sh
mlflow ui --port <PORT>
```