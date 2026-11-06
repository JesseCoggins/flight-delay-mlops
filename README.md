# Flight-delay prediction pipeline and API

I built a service that an airline planner can query for the **average departure delay** from Atlanta given a destination and scheduled departure and arrival times. Starting from the raw flight records, the training pipeline reproduces a held-out root mean squared error of **8.92 minutes**. The estimate applies to routine schedules. I excluded departures delayed by an hour or more from the evaluation.

![Actual versus predicted average flight delays for held-out route and schedule groups](assets/model_performance_test.jpg)

*Actual versus predicted mean delays on held-out route/schedule groups. Departures delayed by an hour or more were excluded.*

[Inspect the serving code](src/deployment_api.py) · [Review the training notebook](notebooks/flight_delay_mlflow_tracking.ipynb) · [Read the API tests](tests/test_api.py)

## From flight records to a prediction

1. **Prepare the data.** Filter the raw on-time records to Atlanta departures, remove duplicates and incomplete records, and build the mapping between IATA airport codes and numeric BTS IDs. The preparation script produced 22,580 cleaned flight records. [Preprocessing code](src/preprocess.py) · [DVC data pointers](data/)
2. **Train and evaluate.** Group by destination and scheduled times to model average delay. Sweep 20 Ridge penalties, track experiments in MLflow and evaluate on later dates. The selected alpha was 3.8; the full held-out metrics are in [Reproducibility](#reproducibility). [Training script](src/train.py) · [Performance plot](assets/performance_plot.png)
3. **Serve and test.** FastAPI loads the model and encoding files from `src/`. The root and packaged entry points share that service. Tests cover a valid prediction, airport aliases, invalid inputs and missing parameters. GitHub Actions builds the Docker image, runs the six API tests and checks the running service. [Tests](tests/test_api.py) · [GitHub Actions](.github/workflows/validate.yml) · [Dockerfile](Dockerfile)

## Run it yourself

### Install and test

From the repository root in Python 3.11 or 3.13:

~~~sh
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m uvicorn deployment_api:app
~~~

Six tests passed locally and inside the Docker image, along with container health and prediction checks. [Validation details](docs/validation.md).

### Query the API

From a second terminal:

~~~sh
curl "http://127.0.0.1:8000/predict/delays?dest_airport=LAX&departure_time=14:30&arrival_time=17:45"
~~~

The packaged model returns about **4.55 minutes**. BTS ID `12892` gives the same result as `LAX`; times are local scheduled times in `HH:MM` format. `GET /` checks service health. `GET /predict/delays` returns HTTP 400 for an unknown airport or invalid time and HTTP 422 for missing parameters.

### Rebuild the model

~~~sh
python src/preprocess.py
python src/train.py
~~~

The scripts export the served model and encodings to `src/`; the root `deployment_api.py` points to the same application. For notebook use, set the working directory to `notebooks/`, run [preparation](notebooks/flight_delay_pipeline.ipynb), then [training](notebooks/flight_delay_mlflow_tracking.ipynb). They read and write `../data/cleaned_data.csv` and export to `../src/`.

Both CSVs are committed in Git. DVC pointers record their checksums; no remote or `dvc pull` is needed. [Data notes](data/README.md).

### Reproducibility

I used a 70/30 training-validation split with `random_state=42` and refitted the selected Ridge alpha before packaging the model. Held-out MSE is **79.6481**, RMSE **8.9246 minutes**, and mean prediction **4.4765 minutes**.

I tested alpha values from 0.0 through 3.8 in steps of 0.2. The best tested value, 3.8, is at the upper boundary; a better value may lie beyond that range.

## Additional coursework case study

Kronkers is a supermarket scenario from a separate business-case assignment in the same deployment course. The [grocery-chain MLOps design](docs/kronkers-mlops-design.md) covers proposed forecasting, monitoring, retraining and rollback requirements. It is a written design, separate from the flight-delay implementation and its reported results.

## Data and environment

Data: WGU on-time reporting extract; [source and DVC notes](data/README.md).  
Install: `python -m pip install -r requirements.txt`; [environment details](requirements-verified.txt).

*WGU M.S. in Data Analytics coursework, D602 Deployment.*
