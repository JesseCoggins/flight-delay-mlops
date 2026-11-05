# Flight-delay prediction pipeline and API

I built a service that an airline planner can query for the **average departure delay** from Atlanta given a destination and scheduled departure and arrival times. Starting from the raw flight records, the training pipeline reproduces a held-out root mean squared error of **8.92 minutes**. The estimate applies to routine schedules. Departures delayed by an hour or more were excluded from the evaluation.

[Inspect the serving code](src/deployment_api.py) · [Review the training notebook](notebooks/flight_delay_mlflow_tracking.ipynb) · [Read the API tests](tests/test_api.py)

## From flight records to a prediction

1. **Prepare the data.** Filter the raw on-time records to Atlanta departures, remove duplicates and incomplete records, and build the mapping between IATA airport codes and numeric BTS IDs. The preparation script produced 22,580 cleaned flight records. [Preprocessing code](src/Task_2.py) · [DVC data pointers](data/)
2. **Train and evaluate.** Group by destination and scheduled times to model average delay. Sweep 20 Ridge penalties, track experiments in MLflow and evaluate on later dates. The selected alpha was 3.8. Test MSE was 79.65 minutes squared and the mean prediction was 4.48 minutes. [Training script](src/Task_2_MLFlow.py) · [Performance plot](assets/performance_plot.png)
3. **Serve and test.** FastAPI loads the model and encoding files from `src/`. The root and packaged entry points share that service. Tests cover a valid prediction, airport aliases, invalid inputs and missing parameters. The GitLab CI configuration includes a job that rebuilds the model before testing it. [Tests](tests/test_api.py) · [GitLab CI](.gitlab-ci.yml) · [Dockerfile](Dockerfile)

## Run and test

In a Python 3.11 environment, run these commands from the repository root:

~~~sh
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m uvicorn deployment_api:app
~~~

## API

With the server running, send a request from a second terminal. For the packaged model, this example returns about **4.55 minutes**. Using BTS destination ID `12892` in place of `LAX` gives the same prediction. Times are local scheduled times in `HH:MM` format.

~~~sh
curl "http://127.0.0.1:8000/predict/delays?dest_airport=LAX&departure_time=14:30&arrival_time=17:45"
~~~

Six API tests passed on a clean checkout and inside the built Docker image. The running container also passed health and prediction smoke checks. [Validation details](docs/validation.md). `GET /` provides a health check. `GET /predict/delays` returns a prediction, HTTP 400 for an unknown airport or invalid time, and HTTP 422 for missing required parameters.

## Rebuild the served model

~~~sh
python src/Task_2.py
python src/Task_2_MLFlow.py
~~~

These scripts write the model and encodings used by the API. DVC metadata accompanies the raw and cleaned CSVs. Both datasets are included. [Data and version pointers](data/) · [Python dependencies](requirements.txt)

## Notebooks

For the interactive coursework workflow, use `notebooks/` as the working directory. Run [flight preparation](notebooks/flight_delay_pipeline.ipynb) first. It writes `../data/cleaned_data.csv`, which [MLflow training and evaluation](notebooks/flight_delay_mlflow_tracking.ipynb) reads. The training notebook writes its model and airport encodings at the repository root. To rebuild the artifacts loaded from `src/` by the API, use the scripts above.

The sweep tested alpha values from 0.0 through 3.8 in steps of 0.2. The best tested value, 3.8, is at the upper boundary; larger penalties were not evaluated, so a better value may lie beyond this range.

## Reproducibility

The 70/30 training-validation split uses `random_state=42`. The pipeline refits the selected Ridge alpha, logs that value in MLflow, and packages matching model and encoding files for the API. The held-out test MSE is **79.6481**, RMSE **8.9246 minutes**, and mean prediction **4.4765 minutes**.

## A separate grocery MLOps design

The [Kronkers case study](docs/kronkers-mlops-design.md) explains how to tie a grocery forecast to stockouts, monitoring, retraining and rollback decisions.

*WGU M.S. in Data Analytics coursework, D602 Deployment.*
