# API and container validation

The [GitHub Actions run](https://github.com/JesseCoggins/flight-delay-mlops/actions/runs/41962341293) validates source commit `944aff386abf7cab2722a35980cbea2ff5174436` from a clean checkout.

| Check | Result |
|---|---|
| Docker build using the published Dockerfile and Python 3.11 base image | Passed |
| Existing API tests executed inside the image | 6 passed |
| Running-container health endpoint | Passed |
| Running-container prediction request | Passed |

The prediction smoke check uses the documented LAX example and asserts that the API returns a finite numeric delay. The six tests cover the health endpoint, valid prediction, airport aliases, invalid airport/time input and missing parameters.

The same staged source passed all six tests locally under Python 3.13.1. The renamed preprocessing/training scripts and both notebooks were also executed in isolated copies, each reproducing test MSE 79.64805182258482, RMSE 8.92457572227301 and mean prediction 4.47651510883925. The checked direct dependency versions are in `requirements-verified.txt`. Docker validation ran on GitHub's Ubuntu runner because Docker was unavailable on the local machine.

The [validation workflow](../.github/workflows/validate.yml) builds the image, mounts the test directory read-only, waits for service readiness, checks the running service, and removes the temporary container. It publishes no image and uses read-only repository permissions.

These checks validate the packaged service. They do not retune the model or change its reported predictive performance.
