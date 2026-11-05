# API and container validation

The [GitHub Actions run](https://github.com/JesseCoggins/flight-delay-mlops/actions/runs/41956030857) validates source commit `8fd8251ab1b83fba2ea62f8e6cc96aa35e916772` from a clean checkout.

| Check | Result |
|---|---|
| Docker build using the published Dockerfile and Python 3.11 base image | Passed |
| Existing API tests executed inside the image | 6 passed |
| Running-container health endpoint | Passed |
| Running-container prediction request | Passed |

The prediction smoke check uses the documented LAX example and asserts that the API returns a finite numeric delay. The six tests cover the health endpoint, valid prediction, airport aliases, invalid airport/time input and missing parameters.

A separate fresh local checkout at `5b9e2d5cec4a520c9566d519e1731f26cdc2cc91` also passed all six tests under Python 3.13.1. Docker validation ran on GitHub's Ubuntu runner because Docker was unavailable on the local machine.

The [validation workflow](../.github/workflows/validate.yml) builds the image, mounts the test directory read-only, waits for service readiness, checks the running service, and removes the temporary container. It publishes no image and uses read-only repository permissions.

These checks validate the packaged service. They do not retune the model or change its reported predictive performance.
