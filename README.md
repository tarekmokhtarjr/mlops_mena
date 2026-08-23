# MiniLM

## Installation
Python dependencies in this project are handled by uv,
check the documentation here: https://docs.astral.sh/uv/ .

### Local installation on Linux

```sh
chmod +x setup.sh
./setup.sh
```

## Using pre-commit for automated code checks
Run pre-commit
```sh
uv run pre-commit run --all
```

## Run app locally using uvicorn
Run pre-commit
```sh
uvicorn --app-dir src/app main:app
```

## Run app using docker
Run pre-commit
```sh
uvicorn --app-dir src/app main:app
```
## Testing
Use this command to run unit tests
```sh
uv run pytest test \
    --cov=src/app \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=html:cov_report \
    --cov-config .coveragerc
```

This will generate coverage report which could be accessed in the generated cov_report folder, also html test results could be found in the generated folder report_output.
