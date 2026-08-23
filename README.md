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
