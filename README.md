# Neural Network Feature Kit

## Install
```shell
python -m pip install http+git@github.com:innerNULL/nnfeakit.git
```

## Build Python Environment
```shell
python -m venv ./_venv --copies
source ./_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# deactivate
```

## Test
```shell
python -m pip install -r requirements-test.txt
python -m pytest ./tests/ --cov=./src/nnfeakit --durations=0 -v
```
