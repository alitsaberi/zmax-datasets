# ZMax Real-Time Analysis
## Installation
### Requirements
- Python >= 3.9
### Development
1. Create a virtual environment
```
python3 -m venv $VENV_PATH
```
2. Install [Poetry](https://python-poetry.org/docs/#installing-manually)
```
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry
```
3. Install dependencies
```
poetry install --with test,dev
```
