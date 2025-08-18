# ZUtils

## Prerequisites

- Python 3.10

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/alitsaberi/zmax-datasets.git
   ```

2. Navigate to the project directory
   ```bash
   cd zmax-datasets
   ```

3. Create a virtual environment and activate it

   **Windows:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install [Poetry](https://python-poetry.org/docs/#installing-manually)
   ```bash
   pip install poetry
   ```

5. Install dependencies
   ```bash
   # Install without extras
   poetry install

   # Install with specific extras
   poetry install -E <extra_name>

   # Install with all extras
   poetry install --all-extras

   # Install with multiple extras
   poetry install -E sleep_scoring -E artifact_detection
   ```

### Available Extras

ZUtils provides several optional extras that can be installed based on your needs:

- `sleep_scoring`: Sleep stage scoring using U-Time models
- `artifact_detection`: Tools for detecting artifacts in EEG signals
- `yasa`: Integration with YASA (Yet Another Spindle Algorithm) for sleep analysis and feature detection
- `notebook`: Jupyter notebook dependencies for interactive analysis and visualization

## Usage

### Sleep Scoring

Score sleep stages from ZMax recordings:

```bash
poetry run sleep-scoring \
    /path/to/zmax/recordings/* \
    --aggregate \
    --output-path /path/to/output
```

For a complete list of options for each script, use `--help`.

## License

This project is licensed under the terms of the MIT license.