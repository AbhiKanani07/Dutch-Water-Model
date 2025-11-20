## Dutch-Water-Model

Python/Pandas/NumPy project to clean Dutch water-use data, generate summaries/plots, and produce simple linear-trend forecasts. The site (`index.html`) shows precomputed plots and an interactive JS widget that reads the forecast CSV.

## Contents

- `main.py` — load/clean data, EDA, plots, and linear forecasts; writes outputs to `plots/`.
- `index.html` — static site that embeds plots and the interactive forecast explorer.
- `app.js` — client-side logic for the forecast explorer (reads `plots/forecast_total_economy.csv`).
- `plots/` — outputs from `main.py` (PNG charts and forecast CSV).
- `table__82883ENG.csv` — source data (place under your Downloads path or adjust `DATA_PATH` in `main.py`).

## Setup

```bash
python -m pip install pandas numpy matplotlib seaborn requests
