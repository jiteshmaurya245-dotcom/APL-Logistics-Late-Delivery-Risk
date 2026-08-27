# Machine Learning–based Late Delivery Risk Prediction in Global Supply Chain Operations

Unified Mentor internship project, sponsored by **APL Logistics (KWE Group)**.

Predicts the probability that an order will arrive late *before* it ships,
so operations teams can intervene proactively instead of reacting after a
delay has already happened.

## Repo structure

```
├── app/
│   └── streamlit_dashboard.py   # the live dashboard
├── data/                         # put APL_Logistics.csv here (git-ignored)
├── docs/
│   └── project-brief.md         # problem statement & methodology
├── scripts/
│   └── train_model.py           # optional: train once, cache the model
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Place `APL_Logistics.csv` in the `data/` folder.

## Run the dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

The dashboard trains the model on first load (cached for the session). For
faster startup — or to reuse a fixed model across sessions — run the
training script once beforehand:

```bash
python scripts/train_model.py
```

## Dashboard features

- **Overview** — risk distribution, tier validation against actual late
  rates, probability distribution, top risk drivers
- **Order Risk** — enter a hypothetical order and get an instant risk score
- **Region & Mode** — risk broken down by shipping mode, market, and
  department
- **Action Panel** — filterable, sortable, downloadable list of high-risk
  orders with recommended interventions per tier

All KPIs and reference lines are computed live from the currently filtered
data (shipping mode / market / segment / risk-threshold filters in the
sidebar) rather than fixed to the full dataset.

## Model

- Random Forest Classifier, class-weighted for the on-time/late imbalance
- Two columns dropped deliberately to avoid target leakage:
  `Days for shipping (real)` and `Delivery Status` — see `docs/project-brief.md`
  for why
- Evaluated on ROC-AUC, precision, recall, and F1

## Deliverables

- Research paper (EDA, insights, recommendations)
- Streamlit dashboard (this repo)
- Executive summary for stakeholder review
