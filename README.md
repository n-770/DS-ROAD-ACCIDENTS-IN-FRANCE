# DS-ROAD-ACCIDENTS-IN-FRANCE

## Road Accidents in France

This repository contains a multi-class classification project predicting road accident injury severity in France using open data from 2019-2024. The analysis focuses on four severity classes: uninjured (42%), slightly injured (39%), hospitalized (16%), and fatal (3%), addressing strong class imbalance through resampling and boosting models.

## Project Overview

The project builds a supervised multi-class classification pipeline on French government data from data.gouv.fr, covering accidents, vehicles, individuals, and locations for mainland France. Rendering 1 handled data exploration, cleaning, and feature engineering, while the final report details modeling with RandomForest, XGBoost, LightGBM, and CatBoost, emphasizing macro F1 and recall for severe cases.

Key contributions include SHAP-driven feature pruning, imbalance strategies like RandomOver/UnderSampler and BorderlineSMOTE, and a binary reformulation (uninjured vs. injured/killed) achieving macro F1 of 0.79-0.80.

## File Structure

```
project-root/
├── data/                  # Raw CSVs (accidents, vehicles, individuals, locations 2019-2024)
├── notebooks/             # Jupyter notebooks for EDA, preprocessing, modeling
│   ├── 01_data_exploration.ipynb     # From Rendering 1
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_imbalance_handling.ipynb
│   ├── 05_shap_analysis.ipynb
│   └── 06_binary_classification.ipynb
├── models/                # Trained models (.pkl): RF baseline, XGBoost tuned
├── reports/               # PDFs: rendering_1.pdf, final_report.pdf
├── assets/                # Visualizations: PCA plots, SHAP summaries
├── requirements.txt       # Dependencies: scikit-learn, xgboost, lightgbm, catboost, shap
└── README.md              # This file
```

Data totals ~190MB with 328,000 accidents (54,500/year average); preprocessing excludes overseas territories and pre-2019 due to schema changes.

## Setup Instructions

1. Clone the repo: `git clone https://github.com/n-770/DS-ROAD-ACCIDENTS-IN-FRANCE.git` or via SSH using `git clone git@github.com:n-770/DS-ROAD-ACCIDENTS-IN-FRANCE.git`
2. Create virtual env: `python -m venv venv` and activate
3. Install deps: `pip install -r requirements.txt`
4. Run notebooks sequentially: `jupyter notebook notebooks/`

Requires Python 3.13+, pandas, numpy, scikit-learn; large files may need external storage for Git.

## Key Findings

- Baseline RandomForest (over/under-sampling): accuracy 0.71, macro F1 0.57; class 4 (fatal) recall 0.14.
- Top features per SHAP: municipality, safety equipment (indsecu1), vehicle category, user category (indcat), urbanization.
- PCA shows heavy class overlap; fatal recall improves to 0.62 with XGBoost but drops precision.
- Risk highest: head-on collisions, night no lighting, fog/storm, two-wheeled vehicles.

Binary models excel: RandomForest/XGBoost macro F1 ~0.80.

## Data Sources

French open data (data.gouv.fr): annual CSVs for bodily injury accidents 2019-2024; enriched with INSEE population density.

## Team

Alke Simmler, Christian Leibold, Jonathan Becker, Michael Munz;
project mentor Yaniv Benichou.

# Acknowledgments

Data science project completed December 2025.

## License
MIT License