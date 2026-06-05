# K-Nearest Neighbors from Scratch - Titanic Survival

## Overview

This project implements k-Nearest Neighbors (KNN) from scratch with NumPy and evaluates it on the Titanic survival dataset. The implementation supports classification, regression, Euclidean distance, Manhattan distance, Minkowski distance, probability estimates, and the optional distance-weighted KNN bonus. The repository also contains from-scratch evaluation metrics, stratified train/validation/test splitting, stratified K-fold cross-validation, exploratory analysis notebooks, and PDF figures.

## Implemented functionality

* `KNN.fit`, `KNN.predict`, and `KNN.predict\_proba`
* Classification by majority vote and regression by neighbor averaging
* Euclidean, Manhattan, and Minkowski distance metrics
* Vectorized pairwise-distance computation with NumPy broadcasting
* Accuracy, precision, recall, F1, and ROC-AUC implemented from scratch
* Stratified 60/20/20 train/validation/test split
* Stratified K-fold cross-validation
* Hyperparameter sweep over `k = \[1, 3, 5, 7, 9, 11, 15, 21, 31, 51]`
* Computational runtime benchmark
* Scaling experiment with `StandardScaler` fitted on each training fold only
* **Bonus 1:** distance-weighted KNN using `1 / (distance + 1e-8)`

## Project structure

```text
.
|-- data/
|   `-- titanic3.csv
|-- figures/
|-- notebooks/
|   |-- 01\_eda.ipynb
|   |-- 02\_evaluation.ipynb
|   `-- 03\_cross\_validation.ipynb
|-- src/
|   |-- knn.py
|   |-- metrics.py
|   |-- splits.py
|   `-- experiments.py
|-- tests/
|   |-- test\_knn.py
|   |-- test\_metrics.py
|   `-- test\_splits.py
|-- pytest.ini
|-- requirements.txt
|-- README.md
`-- pledge.txt
```

## Environment setup

Python 3.11 is required by the assignment. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the tests

From the repository root:

```bash
pytest -q
```

The audited ZIP passed all included tests:

```text
21 passed
```

## Reproduce the analysis

Open and run the notebooks in this order from the `notebooks/` directory:

1. `01\_eda.ipynb`
2. `02\_evaluation.ipynb`
3. `03\_cross\_validation.ipynb`

Run the runtime benchmark from the repository root with:

```bash
python -m src.experiments
```

The module form is important because `src/experiments.py` imports `src.knn` as a package module.

## Dataset and preprocessing

The selected model features are:

```text
pclass, sex, age, sibsp, parch, fare, embarked
```

The target is:

```text
survived
```

The exploratory feature matrix has 1,309 rows and 10 encoded columns. Missing numeric values in `age` and `fare` are imputed with training-subset medians during evaluation. Missing `embarked` values are imputed with the training-subset mode. The unordered categorical features `sex` and `embarked` are one-hot encoded.

## Reproducibility note

All random operations in the submitted implementation use seed `42`. Titanic contains duplicate or exactly equidistant encoded rows with different labels, so a KNN implementation and scikit-learn can make different choices when nearest-neighbor distances are tied. Before submission, rerun the notebooks on the final environment, save the refreshed outputs and figures, and explain any remaining tie-related differences honestly in the report.

## Bonus claim

This submission claims **Bonus 1 - Weighted KNN (+5)**. With `weights="distance"`, each neighbor contributes a weight of:

```text
1 / (distance + 1e-8)
```

The included tests verify that closer neighbors receive more influence, weighted regression uses a weighted average, and distance weighting reduces to uniform weighting when all selected neighbors are equidistant.

