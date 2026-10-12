# Predicting Mortgage-Backed Securities Prepayment with Machine Learning

UE24CS352A Machine Learning - Mini-Project. Loan-level binary classification: *will this loan be prepaid this month?*
Models compared: Logistic Regression (none / L1 / L2), SVM (poly / RBF / sigmoid), Gaussian Discriminant Analysis (linear / quadratic), Feed-forward Neural Network.
Based on C. Guo, J. Boccio & C. Cameron, [*Predicting Mortgage Backed Securities Prepayment Using Machine Learning Methods*](https://cs229.stanford.edu/proj2021spr/report2/81995903.pdf), CS229 project report, Stanford University, Spring 2021 (course project, not peer-reviewed).

**Repository:** https://github.com/gaurav-2030/mbs-prepayment

**Team:** Bhoomika Kshatriya (PES1UG24AM068), Gaurav Girish (PES1UG24AM103) &nbsp;|&nbsp; Section B &nbsp;|&nbsp; Problem #55

**Report:** [`outputs/PES1UG24AM068_PES1UG24AM103.pdf`](outputs/PES1UG24AM068_PES1UG24AM103.pdf) &nbsp;|&nbsp; **Slides:** [`outputs/MBS_Prepayment_Deck.pptx`](outputs/MBS_Prepayment_Deck.pptx)

## Setup
Requires Python 3.11-3.13 (tested on 3.13). Library versions are pinned so results reproduce exactly.
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: py -m venv .venv  then  .venv\Scripts\activate
pip install -r requirements.txt
```
After activation, `python` points at the virtual environment on every OS, so the commands below use `python`.

## Run
```bash
# 1) Pipeline check on bundled synthetic data (no download, ~3 min on a laptop)
python -m src.train --data synthetic

# 2) Real data: Freddie Mac Single-Family Loan-Level Dataset (free registration)
#    https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset  -> download "sample" files
#    unzip so data/freddie/ contains sample_orig_YYYY.txt and sample_svcg_YYYY.txt
python -m src.train --data freddie --folder data/freddie --loans 5000
#    macro features (needed for rate_incentive): build data/macro.csv from public FRED series -
#    Freddie Mac 30-yr mortgage rate, FHFA house price index (12-month % change), BLS unemployment
python -m src.macro
python -m src.train --data freddie --folder data/freddie --macro data/macro.csv

# 3) Live demo with the saved best model (run step 1 or 2 first: it creates outputs/model.joblib)
python -m src.demo --random 6                  # held-out TEST loan-months, half of them prepaid
python -m src.demo --loan examples/loan.json   # score one custom loan (works on every OS)
python -m src.demo --loan '{"credit_score":770,"orig_interest_rate":7.0,"mtg_rate":4.5,"current_upb":60000,"loan_age":48}'   # macOS/Linux inline JSON

# 4) Regenerate the PDF report (outputs/PES1UG24AM068_PES1UG24AM103.pdf) from the latest results
python report/build_report.py

# 5) Regenerate the slide deck (needs Node.js 18+)
npm install
npm run deck

# 6) Tests: data-pipeline guarantees (no loan in two splits, labels, leakage fix, 30-year filter, macro builder)
python -m pytest

# 7) Exploratory data analysis notebook (already executed; GitHub shows it with all plots)
pip install notebook==7.6.3
jupyter notebook notebooks/eda.ipynb

# 8) Hyper-parameter sweep on the validation split (~1 min): C for L1/L2 logistic and RBF SVM, reg_param for QDA
python -m src.tune
```
Outputs land in `outputs/`: `results.csv/json`, `confusion_matrices.png`, `best_confusion.png`, `metrics_bar.png`, `roc_pr.png`, `feature_importance.png`, `pca.png`, `tuning.csv`, `tuning_test.csv`, `tuning.png`, `model.joblib`, `PES1UG24AM068_PES1UG24AM103.pdf`, `MBS_Prepayment_Deck.pptx`.

## Method summary
- **Label:** 1 when the loan's zero-balance code is `01` (voluntary payoff) in that reporting month. Code `01` also covers *maturity*, so only 30-year (360-month) loans are kept.
- **Split:** 60/20/20 grouped **by loan** so no loan appears in two splits.
- **Leakage control:** Freddie Mac reports current UPB = 0 in the payoff month, so the loader uses the balance at the *start* of the month (previous month's UPB).
- **Imbalance (~1:67 on the synthetic run):** accuracy is misleading, so we report precision, recall, F1, ROC-AUC, PR-AUC; decision thresholds are tuned on the validation split only.
- **SVMs** train on a subset (kernel matrix is O(n^2)); **NN** (MLP 128-256-128, Adam) oversamples positives 10x in place of the paper's positive loss weight of 10.

## Results & Analysis
All numbers are from the synthetic run (`python -m src.train --data synthetic`, test split: 35,720 loan-months, 593 prepayments). Full table: `outputs/results.csv`.

**How good is the best model?** Logistic regression with L1 reaches a test PR-AUC of **0.084** and ROC-AUC of **0.801**. A random ranking would score a PR-AUC equal to the prepayment rate, **0.0166**, so the model ranks prepayments about **5x better than chance**. At its tuned threshold it catches **182 of 593** prepayments (recall 0.307) with precision 0.092: roughly one alert in eleven is a real prepayment (1,803 false alarms).

**Why is accuracy (94%) below the "never prepay" baseline (98.5%)?** Each threshold is set on the validation split to maximise F1, which deliberately trades some accuracy for finding prepayments. A model tuned for accuracy would simply predict "no prepayment" every time and catch none, so accuracy is the wrong yardstick at 1:67.

**Why do the linear models win?**
- The four linear models (logistic none/L1/L2 and GDA linear) are essentially tied (F1 0.141, PR-AUC 0.082-0.084).
- The flexible models do worse: QDA 0.063, RBF SVM 0.066, polynomial SVM 0.058, the neural network 0.042 and the sigmoid SVM 0.028.
- The reason is the data: the EDA notebook shows one dominant, monotone driver (`rate_incentive`: 0.07% to 16.2% monthly prepayment across its range) plus small additive effects, which a linear decision boundary already captures.
- With only 1,765 prepayments in the training split, the extra flexibility of QDA, kernel SVMs and the neural network mostly adds variance. The SVMs also see only an 8,000-row subset, and the network trains on 10x-oversampled positives.

**What does regularisation do?** (`python -m src.tune`; values chosen on validation, then scored once on test, in `outputs/tuning*.csv` and `tuning.png`)
- **Logistic L1/L2:** insensitive. Validation PR-AUC is 0.0800-0.0802 for any C >= 0.1. Only very strong penalties hurt (C=0.001: 0.073 for L1), and the C chosen on validation does not change the test score (0.0837 vs 0.0839).
- **GDA (Quadratic) needs strong shrinkage.** With `reg_param=0` it collapses (validation PR-AUC 0.023) because several features are exact duplicates (`loan_age` vs `months_to_maturity`, r = -1.0; original vs current rate, r = 1.0), so the per-class covariance matrices are singular. The best value, `reg_param=0.6`, lifts its test PR-AUC from 0.063 to **0.078**.
- **RBF SVM prefers a softer margin.** C = 0.1 gives validation PR-AUC 0.070 vs 0.059 at the default C = 1; on test 0.070 vs 0.068 (the SVM subset is drawn differently from the main run, which scored 0.066).
- Even tuned, QDA and the SVM stay below logistic regression, so the ranking and the conclusions do not change. `src/models.py` keeps its original settings so the write-up and deck match the reported run.

**Why did the reference paper report 99.9% accuracy?** Its most important feature was `current_UPB`, with importance 974 against 2.7 for the next one. Freddie Mac reports the balance as 0 in the payoff month, so that feature gives the label away. Our loader uses the balance at the start of the month, which is why our scores are realistic rather than near-perfect.

**Limitations.**
- These are synthetic-data results: the label was generated from a logistic hazard, which favours logistic regression by construction.
- Splits are random by loan, not by time, so a model is not tested on a later market period than it was trained on.
- Kernel SVMs are trained on a subset because the kernel matrix grows as O(n^2).

## Work division
| | Gaurav Girish (PES1UG24AM103) | Bhoomika Kshatriya (PES1UG24AM068) |
|---|---|---|
| Area | Data pipeline, evaluation and demo | Models, analysis and reporting |
| Code | `src/data.py`, `src/macro.py`, `src/train.py` (split, thresholds, metrics, plots), `src/demo.py`, `tests/` | `src/models.py`, `src/tune.py`, `notebooks/eda.ipynb`, `report/`, README "Results & Analysis" |
| Slides | 1 Title, 2 Problem, 3 Dataset, 4 Data pipeline, 10 Live demo | 5 Approach, 6 Results, 7 Metrics, 8 Insights, 9 EDA and tuning, 11 Conclusions |

## Structure
```
src/data.py     loader (Freddie Mac + synthetic), grouped split
src/models.py   model zoo
src/train.py    train / evaluate / plots / save best model
src/demo.py     live scoring demo
src/macro.py    macro-feature builder (FRED -> data/macro.csv)
src/tune.py     hyper-parameter sweep (chosen on validation, checked once on test)
tests/          pytest suite for the data pipeline and macro builder
notebooks/      eda.ipynb: class balance, prepayment drivers, feature correlations
examples/       sample loan for the demo
report/         PDF write-up (build_report.py) and slide deck (build_deck.js) generators
```
