# AML 2026 – Project 1: predicting age from MRI features

## Goal

- **Task:** predict a person's age `y` (42–97 years) from 832 anonymized features extracted from brain MRI scans.
- **Data:** `X_train.csv` (1212 rows), `y_train.csv`, `X_test.csv` (776 rows), all with `id` as index. Submissions follow `sample.csv`, with columns `id,y`.
- **Metric:** R², via `sklearn.metrics.r2_score(y, y_pred)`.
- **The organizers perturbed the data.** The report must document three preprocessing steps (outlier detection, feature selection, imputation of missing values) plus a short description of the regression model.
- **Reference point:** other teams reach about R² = 0.82. Our best single model is currently at 0.69.

## Folder layout

| Path | Contents |
|---|---|
| `data_prep.ipynb` | the full preprocessing; writes the prepared CSVs below |
| `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample.csv` | original data |
| `X_train_median.csv`, `X_test_median.csv` | prepared data, gaps filled with the median |
| `X_train_knn.csv`, `X_test_knn.csv` | prepared data, gaps filled with kNN. **The default input for models** |
| `plots/fs2_comparison.png` | the feature-selection validation figure (shown in `data_prep.ipynb`) |
| `ridge_cc.ipynb`, `submission_ridge.csv` | a teammate's Ridge baseline on the raw data, in French (nested CV R² ≈ 0.39) |
| `SVR/SVR.ipynb`, `SVR/test_svr.csv` | tuned SVR, and its test predictions |
| `Stacking/folds.ipynb`, `Stacking/folds.csv` | the shared CV split used by every model |
| `Stacking/oof_svr.csv` | the SVR's out-of-fold predictions |

**Paths:** model notebooks live in subfolders and read the data with `../X_train_knn.csv` and similar. The conda environment is `AML`.

## Data preparation (`data_prep.ipynb`)

**How the notebook works:** each step overwrites `X_train`, `X_test` and `y_train`, so **run it top to bottom** (Restart & Run All), and don't run a single cell twice. Temporary median-filled copies are made only where a method can't handle NaNs. The original NaNs are kept until the final imputation.

| Step | What | Result |
|---|---|---|
| 1. Load | `id` as index | train 1212 × 832 (7.6% NaN), test 776 × 832 (6.3% NaN) |
| 2. Constant columns | found on train, dropped from both | `x104, x129, x489, x530` → 828 features |
| 3. Outlier removal | see below | **60 training rows removed → 1152 rows**. Test rows are never removed |
| 4. Feature selection | see below | **216 features** |
| 5. Imputation | median and kNN, both fitted on the 1152 kept rows | two final datasets, 1152 × 216 and 776 × 216 |

### Outlier removal: logic and results

- **Principle: remove the training rows that are unlike anything in the test set.** The model only has to predict test-like rows.
- **What the PCA overview shows** (robust-scaled, PCA fitted on train, test projected into it):
  - **The test/train variance ratio is 0.69 on PC1, 0.65 on PC2 and 0.98 on PC3.** Train has extra spread along PC1/PC2: a **"ring" of rows around the main cloud with no test rows nearby.**
  - **A second group, the "bump" (~10% of rows, far out along PC3), exists in both train and test.** It is therefore **kept**.
- **Comparison of 216 frameworks plus "no removal":**
  - scaler: robust or standard;
  - detection space: 2 or 3 PCs;
  - detector: robust distance, robust covariance, Isolation Forest, LOF, One-Class SVM, or kNN distance to the test set;
  - cut-off rule: furthest test row, 99.5/99/98% of test rows, or a fixed 2–6% of training rows.
- **How the frameworks were scored:** precision and recall against the test set, in a fixed space (robust-scaled first 3 PCs, each scaled by its MAD, neighbourhoods of 5 points).
  - **Precision:** the share of kept training rows that lie near test rows.
  - **Recall:** the share of test rows still covered by training rows.
  - **Ranking:** recall within 0.005 of "no removal" first, then the highest precision, then the fewest rows removed.
- **Chosen: robust scaler | 3 PCs | kNN distance to test | cut at the 99.5% of test rows.**
  - **60 rows removed;** precision 0.950 → 0.992, recall 0.991 → 0.987.
  - **Why kNN distance to the test set:** centred detectors miss the ring. The PC3 bump stretches any single-radius border, which then swallows the ring. A kNN distance to the test set follows the shape of the test cloud.

### Feature selection: logic and results

- **FS1, how much signal:**
  - each feature's |Spearman ρ| with `y` (NaNs skipped, no filling), against the same with `y` shuffled 20 times;
  - the noise threshold is the 99th percentile of the shuffled values, **|ρ| > 0.078**;
  - **216 of 828 features pass.** The rest sits exactly on the noise distribution, i.e. features added as noise;
  - a near-duplicate check (|r| > 0.95) dropped nothing.
- **FS2, validation** (results recorded in a markdown cell; the code is in the appendix and takes about 15 minutes):
  - 4 rankings (Spearman, F-test, mutual information, random forest) × k ∈ {100, 150, 200, 216, 250, 300, 400} plus all features;
  - judge: default `HistGradientBoostingRegressor`; CV: 5 folds × 3 repeats; ranking refitted inside each fold;
  - **FS1 selection: 0.5946 ± 0.0136. Best option: 0.5991 (Spearman, k = 200). All features: 0.5603.**
  - So FS1 ties with the best option and beats no selection by +0.034.
- **Decision:** keep FS1's 216 features. It's a fixed statistical rule, not the maximum of a noisy CV curve.
- **Confirmed later with the tuned SVR** (selection refitted inside each fold): k = 100 → 0.668, 150 → 0.672, **216 → 0.685**, 300 → 0.598, 400 → 0.557, 828 → 0.513. SVR is far more sensitive to noise features than tree models.

### Imputation: logic and results

- **Median:** the medians of the 1152 kept rows, applied to train and test.
- **kNN reconstruction test:**
  - hide 7% of the *known* values (about the real missing rate), 3 repeats, standardized units;
  - **RMSE: median 1.011, kNN k = 5 0.645** (best; k = 7 0.646, k = 30 0.674).
- **Final kNN fill:** `StandardScaler` → `KNNImputer(n_neighbors=5, weights="distance")`, both fitted on train, then converted back to the original units. Only the gaps change.
- **With the tuned SVR:** median 0.658 vs **kNN 0.694**, a gain of +0.036 ± 0.004, with kNN better in all 5 folds. → **Use the kNN files.**

## Modelling

### Results so far

| Model | Data | CV R² | Notes |
|---|---|---|---|
| Ridge | raw, own pipeline | ≈ 0.39 (nested CV) | teammate's baseline |
| HistGradientBoosting (default) | 216 features, NaNs | 0.595 | only the judge for feature selection |
| **SVR, RBF kernel (tuned)** | `X_train_knn` | **0.6926 ± 0.009; out-of-fold 0.6935** | `C ≈ 9.7`, `gamma ≈ 0.005` (≈ 1/216), `epsilon ≈ 0.001`. The plateau is flat for C between 7 and 45. The model is `TransformedTargetRegressor(StandardScaler → SVR)` with `y` standardized |

### Tested with the SVR and rejected

- **A separate model for the bump rows:** they're predicted almost as well as the rest (RMSE 5.6 vs 5.1 years; 11.4% of the squared error from 9.6% of the rows). The most it could gain is +0.006.
- **Quantile clipping** (0.1–5%): at most +0.003, within noise. The RBF kernel already caps the influence of extreme values.
- **Changing the number of features:** 216 is best (see above).

**Conclusion:** the remaining gap to 0.82 is spread evenly over all rows (about 5 years of error each), not concentrated in a subgroup.

### Models still to train

| Model | Status | Notes |
|---|---|---|
| SVR | ✅ done | |
| XGBoost | in progress (Cyprien) | tune `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_*` with a random search |
| MLP | to do | plain PyTorch `nn.Module` with a hand-written CV loop. Use `HuberLoss`, dropout, weight decay and early stopping. Re-check k around 216 |
| Extra Trees | to do | adds diversity to the stack. Tune `n_estimators`, `max_features`, `min_samples_leaf` |
| Gaussian process with ARD kernel | optional, promising | learns a weight per feature, which suits the noise sensitivity seen with SVR |

**Not suitable:**
- **CNN:** the columns have no spatial order.
- **Transformer trained from scratch:** too little data (1152 rows).

## Stacking

**Shared split:**
- `Stacking/folds.csv` maps each training `id` to a fold from 0 to 4.
- It was built with `KFold(n_splits=5, shuffle=True, random_state=42)` on the row order of `X_train_knn.csv`. That's the same split the SVR used.
- Load it with `.loc[X.index, "fold"]`, so rows are matched by `id`, not by position.

**What every model produces:**
1. **Out-of-fold predictions:** for each fold, train on the other 4 and predict that fold. That gives one prediction per training row (1152 rows), saved as `Stacking/oof_<model>.csv`.
2. **Test predictions:** refit on all 1152 rows, predict the 776 test rows, and save as `<ModelFolder>/test_<model>.csv` in `sample.csv` format (`id,y`).

**Every model must use:**
- the same 1152 rows and the same `y`;
- the same folds.

The input features may differ between models.

**Meta-model:** `LinearRegression(positive=True)` fitted on the out-of-fold columns, then applied to the test predictions.
- **Keep the stack** only if its out-of-fold R² beats the best single model.
- **If the weights come out roughly equal,** a plain average is simpler and just as good.

## Next steps

1. **The teammate finishes XGBoost** with the shared folds and saves `oof_xgb.csv` and `test_xgb.csv`.
2. **Train the MLP and Extra Trees** the same way, and optionally a Gaussian process with ARD.
3. **Build the stacking notebook:** load all out-of-fold files, fit the weights, compare with the best single model, and write the submission in `sample.csv` format.
4. **Write the report:**
   - every step with its numbers: outliers (60 rows, precision/recall), feature selection (216 features, FS2 table and plot), imputation (RMSE 0.645 vs 1.011, R² +0.036);
   - the models and the stack.

## Caveats

- **The CV scores are slightly optimistic,** because the 216 features were selected on all 1152 rows. Measured for the SVR, the effect is about 0.008 (0.693 vs 0.685 with the selection inside the folds).
- **Differences below about 0.01–0.02 R² are noise** (SE ≈ 0.01). Compare models fold by fold, on the same folds.
- **Check before writing code:**
  - which earlier cells a cell depends on, since cells get deleted;
  - the paths, since notebooks live in subfolders;
  - the output format, which must be `id,y` as in `sample.csv`.
