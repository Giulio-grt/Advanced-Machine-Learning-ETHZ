# AML 2026 – Project 1: predicting age from MRI features

## Goal

- **Task:** predict a person's age `y` (42–97 years) from 832 anonymized features extracted from brain MRI scans.
- **Data:** `X_train.csv` (1212 rows), `y_train.csv`, `X_test.csv` (776 rows), all with `id` as index. Submissions follow `sample.csv`, with columns `id,y`.
- **Metric:** R², via `sklearn.metrics.r2_score(y, y_pred)`.
- **The organizers perturbed the data.** The report must document three preprocessing steps (outlier detection, feature selection, imputation of missing values) plus a short description of the regression model.
- **Reference point:** other teams reach about R² = 0.82. Our best single model is the SVR on `log_knn`: out-of-fold R² 0.708. The earlier SVR on `knn` (out-of-fold 0.694) scored 0.7178 on the public leaderboard.

## Folder layout

| Path | Contents |
|---|---|
| `data_prep.ipynb` | the full preprocessing; writes the prepared CSVs below |
| `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample.csv` | original data |
| `X_train_knn.csv`, `X_test_knn.csv` | prepared data, gaps filled with kNN (used by XGBoost and `folds.ipynb`) |
| `X_train_log_knn.csv`, `X_test_log_knn.csv` | the same, with the log of the 11 right-skewed features taken before the kNN fill. **The default input for new models** |
| `plots/fs2_comparison.png` | the feature-selection validation figure (shown in `data_prep.ipynb`) |
| `SVR/SVR.ipynb`, `SVR/submission_svr.csv` | tuned SVR on `log_knn`, and its test predictions |
| `XGboost_ridge/Ridge_vs_XGB.ipynb` | a teammate's Ridge and XGBoost on `knn`; writes `submission_ridge_knn.csv`, `submission_xgboost_knn.csv` and `Stacking/oof_xgb.csv` |
| `Gaussian_process/GP.ipynb` | Gaussian process (to do) |
| `Stacking/folds.ipynb`, `Stacking/folds.csv` | the shared CV split used by every model, and the code every model pastes at its end |
| `Stacking/oof_svr.csv`, `Stacking/oof_xgb.csv` | out-of-fold predictions of each model |
| `Stacking/stacking.ipynb`, `Stacking/submission_stacking.csv` | the meta-model (in French), and the stacked submission |

**Paths:** model notebooks live in subfolders and read the data with `../X_train_log_knn.csv` and similar. The conda environment is `AML`. **Don't move, rename or delete files or folders without the user's approval:** teammates work with this layout.

## Data preparation (`data_prep.ipynb`)

**How the notebook works:** each step overwrites `X_train`, `X_test` and `y_train`, so **run it top to bottom** (Restart & Run All), and don't run a single cell twice. Temporary median-filled copies are made only where a method can't handle NaNs. The original NaNs are kept until the final imputation.

| Step | What | Result |
|---|---|---|
| 1. Load | `id` as index | train 1212 × 832 (7.6% NaN), test 776 × 832 (6.3% NaN) |
| 2. Constant columns | found on train, dropped from both | `x104, x129, x489, x530` → 828 features |
| 3. Outlier removal | see below | **60 training rows removed → 1152 rows**. Test rows are never removed |
| 4. Feature selection | see below | **216 features** |
| 5. Imputation | kNN (k = 5), fitted on the 1152 kept rows | `knn`: 1152 × 216 and 776 × 216 |
| 6. Log transform | log of the 11 right-skewed features, then the same kNN fill in log units | `log_knn`: 1152 × 216 and 776 × 216 |

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

- **Median fill:** now only a baseline, in the reconstruction test and the SVR comparison below. Its files and notebook cell were deleted.
- **kNN reconstruction test:**
  - hide 7% of the *known* values (about the real missing rate), 3 repeats, standardized units;
  - **RMSE: median 1.011, kNN k = 5 0.645** (best; k = 7 0.646, k = 30 0.674).
- **Final kNN fill:** `StandardScaler` → `KNNImputer(n_neighbors=5, weights="distance")`, both fitted on train, then converted back to the original units. Only the gaps change.
- **With the tuned SVR:** median 0.658 vs **kNN 0.694**, a gain of +0.036 ± 0.004, with kNN better in all 5 folds. → **Use the kNN files.**

### Log transform: logic and results

- **Rule:** the log of every feature with skew > 1 on the kept training rows and strictly positive in train and test. That's 11 features: `x115, x159, x415, x458, x507, x531, x542, x641, x665, x702, x742`.
- **Why:**
  - these look like volumes, which change by percentages with age;
  - the log makes them more linear in age (`x115`: r = 0.46 → 0.62; `x641`: 0.37 → 0.51);
  - without it, their long right tails squeeze most rows together after scaling.
- **Order:** the log comes after feature selection and before the kNN fill, so the gaps are filled in log units. The log keeps the order of the values, so the Spearman selection doesn't change. `k = 5` comes from the reconstruction test in the original units.
- **With the SVR** (same hyperparameters, same folds): `knn` vs `log_knn`, **+0.0148 ± 0.0041 (SE), better in 5/5 folds**. After re-tuning on `log_knn`: CV 0.7074, out-of-fold 0.7083.
- **Trees (XGBoost):** no effect expected, because a monotone transform doesn't change their splits. That's why XGBoost stays on `knn`.

## Modelling

### Results so far

| Model | Data | CV R² | Notes |
|---|---|---|---|
| Ridge | raw, own pipeline | ≈ 0.39 (nested CV) | teammate's first baseline (notebook since deleted) |
| Ridge | `X_train_knn` | 0.582 | teammate's, `alpha ≈ 174` |
| HistGradientBoosting (default) | 216 features, NaNs | 0.595 | only the judge for feature selection |
| XGBoost (tuned) | `X_train_knn` | 0.678; out-of-fold 0.6787 | teammate's: `learning_rate 0.1`, `max_depth 3`, `min_child_weight 5`, `n_estimators 500`, `colsample_bytree 0.6`, `subsample 0.8`, `reg_alpha 1`, `reg_lambda 1`. Training R² 0.996. Early stopping didn't help (0.633) |
| SVR, RBF kernel (tuned) | `X_train_knn` | 0.6926 ± 0.009; out-of-fold 0.6935 | public leaderboard 0.7178 |
| **SVR, RBF kernel (tuned)** | `X_train_log_knn` | **0.7074 ± 0.0097; out-of-fold 0.7083** | `C ≈ 9.7`, `gamma ≈ 0.005` (≈ 1/216), `epsilon ≈ 0.001`. The plateau is flat for C between 7 and 45. The model is `TransformedTargetRegressor(StandardScaler → SVR)` with `y` standardized |
| Stack SVR + XGBoost | out-of-fold predictions | 0.716 (simple average 0.714) | with the SVR on `knn`. Weights: SVR 0.60, XGBoost 0.47, intercept −4.9. **To re-run with the SVR on `log_knn`** |

### Tested with the SVR and rejected

- **A separate model for the bump rows:** they're predicted almost as well as the rest (RMSE 5.6 vs 5.1 years; 11.4% of the squared error from 9.6% of the rows). The most it could gain is +0.006.
- **Quantile clipping** (0.1–5%): at most +0.003, within noise. The RBF kernel already caps the influence of extreme values.
- **Changing the number of features:** 216 is best (see above).

**Conclusion:** the remaining gap to 0.82 is spread evenly over all rows (about 5 years of error each), not concentrated in a subgroup.

### Models still to train

| Model | Status | Notes |
|---|---|---|
| SVR | ✅ done | on `log_knn` |
| XGBoost | ✅ done (Cyprien) | on `knn` |
| Gaussian process with ARD kernel | **next** (`Gaussian_process/GP.ipynb`) | on `log_knn`. Learns a weight per feature, which suits the noise sensitivity seen with SVR |
| MLP | to do | plain PyTorch `nn.Module` with a hand-written CV loop. Use `HuberLoss`, dropout, weight decay and early stopping. Re-check k around 216 |
| Extra Trees | to do | adds diversity to the stack. Tune `n_estimators`, `max_features`, `min_samples_leaf` |

**Not suitable:**
- **CNN:** the columns have no spatial order.
- **Transformer trained from scratch:** too little data (1152 rows).

## Stacking

**Shared split:**
- `Stacking/folds.csv` maps each training `id` to a fold from 0 to 4.
- It was built with `KFold(n_splits=5, shuffle=True, random_state=42)` on the row order of `X_train_knn.csv`. That's the same split the SVR used. `X_train_log_knn.csv` has the same ids in the same order.
- Load it with `.loc[X.index, "fold"]`, so rows are matched by `id`, not by position.

**What every model produces** (the code to paste is in `folds.ipynb`):
1. **Out-of-fold predictions:** for each fold, train on the other 4 and predict that fold. That gives one prediction per training row (1152 rows), saved as `Stacking/oof_<model>.csv`.
2. **Test predictions:** refit on all 1152 rows, predict the 776 test rows, and save as `<ModelFolder>/submission_<model>.csv` in `sample.csv` format (`id,y`). XGBoost's file is `XGboost_ridge/submission_xgboost_knn.csv`.
3. **One line in `MODELS`** (cell 2 of `stacking.ipynb`): the model's name and the path to its test predictions.

**Every model must use:**
- the same 1152 rows and the same `y`;
- the same folds.

The input features may differ between models.

**Meta-model (`stacking.ipynb`):**
- `LinearRegression()` (free coefficients and an intercept), fitted on the out-of-fold columns, then applied to the test predictions.
- It's scored on the shared folds against each model and the simple average.
- It writes `Stacking/submission_stacking.csv`.
- **Keep the stack** only if its out-of-fold R² beats the best single model. Otherwise, upload that model's `submission_<model>.csv`.

## Next steps

1. **Re-run `Stacking/stacking.ipynb`.** The SVR files now come from `log_knn`, but the current `submission_stacking.csv` still uses the old SVR.
2. **Gaussian process** on `log_knn`, with the `folds.ipynb` code at the end and one line in `MODELS`.
3. **Optionally, the MLP and Extra Trees** the same way.
4. **Write the report:**
   - every step with its numbers: outliers (60 rows, precision/recall), feature selection (216 features, FS2 table and plot), imputation (RMSE 0.645 vs 1.011, R² +0.036), log transform (11 features, +0.015 with the SVR);
   - the models and the stack.

## Caveats

- **The CV scores are slightly optimistic,** because the 216 features were selected on all 1152 rows. Measured for the SVR, the effect is about 0.008 (0.693 vs 0.685 with the selection inside the folds).
- **Differences below about 0.01–0.02 R² are noise** (SE ≈ 0.01). Compare models fold by fold, on the same folds.
- **Check before writing code:**
  - which earlier cells a cell depends on, since cells get deleted;
  - the paths, since notebooks live in subfolders;
  - the output format, which must be `id,y` as in `sample.csv`.
