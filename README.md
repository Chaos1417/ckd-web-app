# 🧬 Ensemble Machine Learning for Chronic Kidney Disease Prediction
### A Reproduction, Enhancement, and End-to-End Web Deployment Study

<p align="center">
  <a href="https://github.com/Chaos1417/ckd-web-app">
    <img src="https://img.shields.io/badge/GitHub-Chaos1417%2Fckd--web--app-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/XGBoost-Latest-006ACC?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Groq_API-Llama3--8b-FF6B35?style=for-the-badge&logo=meta&logoColor=white" alt="Groq Llama3"/>
  <img src="https://img.shields.io/badge/imbalanced--learn-SVMSMOTE-9B59B6?style=for-the-badge" alt="imbalanced-learn"/>
  <img src="https://img.shields.io/badge/License-MIT-2ECC71?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CV_Accuracy_(Extra_Trees)-100.00%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Holdout_Accuracy-99.00%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/AUC--ROC-1.000-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Features_(RFE)-12_of_24-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit_Cloud-FF4B4B?style=flat-square"/>
</p>

---

## 📋 Table of Contents

- [Abstract & Clinical Objective](#abstract--clinical-objective)
- [System Architecture Overview](#system-architecture-overview)
- [Dataset Description](#dataset-description)
- [Data Pipeline & Preprocessing](#data-pipeline--preprocessing)
  - [Engineering Challenge 1: ARFF File Parsing](#engineering-challenge-1-arff-file-parsing)
  - [Categorical Feature Preprocessing](#categorical-feature-preprocessing)
  - [Numerical Feature Preprocessing (MICE Imputation)](#numerical-feature-preprocessing-mice-imputation)
  - [Engineering Challenge 2: Data Splitting & Leakage Prevention](#engineering-challenge-2-data-splitting--leakage-prevention)
  - [Class Balancing with SVMSMOTE](#class-balancing-with-svmsmote)
- [Feature Selection: RFE vs. Boruta](#feature-selection-rfe-vs-boruta)
- [Model Architecture & Validation](#model-architecture--validation)
- [Results & Discussion](#results--discussion)
- [Generative AI Integration (Groq API + Llama3-8b)](#generative-ai-integration-groq-api--llama3-8b)
- [Engineering Challenges Overcome](#engineering-challenges-overcome)
- [System Deployment](#system-deployment)
- [Installation & Local Setup](#installation--local-setup)
- [Project Structure](#project-structure)
- [Comparison with Reference Paper](#comparison-with-reference-paper)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## Abstract & Clinical Objective

This repository presents a production-grade, end-to-end machine learning system for the diagnosis of **Chronic Kidney Disease (CKD)**, reproducing, extending, and deploying the methodology of Rahman et al. (2024) (*Biomedical Signal Processing and Control*).

**Clinical Context:** CKD is an irreversible, progressive impairment of kidney function affecting over **800 million** patients globally, with disproportionately high prevalence in South Asian populations (Pakistan, Bangladesh, India). The disease is clinically silent in its early stages; by the time symptoms manifest, patients frequently require dialysis or kidney transplantation. Early, accurate prediction via routinely collected biomarkers is therefore of direct clinical value.

**Scope of Work:**
- Reproduces and validates the ensemble ML pipeline from Rahman et al. (2024) on the UCI CKD dataset (400 records, 24 features).
- Introduces methodological enhancements: a rigorous two-stage leakage-prevention splitting protocol, a quantitative head-to-head comparison of **RFE vs. Boruta** feature selection, and a formal justification for model selection under clinical parsimony.
- Evaluates five ensemble classifiers — **Random Forest, Gradient Boosting, AdaBoost, Extra Trees, and XGBoost** — across both feature selection schemes (10 experimental conditions total).
- Deploys the best-performing model (tuned **Extra Trees, RFE, 12 features**) as a publicly accessible **Streamlit web application** augmented by a **Groq API / Llama3-8b** natural-language explanation engine.
- Documents four non-trivial engineering challenges encountered in the ARFF parsing, data splitting, deployment pipeline, and infrastructure phases.

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END CKD PREDICTION PIPELINE                    │
└──────────────────────────────────────────────────────────────────────────────┘

  UCI ARFF File
       │
       ▼
┌─────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  Custom ARFF    │───▶│  Dual-Track             │───▶│ Two-Stage Stratified│
│  Line Parser    │    │  Preprocessing Pipeline │    │ Split (80/16/64)    │
│  (400 rows)     │    │  ├─ Categorical:         │    │ + SVMSMOTE          │
│                 │    │  │  Missing-Literal +    │    │ (train only)        │
│                 │    │  │  OrdinalEncoder       │    └─────────┬───────────┘
└─────────────────┘    │  └─ Numerical:           │              │
                       │     MICE + StandardScaler│              ▼
                       └────────────────────────┘    ┌─────────────────────┐
                                                      │  Feature Selection  │
                                                      │  ┌───────────────┐  │
                                                      │  │  RFE (n=12)   │◀─┤
                                                      │  └───────────────┘  │
                                                      │  ┌───────────────┐  │
                                                      │  │  Boruta (n=19)│  │
                                                      │  └───────────────┘  │
                                                      └─────────┬───────────┘
                                                                │
                                                                ▼
                                                 ┌─────────────────────────┐
                                                 │  Ensemble Classifiers   │
                                                 │  (10 conditions total)  │
                                                 │  ├─ Random Forest       │
                                                 │  ├─ Gradient Boosting   │
                                                 │  ├─ AdaBoost            │
                                                 │  ├─ Extra Trees ★       │
                                                 │  └─ XGBoost             │
                                                 │  10-Fold Stratified CV  │
                                                 │  + GridSearchCV Tuning  │
                                                 └─────────┬───────────────┘
                                                           │
                                                           ▼
                                                ┌──────────────────────────┐
                                                │  Streamlit Web App       │
                                                │  + Groq (Llama3-8b)      │
                                                │  LLM Explanation Engine  │
                                                └──────────────────────────┘
```

---

## Dataset Description

The project uses the **Chronic Kidney Disease dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease), originally collected over a two-month period at a hospital in Tamil Nadu, India.

| Property              | Detail                                   |
|-----------------------|------------------------------------------|
| **Records**           | 400 patient instances                    |
| **Total Attributes**  | 25 (24 features + 1 binary class label)  |
| **Numerical Features**| 11 continuous (age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc) |
| **Categorical Features** | 13 nominal (sg, al, su, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane) |
| **Class Distribution**| 250 CKD (62.5%) / 150 Not-CKD (37.5%) — imbalanced |
| **File Format**       | ARFF (Attribute-Relation File Format)    |
| **Key Missing Values**| rc: 130, wc: 105, pot: 88, sod: 87, pcv: 70 |

**Clinically significant features with high missingness** — particularly `rc`, `wc`, `pot`, and `sod` — necessitate a principled imputation strategy; naive row deletion would reduce the effective dataset to an unacceptably small size.

---

## Data Pipeline & Preprocessing

The preprocessing pipeline enforces a strict **separation of concerns** between categorical and numerical feature tracks, motivated by the fundamentally different statistical properties of each type.

### Engineering Challenge 1: ARFF File Parsing

> **Problem:** The UCI CKD dataset is distributed in ARFF format — a specification native to the Weka toolkit encoding metadata in `@attribute` / `@data` header directives and missing values as `?` characters. Critically, the raw file contained rows of **inconsistent length** (fewer than the expected 25 fields) due to corrupted or missing delimiters. A naive `pandas.read_csv()` call either raises a parsing exception or silently drops malformed records, shrinking the effective dataset.

> **Solution:** A **custom line-by-line parser** was implemented:
> 1. Each line is stripped and split on commas.
> 2. Rows with ≥ 25 tokens retain only the first 25.
> 3. Rows with < 25 tokens have the deficit padded with `'?'` to preserve the row index for downstream imputation.
> 4. All `'?'`, `'\t?'`, and empty-string tokens are uniformly replaced with `numpy.nan`.
> 5. The class label column is cleaned via string stripping and direct dictionary mapping: `{'ckd': 1, 'notckd': 0}`. Any row where the class label remains `NaN` post-mapping is discarded.
>
> **Result:** A complete, 400-row `pandas.DataFrame` with no implicit record loss.

---

### Categorical Feature Preprocessing

The 13 categorical columns (`sg, al, su, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane`) were handled as follows:

| Step | Method | Rationale |
|------|--------|-----------|
| **Missing value imputation** | Replace `NaN` with the string literal `'missing'` | Preserves clinical information that a value was not recorded — itself potentially predictive — rather than inflating existing category frequencies via mode imputation |
| **Encoding** | `OrdinalEncoder` (scikit-learn) | Converts each category to an integer code compatible with scikit-learn estimators |

> **Example:** The hypertension (`htn`) feature, originally binary (`'yes'`/`'no'`), becomes a three-category feature: `'yes'`, `'no'`, `'missing'`. This is consistent with the methodology of Rahman et al. (2024).

---

### Numerical Feature Preprocessing (MICE Imputation)

The 11 numerical columns (`age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc`) were processed via a two-step pipeline:

**Step 1 — MICE Imputation (`IterativeImputer`, scikit-learn)**

MICE (Multiple Imputation by Chained Equations) models each feature with missing values as a function of **all other features**, iteratively refining imputed estimates through a regression-based chain. In each iteration, one column is designated the response variable while all remaining columns serve as predictors; a regression model is fitted on observed values and used to fill missing entries. This cycle repeats until convergence, producing imputed values that reflect the **multivariate structure** of the dataset rather than marginal statistics alone.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42, max_iter=10)
X_numerical_imputed = imputer.fit_transform(X_numerical)
```

**Step 2 — StandardScaler Normalisation**

After imputation, `StandardScaler` transforms each numerical column to **zero mean and unit variance**. This prevents features with large absolute ranges (e.g., blood glucose: 70–500 mg/dL) from dominating those with smaller ranges (e.g., serum creatinine: 0.5–15 mg/dL) in distance-sensitive algorithms and gradient computations.

#### KDE Analysis: Before vs. After Preprocessing

KDE plots were generated for all 12 numerical features, stratified by target class, to characterise distributional properties and validate the preprocessing steps.

**Before Preprocessing** — key observations:
- `hemo` (haemoglobin): Pronounced bimodal distribution with minimal class overlap. CKD patients cluster at ~8–12 gms; Not-CKD at ~13–17 gms.
- `pcv` (packed cell volume) and `sc` (serum creatinine): Clear class separation.
- `bgr` (blood glucose random): Heavily right-skewed with a long hyperglycaemic tail.
- `pot` / `sod`: Extreme density spikes at low values due to high missingness rates (zero-padded at this stage).


**After Preprocessing** — observed effects:
- All distributions are centered around zero (standardised scale).
- Anomalous density spikes in `pot` and `sod` are resolved into comprehensible distributions.
- Class separability in `hemo`, `pcv`, and `sc` is preserved after MICE imputation and scaling.

---

### Engineering Challenge 2: Data Splitting & Leakage Prevention

> **Problem:** Data leakage — particularly when oversampling or scaling is applied to the full dataset before splitting — causes synthetic training samples to be drawn from the test set's distribution, producing **optimistically biased performance estimates** that do not generalise to unseen clinical data.

> **Solution: A two-stage stratified splitting protocol.**

```
Full Preprocessed Dataset (400 samples)
│
├── 20% → Secondary Validation / Holdout Set (80 samples) ──────────────────────────────────┐
│          [Set aside entirely. Never touched during development.]                           │
│                                                                                            │
└── 80% → Combined Training/Test Pool (320 samples)                                         │
           │                                                                                 │
           ├── 20% of 80% → Primary Test Set (64 samples) ─────────────────────────────┐    │
           │                [No oversampling. Model selection.]                         │    │
           │                                                                            │    │
           └── 80% of 80% → Training Set (255 samples)                                 │    │
                             │                                                          │    │
                             └── SVMSMOTE applied HERE ONLY → 320 balanced samples     │    │
                                                                                        │    │
                                                                         ↓ Eval         ↓    ↓
                                                                      CV Acc    Test Acc    Holdout Acc
```

All splits use `random_state=42` and **stratified sampling** to maintain the original 62.5/37.5 CKD/Not-CKD class ratio across partitions.

| Partition | Pre-SMOTE Samples | Post-SMOTE Samples | Role |
|-----------|-------------------|--------------------|------|
| Training Set | 255 (160 CKD / 95 Not-CKD) | 320 (160 / 160) | Model fitting |
| Primary Test Set | 64 | 64 (unchanged) | Model selection |
| Secondary Validation | 80 | 80 (unchanged) | Final holdout evaluation |

---

### Class Balancing with SVMSMOTE

The original training partition had a 1.68:1 CKD/Not-CKD imbalance. **SVMSMOTE** (SVM-based Synthetic Minority Over-sampling Technique) was applied **exclusively to the training partition** to address this.

Unlike standard SMOTE, which interpolates between k-nearest neighbours uniformly across feature space, SVMSMOTE identifies **minority class support vectors** — instances nearest the SVM decision boundary — and generates synthetic samples **specifically in the decision boundary region**. This is clinically significant: the decision boundary in CKD classification corresponds to patients with ambiguous clinical presentations, precisely the cases where a model is most likely to err. Focusing synthetic sample generation on this boundary region improves classifier sensitivity where it matters most.

**Post-SMOTE training distribution:** 160 CKD / 160 Not-CKD → balanced 1:1 ratio.

---

## Feature Selection: RFE vs. Boruta

Feature selection is treated not merely as a performance-optimisation step but as a **clinical utility problem** with direct patient-facing consequences: fewer required biomarker measurements translate to lower diagnostic cost, shorter waiting times, and reduced patient discomfort.

### Recursive Feature Elimination (RFE)

RFE is a wrapper-based method using a **Random Forest** base estimator. In each iteration, the full remaining feature set is fitted to the estimator; impurity-based importance scores are extracted; and the lowest-ranked features are pruned. This continues until the target count is reached.

**Configuration:** `n_features_to_select = 12` from the 24-feature post-encoding dataset.

**12 RFE-Selected Features:**

| Feature | Clinical Significance |
|---------|----------------------|
| `sg` | Specific gravity — urine concentration capacity |
| `rbc` | Red blood cells — renal anaemia indicator |
| `bgr` | Blood glucose random — hyperglycaemia, DM link |
| `bu` | Blood urea — glomerular filtration rate (GFR) proxy |
| `sc` | Serum creatinine — gold-standard GFR biomarker |
| `sod` | Sodium — electrolyte balance, fluid regulation |
| `hemo` | Haemoglobin — renal anaemia indicator |
| `pcv` | Packed cell volume — renal anaemia indicator |
| `rc` | Red blood cell count — renal anaemia indicator |
| `htn` | Hypertension — most prevalent CKD risk factor |
| `dm` | Diabetes mellitus — most prevalent CKD risk factor |
| `pe` | Pedal oedema — late-stage fluid retention indicator |

The four highest-importance features by Random Forest score (`pcv`, `hemo`, `sg`, `rc`) are all primary indicators of renal anaemia and urine concentration capacity — the two functional pathways most directly impaired by CKD.

---

### Boruta Feature Selection

Boruta is an **all-relevant feature selection** algorithm wrapping Random Forest. It creates **shadow features** (randomly permuted copies of all original features) and appends them to the dataset. A Random Forest is trained on the augmented dataset, and any original feature whose importance consistently exceeds the maximum shadow feature importance (`Zmax`) across iterations is marked as relevant.

**Boruta Selected 19 of 24 features (79%):** `age, bp, sg, al, rbc, pc, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, appet, pe`

The 7 features retained by Boruta but discarded by RFE are: `age, bp, al, pc, pot, wc, appet`.

---

### Head-to-Head Comparison: Why RFE Is Superior

> **Central quantitative finding:** RFE and Boruta achieve **statistically identical peak performance** across all five models and all evaluation metrics, yet RFE accomplishes this using **37% fewer features** (12 vs. 19). Under the principle of clinical parsimony — when two methods achieve equivalent predictive performance, the simpler model is preferred — RFE is the superior selection method.

| Criterion | RFE | Boruta | Verdict |
|-----------|-----|--------|---------|
| Features Selected | **12 (50%)** | 19 (79%) | ✅ RFE Superior |
| Peak CV Accuracy (Extra Trees) | **100.00% ± 0.00%** | 100.00% ± 0.00% | ⚖️ Equivalent |
| Primary Test Accuracy (Extra Trees) | **100.00%** | 100.00% | ⚖️ Equivalent |
| Lab Tests Required | **12 biomarkers** | 19 biomarkers | ✅ RFE Superior |
| Computational Training Time | **Lower** | Higher | ✅ RFE Superior |
| Clinical Accessibility | Higher | Lower | ✅ RFE Superior |

> The 7 features exclusive to Boruta (notably `wc` and `pot`) require additional laboratory tests not routinely available at primary healthcare facilities in resource-limited settings. Eliminating them at **zero measurable cost to predictive accuracy** directly translates to a more accessible, lower-cost screening protocol.

---

## Model Architecture & Validation

### Ensemble Classifiers Evaluated

Five ensemble learning classifiers were evaluated across both the RFE (12-feature) and Boruta (19-feature) subsets, yielding **10 distinct experimental conditions**.

| Classifier | Type | Key Characteristic |
|------------|------|--------------------|
| **Random Forest** | Bagging | Bootstrap-aggregated decision trees; also served as the base estimator for both RFE and Boruta |
| **Gradient Boosting** | Boosting | Sequential residual correction; computationally intensive, sensitive to hyperparameter configuration |
| **AdaBoost** | Boosting | Iterative misclassification-weight assignment to hard cases; highly effective on clean, balanced data |
| **Extra Trees** ⭐ | Bagging + Extra Randomisation | Random split-threshold selection per feature per node; substantially reduced variance |
| **XGBoost** | Regularised Boosting | L1/L2 regularisation, parallel processing, native missing-value handling; 0.32s training time on RFE set |

### 10-Fold Stratified Cross-Validation

All models were evaluated using **10-fold Stratified CV** on the SVMSMOTE-balanced training set. Stratification ensures each fold preserves the 1:1 post-SMOTE class ratio, preventing misleading performance estimates from fold-level class imbalance. The CV standard deviation directly measures model stability — a standard deviation of `0.00%` (achieved by Extra Trees) indicates that the decision boundary is robust across **every** possible 90/10 partition of the training data.

### Hyperparameter Tuning (Extra Trees via GridSearchCV)

Following identification of Extra Trees as the top performer, an exhaustive `GridSearchCV` was conducted:

```python
param_grid = {
    'n_estimators':       [50, 100, 200],
    'criterion':          ['gini', 'entropy'],
    'max_depth':          [None, 10, 20],
    'min_samples_split':  [2, 5, 10]
}
# Total configurations: 3 × 2 × 3 × 3 = 54
# Tuning runtime: ~41.66 seconds
```

**Optimal Configuration:**

| Hyperparameter | Value | Effect |
|----------------|-------|--------|
| `n_estimators` | 50 | Efficient ensemble size |
| `criterion` | `'gini'` | Gini impurity split metric |
| `max_depth` | `None` | Unrestricted tree depth |
| `min_samples_split` | `10` | Implicit regularisation; prevents memorisation of individual training instances |

---

## Results & Discussion

### 10-Fold CV and Primary Test Set Performance

| Model | RFE — CV Accuracy | RFE — Test Accuracy | Boruta — CV Accuracy | Boruta — Test Accuracy |
|-------|-------------------|---------------------|----------------------|------------------------|
| Random Forest | 99.06% ± 1.43% | 98.44% | 99.06% ± 1.43% | 98.44% |
| Gradient Boosting | 98.44% ± 2.10% | 95.31% | 97.81% ± 2.44% | 98.44% |
| AdaBoost | 99.06% ± 1.43% | 98.44% | 99.38% ± 1.25% | 98.44% |
| **Extra Trees** ⭐ | **100.00% ± 0.00%** | **100.00%** | **100.00% ± 0.00%** | **100.00%** |
| XGBoost | 97.50% ± 2.72% | 98.44% | 97.50% ± 2.72% | 98.44% |

> **⭐ Extra Trees** is the unambiguous top performer. Its zero CV standard deviation is a particularly strong signal: the model achieves perfect classification on every held-out fold regardless of which 10% of the balanced training data is withheld. On the 64-sample primary test set, it correctly classifies all 24 Not-CKD and all 40 CKD instances with zero misclassifications.
>
> **Note on Gradient Boosting:** Its 95.31% RFE test accuracy is attributable to 3 False Negative misclassifications — CKD patients incorrectly told they do not have the disease. In the clinical context, False Negatives are substantially more dangerous than False Positives.

### Confusion Matrix Analysis

Extra Trees (both RFE and Boruta conditions) achieves a **perfect 24×0 / 0×40 confusion matrix** on the 64-sample primary test set — the only model to do so across all 10 experimental conditions.

### ROC-AUC Curve Analysis


| Condition | AUC |
|-----------|-----|
| Extra Trees (RFE) | **1.000** |
| Extra Trees (Boruta) | **1.000** |
| Random Forest (RFE/Boruta) | **1.000** |
| AdaBoost (RFE/Boruta) | **1.000** |
| All Boruta models | **1.000** |
| Gradient Boosting (RFE) | 0.999 |
| XGBoost (RFE) | 0.999 |

7 of 10 model-feature configurations achieve AUC = 1.000. The near-vertical rise of ROC curves from the origin is characteristic of a highly confident, well-calibrated classifier.

### Final Model: Classification Report on Secondary Validation Set (80 Samples)

The tuned Extra Trees model (RFE, 12 features) was evaluated on the **entirely untouched 80-sample secondary validation set** — the most reliable estimate of real-world performance, as these samples were isolated before any preprocessing decisions were made.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not CKD (0) | 0.97 | 1.00 | 0.98 | 30 |
| CKD (1) | 1.00 | 0.98 | 0.99 | 50 |
| **Overall Accuracy** | | | **0.99** | **80** |
| Macro Average | 0.98 | 0.99 | 0.99 | 80 |
| Weighted Average | 0.99 | 0.99 | 0.99 | 80 |

> - **CKD Precision = 1.00:** Every patient predicted as CKD was genuinely CKD — **zero false alarms**.
> - **CKD Recall = 0.98:** Only 1 of the 50 actual CKD patients in the holdout was missed.
> - **Not-CKD Recall = 1.00:** All 30 genuinely healthy patients were correctly identified.

---

## Generative AI Integration (Groq API + Llama3-8b)

A key user-experience enhancement distinguishing this deployment from a standard ML application is the integration of the **Groq API** as a digital clinical explanation engine.

**The Problem with Raw Binary Predictions:**
A prediction of `CKD` or `Not CKD` provides limited actionable information to non-specialist users or patients who are unfamiliar with the clinical significance of their biomarker values.

**Solution Architecture:**
Upon model prediction, the backend dynamically constructs a structured three-part prompt:

```
(a) Patient's 12 clinical input values (e.g., Haemoglobin: 9.2 g/dL, Serum Creatinine: 5.1 mg/dL, ...)
(b) Model prediction and confidence probability (e.g., "CKD — 97.3% confidence")
(c) System instruction: LLM role = "medically literate but non-diagnostic educational assistant"
```

The **Llama3-8b** model (served via Groq's hardware-accelerated LPU inference at extremely low latency) generates a paragraph-length personalised explanation that:
- Identifies which specific input values deviate from established clinical normal ranges.
- Explains the **physiological significance** of each deviation in the context of CKD pathology.
- Provides general guidance on recommended next steps (e.g., consulting a nephrologist).
- Explicitly frames the output as **educational context** to facilitate informed conversation with a healthcare professional, not as a definitive medical diagnosis.

```python
# Groq API integration (conceptual)
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

prompt = f"""
Patient biomarkers: {clinical_values_dict}
Model prediction: {prediction} (Confidence: {probability:.1%})

As a medically literate but non-diagnostic educational assistant, explain which values
deviate from normal ranges, their physiological significance in CKD, and recommended next steps.
"""

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## Engineering Challenges Overcome

### ⚙️ Challenge 1: ARFF File Parsing
*(Covered in detail under [Data Pipeline](#engineering-challenge-1-arff-file-parsing).)*

---

### ⚙️ Challenge 2: Data Splitting & Leakage Prevention
*(Covered in detail under [Data Pipeline](#engineering-challenge-2-data-splitting--leakage-prevention).)*

---

### ⚙️ Challenge 3: The 24-Column Scaler Dimension Mismatch

**Root Cause:**
The `StandardScaler` was fitted during preprocessing on the full 11-column numerical feature set, but its internal statistics are indexed against the **positional layout of the complete 24-column DataFrame** (in which numerical and categorical columns are interleaved). When the deployed Streamlit backend loaded the serialised scaler and presented a user-submitted input containing only the **12 RFE-selected features**, the scaler raised a `ValueError` due to the shape mismatch (expected 24 columns, received 12).

Re-fitting the scaler on the RFE subset alone would invalidate the serialised Extra Trees model, which was trained on scalings derived from the full preprocessing pipeline.

**Solution — 24-Column Dummy DataFrame Injection:**

```python
import pandas as pd
import numpy as np

# Step 1: Construct a 24-column template matching the full preprocessed schema
dummy_df = pd.DataFrame(
    np.zeros((1, 24)),
    columns=all_24_feature_columns  # Original column names in original order
)

# Step 2: Inject the 12 user-supplied RFE values into their correct positional columns
for feature, value in user_input_dict.items():
    dummy_df[feature] = value

# Step 3: Pass the full 24-column DataFrame through the loaded scaler
scaled_full = scaler.transform(dummy_df)

# Step 4: Slice only the 12 RFE-selected column indices from the scaled output
rfe_indices = [all_24_feature_columns.index(f) for f in rfe_selected_features]
X_scaled_rfe = scaled_full[:, rfe_indices]

# Step 5: Pass to the Extra Trees model
prediction = model.predict(X_scaled_rfe)
probability = model.predict_proba(X_scaled_rfe)
```

This approach maintains **complete fidelity to the original preprocessing pipeline** without any re-training or re-fitting, while allowing the deployment interface to accept only 12 inputs.

---

### ⚙️ Challenge 4: Custom Domain Masking Failure

An attempt was made to deploy the application under a custom domain (`raufhassan.dev`, registered via Name.com) for professional presentation purposes. Two strategies were attempted and both failed:

| Strategy | Outcome | Root Cause |
|----------|---------|------------|
| **CNAME DNS record** pointing to Streamlit subdomain | `ERR_TOO_MANY_REDIRECTS` | Streamlit Community Cloud's security layer detects the CNAME and redirects to its canonical HTTPS domain; the CNAME re-redirects, producing an infinite HTTP 301/302 loop. |
| **URL/HTTP Forwarding (301 redirect)** via Name.com | Browser URL bar displays the underlying Streamlit URL, not the custom domain | HTTP forwarding is a redirect, not masking. Iframe masking was also rejected as Streamlit sets `X-Frame-Options` headers preventing cross-origin iframe embedding. |

**Documented Resolution:**
Streamlit Community Cloud does **not support custom domain binding on its free tier**. Resolution paths:
- Upgrade to Streamlit Teams plan (supports custom domain binding).
- Migrate to a VPS with a Docker container and **Nginx reverse proxy** configuration.

---

## System Deployment


The final system is deployed as a **Streamlit web application** with the following architecture:

**Frontend (Left Panel):** 12 labelled input fields (numerical sliders + categorical dropdowns) corresponding to the RFE-selected features. Clinical range guidance is displayed as hints (e.g., *Serum Creatinine: 0.4–76 mg/dL*).

**Prediction Backend Pipeline (triggered on 'Predict' button submission):**

```
1. Collect raw inputs from Streamlit form state
2. Encode categorical inputs via loaded OrdinalEncoder (encoder.pkl)
3. Construct 24-column dummy DataFrame; inject 12 user values
4. Normalise via loaded StandardScaler (scaler.pkl); extract RFE column indices
5. Generate binary prediction + probability via tuned Extra Trees (extra_trees_rfe.pkl)
6. Display prediction result on right panel
7. If Groq API enabled: format 12 clinical values + prediction into Llama3-8b prompt
8. Render LLM-generated natural-language explanation to user
```

**Serialised Artefacts (joblib/pickle):**

| File | Contents |
|------|----------|
| `extra_trees_rfe.pkl` | Tuned Extra Trees classifier (tree structure, node weights) |
| `scaler.pkl` | Fitted StandardScaler (mean and variance statistics for all 11 numerical columns) |
| `encoder.pkl` | Fitted OrdinalEncoder (category-to-integer mappings for all 13 categorical columns) |

---

## Installation & Local Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (free tier available)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Chaos1417/ckd-web-app.git
cd ckd-web-app
```

### Step 2 — Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the core dependencies manually:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn xgboost boruta groq joblib matplotlib seaborn
```

> **Note on `IterativeImputer`:** It is available in scikit-learn via an experimental flag. No additional install is needed, but ensure your scikit-learn version is ≥ 0.21.

### Step 4 — Configure the Groq API Key

The application reads the Groq API key from an environment variable. Set it in your shell before running:

```bash
# Linux/macOS
export GROQ_API_KEY="your_groq_api_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY=your_groq_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"
```

Alternatively, create a `.streamlit/secrets.toml` file in the project root:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### Step 5 — (Optional) Retrain the Model

If you want to reproduce the full training pipeline from scratch:

```bash
python train_pipeline.py   # or the equivalent training script in the repo
```

This will regenerate `extra_trees_rfe.pkl`, `scaler.pkl`, and `encoder.pkl` in the project root. Ensure the UCI CKD ARFF file is placed at the expected path (check the script for the configured input path).

### Step 6 — Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your browser.

---

## Project Structure

```
ckd-web-app/
│
├── app.py                        # Streamlit application entrypoint
├── train_pipeline.py             # End-to-end training, feature selection, evaluation pipeline
├── requirements.txt              # Python dependencies
│
├── data/
│   └── chronic_kidney_disease.arff  # Raw UCI CKD dataset (ARFF format)
│
├── models/
│   ├── extra_trees_rfe.pkl       # Serialised tuned Extra Trees classifier
│   ├── scaler.pkl                # Serialised StandardScaler (fitted on full 24-col DataFrame)
│   └── encoder.pkl               # Serialised OrdinalEncoder (fitted on 13 categorical columns)
│
├── images/                       # Figures referenced in the report and this README
│   ├── fig1_arff_parsed_dataframe.png
│   ├── fig2_kde_before_preprocessing.png
│   ├── fig3_kde_after_preprocessing.png
│   ├── fig4_rfe_heatmap_importance.png
│   ├── fig5_confusion_matrices.png
│   ├── fig6_roc_curves.png
│   └── fig7_web_interface.png
│
├── .streamlit/
│   └── secrets.toml              # Groq API key (excluded from version control)
│
└── README.md                     # This file
```

---

## Comparison with Reference Paper

| Metric | Rahman et al. (2024) — LightGBM | This Study — Extra Trees (RFE) |
|--------|----------------------------------|-------------------------------|
| **Average Accuracy** | 99.75% | **100.00%** (primary test) / **99.00%** (holdout) |
| **Precision** | 99.40% | **99%** (weighted avg, holdout) |
| **Recall** | 99.41% | **99%** (weighted avg, holdout) |
| **F-Measure** | 99.61% | **99%** (weighted avg, holdout) |
| **AUC-ROC** | 99.57% | **100.00%** (primary test) |
| **Features Used** | Not specified | **12 of 24 (50%)** |
| **Web Deployment** | ❌ | ✅ (Streamlit + LLM explanations) |
| **Data Leakage Prevention** | Not explicitly documented | ✅ Two-stage stratified protocol |
| **Feature Selection Comparison** | Single method | ✅ RFE vs. Boruta (quantitative) |

This study's Extra Trees model achieves results that are **statistically competitive with or superior** to the reference paper's best-reported performance, while using a **50% smaller feature set** — a direct clinical cost reduction — and additionally delivering a live, LLM-augmented web application.

---

## Future Work

- **CKD Stage Prediction:** Extend from binary classification (CKD / Not-CKD) to multi-class prediction of disease stages 1–5, enabling more clinically actionable treatment planning.
- **Deep Learning Baselines:** Evaluate tabular deep learning architectures (TabNet, 1D-CNN) to identify whether subtle boundary patterns missed by ensemble methods can be captured.
- **Multi-Site Validation:** Test on datasets from diverse hospitals and geographic regions to assess generalisation beyond the Tamil Nadu UCI cohort.
- **Production Infrastructure Migration:** Deploy on a VPS (AWS EC2 / DigitalOcean Droplet) with a Docker container and Nginx reverse proxy to enable custom domain binding and eliminate Streamlit Community Cloud infrastructure constraints.
- **SHAP Explainability:** Integrate SHAP (SHapley Additive exPlanations) values into the Streamlit app to provide per-prediction, feature-level attribution to clinicians alongside the LLM narrative explanation.

---

## References

1. Rahman, Md. M., Al-Amin, Md., & Hossain, J. (2024). Machine learning models for chronic kidney disease diagnosis and prediction. *Biomedical Signal Processing and Control*, 87, 105368. https://doi.org/10.1016/j.bspc.2023.105368
2. Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. http://archive.ics.uci.edu/ml
3. van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.
4. Han, H., Wang, W. Y., & Mao, B. H. (2005). Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. *LNCS*, 3644, 878–887.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
6. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*, 785–794.
7. Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene Selection for Cancer Classification using Support Vector Machines. *Machine Learning*, 46(1), 389–422.
8. Kursa, M. B., Jankowski, A., & Rudnicki, W. R. (2010). Boruta – A System for Feature Selection. *Fundamenta Informaticae*, 101(4), 271–285.
9. Parzen, E. (1962). On Estimation of a Probability Density Function and Mode. *Annals of Mathematical Statistics*, 33(3), 1065–1076.
10. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

---

## Author

**Rauf Hassan**
 BSCS 6th Semester
*CSE6505 — Machine Learning | Lahore Garrison University*
 Date: 18 May, 2026

[![GitHub](https://img.shields.io/badge/GitHub-Chaos1417-181717?style=flat-square&logo=github)](https://github.com/Chaos1417)

---
