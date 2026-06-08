# Credit Scoring — Probability of Default (PD) Model

[🇪🇸 Español](README.md) · **🇬🇧 English**

End-to-end credit scoring pipeline built on Risk Analytics and Model Risk
Management practices (SR 11-7, TRIM-ECB, Basel III).

## Results

| Model | AUC | Gini (full model) | Gini (pure origination)¹ | KS |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8610 | 72.20% | 58.73% | 0.5734 |
| XGBoost (challenger) | 0.9499 | 89.98% | **80.70%** | 0.7520 |

¹ **Pure origination** = model retrained without the pricing variables
(`loan_grade`, `loan_int_rate`), which are outputs of a prior risk-assessment
process rather than raw applicant attributes. The full model leans partly on
that inherited judgment; the pure-origination column isolates the discriminatory
power the model builds from the applicant's own characteristics. Full
sensitivity analysis in notebook section 10.4.2.

> Stripped of all pricing variables, XGBoost still holds an **80.70% Gini**
> ("excellent" on standard regulatory scales) versus 58.73% for the logistic
> model — evidence that performance comes from modeling genuine non-linear
> applicant patterns, not from the inherited grade and rate.

**Recommended champion:** Logistic Regression (regulated bank) · XGBoost (fintech)
**Population Stability Index (PSI):** 0.00120 — stable model

## Pipeline

| Chapter | Content |
|---|---|
| 2–3 | Data governance (3 tiers) + imputation |
| 4 | EDA focused on discriminatory power |
| 5 | Feature engineering (WoE/IV, derived features) |
| 6 | Modeling: Logistic + XGBoost + VIF diagnostics |
| 7 | Regulatory metrics (AUC, Gini, KS) |
| 8 | Calibrated scorecard + cut-off policy |
| 9 | Champion vs. Challenger + SHAP |
| 10 | Model governance + PSI + stress tests (incl. pricing-variable sensitivity, 10.4.2) |

## Key methodological decisions

- **Pricing variables handled explicitly.** `loan_grade` and `loan_int_rate`
  encode a prior risk judgment made by the originator. They are kept in the full
  model, but their contribution is quantified separately (section 10.4.2) so the
  reported performance can be read against two distinct use cases: independent
  origination (exclude them) vs. grade-assuming pricing/provisioning (keep them).
- **Multicollinearity and endogeneity kept distinct.** The logistic model drops
  `loan_int_rate` due to multicollinearity with the grade (VIF analysis, section
  6.3); the pricing-variable test is a separate, conceptual exercise about which
  information the model is allowed to use.
- **Known limitation — imputation fitted on the full dataset.** Median
  imputation is computed before the train/test split. The numerical impact is
  minor (a robust statistic over missing values), but a strictly leakage-free
  pipeline would fit imputation on the training set only. Documented rather than
  hidden.

## Stack

Python · scikit-learn · XGBoost · SHAP · statsmodels · pandas · seaborn

## Dataset

[Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
Academic dataset used for methodological demonstration.
