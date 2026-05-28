# Credit Scoring — Modelo de Probabilidad de Default (PD)

Pipeline end-to-end de credit scoring siguiendo prácticas profesionales 
de Risk Analytics y Model Risk Management en banca.

## Resultados

| Modelo | AUC | Gini | KS |
|---|---|---|---|
| Regresión Logística (baseline) | 0.8610 | 72.20% ✦ Excelente | 0.5734 |
| XGBoost (challenger) | 0.9499 | 89.98% ✦ Excelente | 0.7520 |

**Champion recomendado:** Logística (banco regulado) · XGBoost (fintech)  
**PSI:** 0.00120 — modelo estable ✦ Verde

## Pipeline

| Capítulo | Contenido |
|---|---|
| 2–3 | Gobierno de datos (3 tiers) + Imputación |
| 4 | EDA orientado a poder discriminante |
| 5 | Feature Engineering (WoE/IV, variables derivadas) |
| 6 | Modelado: Logística + XGBoost + diagnóstico VIF |
| 7 | Métricas regulatorias (AUC, Gini, KS) |
| 8 | Scorecard calibrada + Política de corte |
| 9 | Champion vs. Challenger + SHAP |
| 10 | Model Governance + PSI + Stress test |

## Tecnologías

Python · scikit-learn · XGBoost · SHAP · statsmodels · pandas · seaborn

## Dataset

[Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
Dataset académico utilizado para demostración metodológica.
