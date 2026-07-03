# Credit Scoring — Modelo de Probabilidad de Default (PD)

**🇪🇸 Español** · [🇬🇧 English](README.en.md)

Pipeline end-to-end de credit scoring siguiendo prácticas de Risk Analytics
y Model Risk Management en banca (SR 11-7, TRIM-ECB, Basilea III).

**▶️ [Probar la app en vivo](https://credit-scoring-end-to-end.streamlit.app/)** — cargá un solicitante y obtené decisión, score y motivo explicado.

## Resultados

| Modelo | AUC | Gini (modelo completo) | Gini (originación pura)¹ | KS |
|---|---|---|---|---|
| Regresión Logística (baseline) | 0.8610 | 72.20% | 58.73% | 0.5734 |
| XGBoost (challenger) | 0.9499 | 89.98% | **80.70%** | 0.7520 |

¹ **Originación pura** = modelo reentrenado sin las variables de pricing
(`loan_grade`, `loan_int_rate`), que son output de un proceso de evaluación de
riesgo previo y no atributos crudos del solicitante. El modelo completo se apoya
parcialmente en ese juicio heredado; la columna de originación pura aísla el
poder discriminante que el modelo construye a partir de las características
propias del solicitante. Análisis de sensibilidad completo en la sección 10.4.2
del notebook.

> Removiendo todas las variables de pricing, el XGBoost sostiene un **Gini de
> 80,70%** ("excelente" en las escalas regulatorias habituales) frente al 58,73%
> de la logística: el desempeño proviene de modelar relaciones no lineales
> reales del solicitante, no del grado y la tasa heredados.

**Champion recomendado:** Regresión Logística (banco regulado) · XGBoost (fintech)
**Índice de Estabilidad Poblacional (PSI):** 0.00120 — modelo estable

## Pipeline

| Capítulo | Contenido |
|---|---|
| 2–3 | Gobierno de datos (3 tiers) + imputación |
| 4 | EDA orientado a poder discriminante |
| 5 | Feature Engineering (WoE/IV, variables derivadas) |
| 6 | Modelado: Logística + XGBoost + diagnóstico VIF |
| 7 | Métricas regulatorias (AUC, Gini, KS) |
| 8 | Scorecard calibrada + política de corte |
| 9 | Champion vs. Challenger + SHAP |
| 10 | Model Governance + PSI + stress tests (incl. sensibilidad a variables de pricing, 10.4.2) |

## Decisiones metodológicas

- **Variables de pricing tratadas de forma explícita.** `loan_grade` y
  `loan_int_rate` codifican un juicio de riesgo previo del originador. Se
  conservan en el modelo completo, pero su aporte se cuantifica por separado
  (sección 10.4.2), de modo que el desempeño reportado puede leerse según dos
  casos de uso distintos: originación independiente (excluirlas) vs.
  pricing/provisión que asume el grado como dado (conservarlas).
- **Multicolinealidad y endogeneidad tratadas como problemas distintos.** El
  modelo logístico descarta `loan_int_rate` por multicolinealidad con el grado
  (análisis VIF, sección 6.3); el test de variables de pricing es un ejercicio
  conceptual aparte, sobre qué información se le permite usar al modelo.
- **Limitación conocida — imputación ajustada sobre el dataset completo.** La
  imputación por mediana se calcula antes de la partición train/test. El impacto
  numérico es menor (estadístico robusto sobre valores faltantes), pero un
  pipeline estrictamente libre de fuga ajustaría la imputación solo sobre el
  conjunto de entrenamiento. Documentado, no ocultado.

## Reason codes con IA generativa

Una capa adicional convierte cada decisión en un motivo en lenguaje natural para
el solicitante, en la línea de los *adverse action notices* que exige la
regulación (ECOA en EE.UU.; derecho a explicación bajo GDPR/EBA en la UE). El
principio de diseño es la **separación estricta de responsabilidades**:

- **El modelo decide.** La PD calibrada se convierte en score y se compara contra
  el corte de la scorecard. La IA generativa no interviene en la decisión.
- **Las contribuciones logísticas cuantifican.** Para cada solicitante, la
  contribución de cada variable al log-odds (`coef × valor estandarizado`)
  identifica los factores que pesaron —exacta por tratarse del modelo logístico—.
  Los factores se seleccionan según el sentido de la decisión (los que empujan al
  default en un rechazo; los que la sostienen en una aprobación).
- **El LLM solo redacta.** Recibe la decisión y los factores ya calculados y
  produce el texto; no accede a los datos crudos ni decide nada. Un **guardrail
  determinístico** filtra la salida para garantizar el cumplimiento de las reglas.

Modelo `llama-3.3-70b-versatile` vía API de Groq, `temperature=0`. La aplicación
web ([Streamlit](https://credit-scoring-end-to-end.streamlit.app/)) permite cargar
un solicitante desde un formulario o evaluar un caso de ejemplo.

## Tecnologías

Python · scikit-learn · XGBoost · SHAP · statsmodels · pandas · seaborn · Streamlit · Groq (LLM)

## Dataset

[Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
Dataset académico utilizado para demostración metodológica.
