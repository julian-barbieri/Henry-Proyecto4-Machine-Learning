# Prediccion de Churn Bancario — FinanceGuard

**Autor:** Julian Barbieri
**Proyecto:** Machine Learning — Modulo 4
**Dataset:** Churn_Modelling.csv (10,000 clientes)

---

## Descripcion del Problema

FinanceGuard, un banco digital, enfrenta una tasa de abandono (churn) del **20% anual**, lo que representa perdidas millonarias en su base de 10,000 clientes activos. El objetivo del proyecto es construir un sistema de prediccion temprana que permita reducir el churn al 15% mediante modelos de machine learning supervisados y no supervisados, combinados con estrategias personalizadas de retencion.

---

## Estructura del Proyecto

```
ProyectoM4_JulianBarbieri/
├── 1_EDA_RegresionLogistica.ipynb          # Exploracion y modelo baseline
├── 2_GradientBoosting_Optimizacion.ipynb   # Modelos avanzados y ensamble
├── 3_AprendizajeNoSupervisado.ipynb        # Clustering de clientes
├── Churn_Modelling.csv                     # Dataset principal
└── Reporte de resultados.pdf               # Reporte ejecutivo del proyecto
```

---

## Dataset

| Caracteristica       | Detalle                              |
| -------------------- | ------------------------------------ |
| Registros            | 10,000 clientes                      |
| Variables            | 14 (11 features + 3 identificadores) |
| Variable objetivo    | `Exited` (0 = activo, 1 = abandono)  |
| Desbalance de clases | 79.63% no-churn / 20.37% churn       |
| Valores nulos        | Ninguno                              |
| Duplicados           | Ninguno                              |

**Variables principales:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

---

## Notebook 1 — EDA y Regresion Logistica

### Analisis Exploratorio

Hallazgos clave del analisis bivariado con la variable objetivo:

| Variable       | Insight                                       |
| -------------- | --------------------------------------------- |
| Geography      | Alemania: 40% churn; Francia: 14%; España: 8% |
| Gender         | Mujeres presentan mayor tasa de churn         |
| IsActiveMember | Inactivos: 45% churn vs Activos: 7% churn     |
| Age            | Correlacion positiva con churn (r = 0.29)     |
| HasCrCard      | Sin diferencia significativa                  |

### Preprocesamiento

- **OneHotEncoder** para variables categoricas (`Geography`, `Gender`), con `drop="first"`
- **StandardScaler** para variables numericas
- **Split estratificado** 80/20 preservando la proporcion de clases

### Modelo Baseline — Regresion Logistica

| Metrica  | Valor  |
| -------- | ------ |
| ROC-AUC  | 0.7511 |
| PR-AUC   | 0.3882 |
| Recall   | 0.67   |
| F1-Score | 0.49   |
| Accuracy | 0.71   |

**Factores de mayor impacto (Odds Ratios):**

- `Age` (+2.25x): El mas determinante
- `Geography_Germany` (+2.05x): Alta incidencia en Alemania
- `Gender_Male` (−0.41x): Hombres menos propensos al churn

---

## Notebook 2 — Gradient Boosting y Ensamble

### Feature Engineering

Se creo la variable `AgeGroup` (Joven: 0-30, Adulto: 30-50, Mayor: 50+).

### Comparacion de Modelos Base (Validacion Cruzada 5-fold)

| Modelo        | ROC-AUC             | F1-Score   | Tiempo |
| ------------- | ------------------- | ---------- | ------ |
| CatBoost      | **0.8748 ± 0.0049** | **0.6130** | 6.77s  |
| XGBoost       | 0.8639 ± 0.0045     | 0.6032     | 2.35s  |
| LightGBM      | 0.8533 ± 0.0115     | 0.5983     | 6.07s  |
| Random Forest | 0.8555 ± 0.0053     | 0.6059     | 21.57s |

### Optimizacion con Optuna (XGBoost)

Mejores hiperparametros encontrados (50 trials):

```
n_estimators:      341
learning_rate:     0.011
max_depth:         6
subsample:         0.819
colsample_bytree:  0.648
```

### Ensamble Final — Stacking

**Arquitectura:**

- **Nivel 1 (modelos base):** XGBoost_Optuna, LightGBM, CatBoost
- **Nivel 2 (meta-learner):** Regresion Logistica
- **Validacion:** Cross-validation de 5 folds

### Comparacion Final en Test

| Modelo         | ROC-AUC    | PR-AUC     | Recall     | F1-Score   |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| **Stacking**   | **0.8805** | **0.7290** | **0.7676** | **0.6246** |
| CatBoost       | 0.8772     | 0.7261     | 0.5387     | 0.6375     |
| XGBoost_Optuna | 0.8769     | 0.7298     | 0.4824     | 0.6130     |
| LightGBM       | 0.8765     | 0.7144     | 0.7570     | 0.6152     |
| XGBoost_Base   | 0.8705     | 0.7042     | 0.5211     | 0.6103     |
| Random Forest  | 0.8546     | 0.7012     | 0.7183     | 0.5887     |
| Reg. Logistica | 0.7511     | 0.3882     | 0.6700     | 0.4900     |

**Ganador: Stacking** — Mejor ROC-AUC, PR-AUC y Recall.

### Top 5 Variables mas Importantes (Stacking)

1. `NumOfProducts` (0.3400) — Factor 3x mas importante que el siguiente
2. `AgeGroup_Mayor` (0.1292)
3. `IsActiveMember` (0.1058)
4. `Age` (0.1025)
5. `Geography_Germany` (0.0817)

---

## Notebook 3 — Aprendizaje No Supervisado

### Feature Engineering Adicional

| Variable creada     | Descripcion                  |
| ------------------- | ---------------------------- |
| `AgeGroup`          | Segmentos por edad           |
| `BalanceGroup`      | Bajo / Medio / Alto          |
| `ProductsPerTenure` | NumOfProducts / (Tenure + 1) |
| `ActiveWithCard`    | IsActiveMember \* HasCrCard  |

### PCA

- 2 componentes explican el **37.1%** de la varianza
- Se requieren 9 componentes para alcanzar el **90%**

### K-Means — Segmentacion en 3 Clusters

K optimo seleccionado por analisis de Silhouette Score, Calinski-Harabasz y metodo del codo.

| Cluster | Nombre                   | Clientes      | Churn Rate | Caracteristicas Clave                                                         |
| ------- | ------------------------ | ------------- | ---------- | ----------------------------------------------------------------------------- |
| 0       | **Clientes Fieles**      | 2,643 (33%)   | 15.65%     | Balance bajo (~1,600), mas productos (1.76), dominan Francia y España         |
| 1       | **Alto Valor en Riesgo** | 4,357 (54%)   | 23.04%     | Balance alto (~120,000), pocos productos (1.29), fuerte presencia en Alemania |
| 2       | **Nuevos e Inestables**  | 1,000 (12.5%) | 20.21%     | Tenure muy bajo (1.1 años), balance medio, en proceso de adopcion             |

### DBSCAN (Comparativa)

| Metrica           | K-Means  | DBSCAN |
| ----------------- | -------- | ------ |
| Silhouette        | 0.199    | 0.370  |
| Calinski-Harabasz | 1560     | 401    |
| Davies-Bouldin    | 1.764    | 0.928  |
| Interpretabilidad | **Alta** | Baja   |

**Decision: K-Means** — Aunque DBSCAN mejora algunas metricas, produce casi un unico mega-cluster con escasa utilidad practica. K-Means ofrece 3 segmentos interpretables y accionables para el negocio.

---

## Resultados y Recomendaciones Estrategicas

### Por Cluster

| Cluster       | Estrategia Recomendada                                                        |
| ------------- | ----------------------------------------------------------------------------- |
| Fieles (0)    | Programas de fidelizacion, cross-sell moderado, beneficios por saldo          |
| Riesgosos (1) | **Maxima prioridad**: asesor dedicado, plan VIP, bonificacion por permanencia |
| Nuevos (2)    | Onboarding intensivo, incentivos primer mes, seguimiento a 3-6 meses          |

### Guia de Seleccion de Modelo

| Escenario                       | Modelo Recomendado  |
| ------------------------------- | ------------------- |
| Maxima performance predictiva   | Stacking            |
| Balance rendimiento / velocidad | XGBoost + Optuna    |
| Interpretabilidad para negocio  | Regresion Logistica |
| Segmentacion y estrategias CRM  | K-Means (K=3)       |

---

## Tecnologias Utilizadas

- **Python 3.x**
- `pandas`, `numpy` — Manipulacion de datos
- `matplotlib`, `seaborn` — Visualizacion
- `scikit-learn` — Preprocesamiento, modelos, metricas, ensamble
- `xgboost`, `lightgbm`, `catboost` — Gradient Boosting
- `optuna` — Optimizacion de hiperparametros
- `umap` / `sklearn.manifold` (t-SNE) — Reduccion de dimensionalidad

---

## Impacto Esperado

Mediante la combinacion de modelos predictivos de alta performance (ROC-AUC = 0.8805, Recall = 76.8%) con segmentacion de clientes, el sistema permite identificar de forma temprana a los clientes con mayor probabilidad de abandono y aplicar estrategias diferenciadas, con el objetivo de **reducir la tasa de churn del 20% al 15%** y maximizar la retencion de clientes de alto valor.
