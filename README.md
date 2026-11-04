# NBA Rookies Dashboard: ¿quién llega al Hall of Fame?

Dashboard interactivo hecho con **Dash + Plotly** que analiza las estadísticas de **1.537 rookies de la NBA (drafts 1980-2016)** e incluye un **modelo de Regresión Logística** que estima la probabilidad de que un jugador entre en el Hall of Fame a partir de su temporada de novato.

Proyecto de la asignatura *Visualización de Datos* (Ingeniería en IA, TECNUN – Universidad de Navarra, curso 2025-26).

## Qué incluye

**Pestaña 1 · Análisis visual**
- **Evolución de cada tipo de tiro** por año de draft (3PA, FGA, FTA), con selector. Se ve claramente el auge del triple.
- **¿Qué hace a un Hall of Fame?** Boxplot que compara HOF con el resto de jugadores en 11 estadísticas a elegir.
- **Comparador de jugadores:** gráfico radar (PTS, REB, AST, STL, BLK) para superponer dos rookies.

**Pestaña 2 · Modelo de predicción**
- **Buscador de estrellas:** scatter con ejes configurables en el que los HOF aparecen resaltados en dorado, con el nombre al pasar el ratón.
- **Predictor de Hall of Fame:** introduces minutos, puntos, rebotes y asistencias de un rookie y el modelo devuelve la probabilidad de que llegue al HOF.

## El modelo

| | |
|---|---|
| Algoritmo | Regresión Logística (`class_weight='balanced'`) |
| Features | `MIN`, `PTS`, `REB`, `AST` (medias por partido del año rookie) |
| Datos de entrenamiento | Rookies drafteados hasta **1992** (569 jugadores, 23 HOF) |
| Validación (CV estratificada, 5 folds) | **ROC AUC 0.84** · recall HOF 0.65 · precisión HOF 0.14 (umbral 0.5) |

**Decisiones de diseño**
- **Corte temporal en 1992:** los jugadores más recientes pueden seguir en activo o no haber tenido tiempo de ser elegidos. Incluirlos como "no HOF" metería falsos negativos en el entrenamiento.
- **Desbalance extremo** (~4 % de positivos): se compensa con `class_weight='balanced'`. Por eso el modelo detecta la mayoría de los HOF a costa de muchos falsos positivos.

**Limitaciones:** con solo 23 positivos y 4 variables, el modelo sirve como herramienta exploratoria y no como predictor fiable. Entrar en el Hall of Fame depende de toda la carrera, no solo del año rookie.

## Ejecutar en local

```bash
git clone https://github.com/amarcalainm-oss/nba-rookies-hall-of-fame-dash.git
cd nba-rookies-hall-of-fame-dash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                   # http://127.0.0.1:8050
```

## Desplegar

El repo trae un `render.yaml` para desplegarlo gratis en [Render](https://render.com): **New → Blueprint →** seleccionar este repositorio. En producción se sirve con `gunicorn app:server`.

## Estructura

```
├── app.py                              # aplicación Dash (datos + modelo + layout + callbacks)
├── data/nba_rookies.csv                # dataset
├── notebooks/nba_rookies_dash_colab.ipynb   # notebook original (Google Colab)
├── requirements.txt
└── render.yaml                         # configuración de despliegue
```

## Datos

[NBA Rookies: Performance Statistics and Minutes Played](https://www.kaggle.com/datasets/thedevastator/nba-rookies-performance-statistics-and-minutes-p) (Kaggle).

## Tecnologías

Python · Dash · Dash Bootstrap Components · Plotly · pandas · scikit-learn
