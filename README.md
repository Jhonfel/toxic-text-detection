# 🛡️ Detección de Texto Tóxico

## 📝 Descripción del Proyecto

Este proyecto implementa un sistema de detección automática de textos que contienen lenguaje ofensivo (incluyendo género, raza, etc.) o vulgar, utilizando técnicas de aprendizaje automático basadas en árboles de decisión.

### 🎯 Objetivo

Desarrollar un modelo de clasificación utilizando algoritmos basados en árboles de decisión (como Random Forest, XGBoost, LightGBM, etc.) para identificar textos tóxicos en español.

## 🗂️ Estructura del Proyecto

```
toxic-text-detection/
├── 📁 data/
│   ├── 📁 raw/
│   │   └── 📄 data_toxic.csv
│   └── 📁 processed/
├── 📁 notebooks/
│   ├── 📓 01_exploratory_data_analysis.ipynb
│   ├── 📓 02_feature_engineering.ipynb
│   ├── 📓 03_model_development.ipynb
│   └── 📓 04_model_evaluation.ipynb
├── 📁 src/
│   ├── 📁 toxic_text_detection/
│   ├── 📄 __init__.py
│   ├── 📄 data_processing.py
│   ├── 📄 feature_engineering.py
│   └── 📄 model.py
├── 📁 tests/
│   ├── 📄 __init__.py
│   ├── 📄 test_data_processing.py
│   ├── 📄 test_feature_engineering.py
│   └── 📄 test_model.py
├── 📁 models/
│   └── 📄 best_model.pkl
├── 📁 reports/
│   ├── 📁 figures/
│   └── 📓 final_report.ipynb
├── 📁 app/
│   └── 📄 app.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
└── 📄 setup.py
└── 📄 Dockerfile
```

## 🚀 Instalación

Para instalar las dependencias del proyecto:

```bash
pip install -e .
```

## 🖥️ Uso

1. **Exploración de datos:**
   ```
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

2. **Ingeniería de características:**
   ```
   jupyter notebook notebooks/02_feature_engineering.ipynb
   ```

3. **Desarrollo del modelo:**
   ```
   jupyter notebook notebooks/03_model_development.ipynb
   ```

4. **Evaluación del modelo:**
   ```
   jupyter notebook notebooks/04_model_evaluation.ipynb
   ```

## 🧪 Ejecución de pruebas

Para ejecutar las pruebas unitarias:

```bash
python -m unittest discover tests
```

## 📊 Resultados

Los resultados finales y el análisis comparativo de los modelos se pueden encontrar en `reports/final_report.ipynb`.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue primero para discutir lo que te gustaría cambiar.
