# Predicción de Enfermedades en Mascotas

## Descripción
Este proyecto tiene como objetivo predecir enfermedades en mascotas a partir de un conjunto de datos con información clínica. Se utiliza un modelo de aprendizaje automático basado en Random Forest para la clasificación de enfermedades en función de diferentes características como edad, peso, raza y síntomas.

## Estructura del Proyecto
- **datos_mascotas.csv**: Conjunto de datos con información clínica de mascotas.
- **pet_disease_prediction.py**: Código en Python para análisis de datos, entrenamiento y evaluación del modelo.

## Requisitos
- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn

## Uso
1. Instalar las dependencias necesarias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Ejecutar el script de predicción:
   ```bash
   python pet_disease_prediction.py
   ```

## Resultados
El modelo analiza las características de cada mascota y predice la posible enfermedad basada en datos previos. Se incluyen métricas de evaluación y una visualización de la importancia de las características en la predicción.

