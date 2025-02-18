import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("Datos_de_Mascotas.csv")

# Codificar variables categóricas
label_enc = LabelEncoder()
df["Raza"] = label_enc.fit_transform(df["Raza"])
df["Enfermedad"] = label_enc.fit_transform(df["Enfermedad"])

# Definir variables predictoras y objetivo
X = df.drop(columns=["ID", "Enfermedad"])
y = df["Enfermedad"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualización de importancia de características
importances = modelo.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de cada característica en la predicción")
plt.show()
