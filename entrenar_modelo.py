"""
Entrena el modelo Random Forest y lo guarda comprimido en modelo_gama_vehiculos.pkl
Uso: python entrenar_modelo.py
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

CATEGORICAS = ["model", "transmission", "fuelType", "brand"]
NUMERICAS = ["year", "mileage", "tax", "mpg", "engineSize"]


def entrenar(ruta_csv="categoria_precio.csv"):
    df = pd.read_csv(ruta_csv).drop_duplicates()
    X = df.drop(columns="categoria_precio")
    y = df["categoria_precio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
    ])
    modelo = Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=42)),
    ])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    metricas = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "f1_por_clase": dict(zip(
            ["gama_baja", "gama_media", "gama_alta"],
            [round(float(v), 3) for v in f1_score(y_test, pred, average=None,
                                           labels=["gama_baja", "gama_media", "gama_alta"])],
        )),
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
    }
    return modelo, metricas


if __name__ == "__main__":
    modelo, metricas = entrenar()
    joblib.dump(modelo, "modelo_gama_vehiculos.pkl", compress=3)
    with open("metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("Modelo guardado. Métricas:", metricas)
