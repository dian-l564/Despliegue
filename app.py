"""
Interfaz Streamlit: Clasificador de gama de vehículos usados
Ejecutar localmente: streamlit run app.py
"""
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificador de gama de vehículos", page_icon="🚗", layout="wide")

NOMBRES = {"gama_baja": "Gama baja", "gama_media": "Gama media", "gama_alta": "Gama alta"}
COLORES = {"gama_baja": "#2e7d32", "gama_media": "#f9a825", "gama_alta": "#c62828"}
ORDEN = ["gama_baja", "gama_media", "gama_alta"]


@st.cache_data
def cargar_datos():
    return pd.read_csv("categoria_precio.csv").drop_duplicates()


@st.cache_resource
def cargar_modelo():
    """Carga el modelo guardado; si falla (p. ej. por versiones distintas), lo reentrena."""
    try:
        modelo = joblib.load("modelo_gama_vehiculos.pkl")
        with open("metricas.json") as f:
            metricas = json.load(f)
    except Exception:
        from entrenar_modelo import entrenar
        modelo, metricas = entrenar("categoria_precio.csv")
    return modelo, metricas


df = cargar_datos()
with st.spinner("Cargando el modelo..."):
    modelo, metricas = cargar_modelo()

modelos_por_marca = {m: sorted(df.loc[df["brand"] == m, "model"].unique())
                     for m in sorted(df["brand"].unique())}

if "historial" not in st.session_state:
    st.session_state.historial = []

# ---------------- Encabezado ----------------
st.title("🚗 Clasificador de gama de vehículos usados")
st.write("Ingresa las características del vehículo y el modelo estimará si pertenece "
         "a la **gama baja**, **media** o **alta** según su precio esperado.")

tab_pred, tab_info, tab_hist = st.tabs(["🔮 Predicción", "📊 Acerca del modelo", "🗂️ Historial"])

# ---------------- Pestaña de predicción ----------------
with tab_pred:
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Características del vehículo")
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Marca", list(modelos_por_marca), index=1)
        opciones = modelos_por_marca[brand]
        model = c2.selectbox("Modelo", opciones,
                             index=opciones.index("3 Series") if "3 Series" in opciones else 0)

        year = st.slider("Año", int(df["year"].min()), int(df["year"].max()), 2018)

        c3, c4 = st.columns(2)
        transmission = c3.radio("Transmisión", ["Manual", "Automatic", "Semi-Auto"], index=1)
        fuelType = c4.radio("Combustible", ["Diesel", "Petrol", "Hybrid"])

        c5, c6 = st.columns(2)
        mileage = c5.number_input("Kilometraje (millas)", min_value=0, max_value=300000,
                                  value=30000, step=1000)
        engineSize = c6.number_input("Tamaño del motor (L)", min_value=0.0, max_value=7.0,
                                     value=2.0, step=0.1)
        c7, c8 = st.columns(2)
        tax = c7.number_input("Impuesto anual", min_value=0, max_value=600, value=145, step=5)
        mpg = c8.number_input("Consumo (mpg)", min_value=1.0, max_value=500.0,
                              value=55.0, step=0.5)

        predecir = st.button("Predecir gama", type="primary", width="stretch")

    with col_res:
        st.subheader("Resultado")
        if predecir:
            entrada = pd.DataFrame([{
                "model": model, "year": year, "transmission": transmission,
                "mileage": mileage, "fuelType": fuelType, "tax": tax,
                "mpg": mpg, "engineSize": engineSize, "brand": brand,
            }])
            probs = dict(zip(modelo.classes_, modelo.predict_proba(entrada)[0]))
            gama = max(probs, key=probs.get)
            confianza = probs[gama]

            st.markdown(
                f"<div style='padding:1.2rem;border-radius:12px;background:{COLORES[gama]};"
                f"color:white;text-align:center'>"
                f"<div style='font-size:0.95rem'>Gama estimada</div>"
                f"<div style='font-size:2.2rem;font-weight:700'>{NOMBRES[gama]}</div>"
                f"<div>Confianza: {confianza:.0%}</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            for g in ORDEN:
                st.write(f"{NOMBRES[g]}: **{probs[g]:.1%}**")
                st.progress(float(probs[g]))

            if confianza < 0.5:
                st.warning("Confianza baja: el vehículo está cerca del límite entre dos gamas.")

            registro = entrada.iloc[0].to_dict()
            registro.update({"prediccion": gama, "confianza": round(float(confianza), 3),
                             "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            st.session_state.historial.append(registro)
        else:
            st.info("Completa el formulario y pulsa **Predecir gama**.")

# ---------------- Pestaña de información ----------------
with tab_info:
    st.subheader("Cómo funciona")
    st.write("Modelo de **clasificación supervisada** (Random Forest, 300 árboles) entrenado con "
             f"{metricas['n_entrenamiento']:,} vehículos usados de Audi, BMW y Hyundai. "
             "Las gamas se definieron por terciles del precio.")
    st.table(pd.DataFrame({
        "Gama": ["Gama baja", "Gama media", "Gama alta"],
        "Rango de precio": ["≤ 14.996", "14.997 – 22.990", "> 22.990"],
    }))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (test)", f"{metricas['accuracy']:.1%}")
    c2.metric("F1 gama baja", metricas["f1_por_clase"]["gama_baja"])
    c3.metric("F1 gama media", metricas["f1_por_clase"]["gama_media"])
    c4.metric("F1 gama alta", metricas["f1_por_clase"]["gama_alta"])
    st.caption("Variables más influyentes: kilometraje, año, consumo (mpg), impuesto y tamaño del motor. "
               "Datos de vehículos de 1998 a 2020.")

# ---------------- Pestaña de historial (monitoreo) ----------------
with tab_hist:
    st.subheader("Predicciones de esta sesión")
    if st.session_state.historial:
        hist = pd.DataFrame(st.session_state.historial)
        st.dataframe(hist, width="stretch")
        st.download_button("Descargar historial (CSV)", hist.to_csv(index=False),
                           "registro_predicciones.csv", "text/csv")
    else:
        st.write("Aún no hay predicciones.")
