import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# config
st.set_page_config(page_title="Prediksi Emisi CO₂", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("best_model_co2.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, scaler, encoder, model_features

@st.cache_data
def load_data():
    return pd.read_csv("CO2 Emissions_Canada.csv")

model, scaler, encoder, model_features = load_model()
df = load_data()

num_cols = list(scaler.feature_names_in_)
cat_cols = list(encoder.feature_names_in_)
cat_options = {c: list(encoder.categories_[i]) for i, c in enumerate(cat_cols)}
cat_ohe_names = list(encoder.get_feature_names_out())

dataset_mean = df["CO2 Emissions(g/km)"].mean()

# ================= MAPPING =================
transmission_map = {
    "A4":"Automatic 4 Speed","A5":"Automatic 5 Speed","A6":"Automatic 6 Speed",
    "A7":"Automatic 7 Speed","A8":"Automatic 8 Speed","A9":"Automatic 9 Speed","A10":"Automatic 10 Speed",
    "AM5":"Automated Manual 5 Speed","AM6":"Automated Manual 6 Speed","AM7":"Automated Manual 7 Speed",
    "AM8":"Automated Manual 8 Speed","AM9":"Automated Manual 9 Speed",
    "AS4":"Automatic Select Shift 4 Speed","AS5":"Automatic Select Shift 5 Speed","AS6":"Automatic Select Shift 6 Speed",
    "AS7":"Automatic Select Shift 7 Speed","AS8":"Automatic Select Shift 8 Speed","AS9":"Automatic Select Shift 9 Speed","AS10":"Automatic Select Shift 10 Speed",
    "AV":"CVT","AV6":"CVT Simulated 6 Speed","AV7":"CVT Simulated 7 Speed","AV8":"CVT Simulated 8 Speed","AV10":"CVT Simulated 10 Speed",
    "M5":"Manual 5 Speed","M6":"Manual 6 Speed","M7":"Manual 7 Speed"
}

fuel_map = {
    "D":"Diesel",
    "E":"Ethanol",
    "N":"Natural Gas",
    "X":"Regular Gasoline",
    "Z":"Premium Gasoline"
}

# ================= FUNCTIONS =================
def l_per_100km_to_mpg(l):
    return 235.214583 / l

def build_input(numeric_input, categorical_input):

    num_df = pd.DataFrame([numeric_input])
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols)

    cat_df = pd.DataFrame([categorical_input])
    cat_transformed = encoder.transform(cat_df)

    if hasattr(cat_transformed, "toarray"):
        cat_transformed = cat_transformed.toarray()

    cat_ohe_df = pd.DataFrame(cat_transformed, columns=cat_ohe_names)

    full = pd.concat([num_scaled_df, cat_ohe_df], axis=1)
    full = full.reindex(columns=model_features, fill_value=0)

    return full

def kategori_emisi(value):
    if value < 150:
        return "Ramah Lingkungan"
    elif value < 250:
        return "Standar"
    else:
        return "Tinggi"

# ================= HEADER =================
st.title("Sistem Prediksi Emisi CO₂ Kendaraan 🚗💨")
st.caption("Sistem ini membantu mengevaluasi dampak lingkungan kendaraan berdasarkan spesifikasi teknis.")
st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3YXEwZWhsamJhb3o1Y2RmZmExYWs0Y2V3Zmpub3k4ZjQ1bnl3aDE4ZSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3gYRUUTFQT7OESoips/giphy.gif")

# ================= SIDEBAR =================
st.sidebar.header("Input Spesifikasi Kendaraan")

engine_size = st.sidebar.slider("Engine Size (L)",1.0,8.0,2.0,0.1)
cylinders = st.sidebar.slider("Cylinders",2,16,4)

fuel_city = st.sidebar.number_input("Fuel Consumption City",2.0,40.0,8.0)
fuel_hwy = st.sidebar.number_input("Fuel Consumption Highway",2.0,40.0,6.0)
fuel_comb = st.sidebar.number_input("Fuel Consumption Combined",2.0,40.0,7.0)

vehicle_class = st.sidebar.selectbox("Vehicle Class",cat_options["vehicle_class"])
transmission = st.sidebar.selectbox("Transmission",cat_options["transmission"])
fuel_type = st.sidebar.selectbox("Fuel Type",cat_options["fuel_type"])

predict = st.sidebar.button("🚀 Analisis Emisi")

# ================= INFO KODE (SELALU TAMPIL) =================
st.divider()
st.info("Klik untuk melihat arti kode kendaraan")

with st.expander("Arti Kode Transmission"):
    st.dataframe(pd.DataFrame({
        "Kode": transmission_map.keys(),
        "Arti": transmission_map.values()
    }), use_container_width=True)

with st.expander("Arti Kode Fuel Type"):
    st.dataframe(pd.DataFrame({
        "Kode": fuel_map.keys(),
        "Arti": fuel_map.values()
    }), use_container_width=True)

# ================= PREDICTION =================
if predict:

    fuel_comb_mpg = l_per_100km_to_mpg(fuel_comb)

    numeric_input = {
        "engine_size": engine_size,
        "cylinders": cylinders,
        "fuelcons_city": fuel_city,
        "fuelcons_hwy": fuel_hwy,
        "fuelcons_comb": fuel_comb,
        "fuelcons_combmpg": fuel_comb_mpg,
    }

    categorical_input = {
        "vehicle_class": vehicle_class,
        "transmission": transmission,
        "fuel_type": fuel_type
    }

    X = build_input(numeric_input, categorical_input)
    prediction = model.predict(X)[0]
    kategori = kategori_emisi(prediction)

    st.write("")
    st.write("")
    st.write("")

    # ===== METRIC =====
    col1,col2,col3 = st.columns(3)
    col1.metric("Prediksi Emisi",f"{prediction:.2f} g/km")
    col2.metric("Rata-rata Dataset",f"{dataset_mean:.2f} g/km")
    col3.metric("Kategori",kategori)

    st.divider()

    # ===== INTERPRETASI =====
    st.subheader("📘 Panduan Interpretasi Emisi")
    st.table(pd.DataFrame({
        "Emisi":["<150","150-250","≥250"],
        "Kategori":["Ramah Lingkungan","Standar","Tinggi"]
    }))

    st.write("")
    st.write("")

    # ===== STATUS =====
    st.subheader("🔎 Evaluasi Kendaraan Anda")

    if kategori=="Ramah Lingkungan":
        st.success("Kendaraan termasuk kategori RAMAH LINGKUNGAN dan memiliki dampak lingkungan rendah.")
    elif kategori=="Standar":
        st.warning("Kendaraan termasuk kategori STANDAR dengan dampak moderat.")
    else:
        st.error("Kendaraan termasuk kategori EMISI TINGGI dan berpotensi meningkatkan polusi udara.")

    st.write("")
    st.write("")

    # estimasi tahunan
    st.subheader("📆 Estimasi Emisi Tahunan")

    annual_km = 15000
    annual_emission_kg = (prediction * annual_km) / 1000

    st.write(f"""Jika kendaraan digunakan sejauh {annual_km:,} km per tahun, maka estimasi total emisi adalah sekitar **{annual_emission_kg:,.2f} kg CO₂ per tahun**.""")

    st.write("")
    st.write("")

    # distribusi dataset
    tab1, tab2, tab3 = st.tabs(["📊 Posisi terhadap Distribusi Dataset", "📈 Hubungan Konsumsi BBM vs Emisi", "📊 Faktor Paling Berpengaruh"])

    with tab1:
        st.subheader("📊 Posisi terhadap Distribusi Dataset")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["CO2 Emissions(g/km)"], bins=30, color='orange', kde=True, ax=ax1)
        ax1.axvline(prediction, color="red", linestyle="--")
        st.pyplot(fig1)

    # bbm vs emisi
    with tab2:
        st.subheader("📈 Hubungan Konsumsi BBM vs Emisi")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            data=df,
            x="Fuel Consumption Comb (L/100 km)",
            y="CO2 Emissions(g/km)",
            alpha=0.3,
            ax=ax2,
            color="orange"
        )
        ax2.scatter(fuel_comb, prediction, color="red", s=100)
        st.pyplot(fig2)

    # feature importance
    with tab3:
        if hasattr(model, "feature_importances_"):
            st.subheader("📊 Faktor Paling Berpengaruh Terhadap Emisi")
            importance = pd.Series(model.feature_importances_, index=model_features)
            top5 = importance.sort_values(ascending=False).head(5)
            fig3, ax3 = plt.subplots()
            top5.sort_values().plot(kind="barh", color="brown", ax=ax3)
            st.pyplot(fig3)

    st.divider()
    st.write("")

    st.subheader("💡 Insight Utama")
    st.write("""
    - Konsumsi BBM faktor paling dominan
    - Engine besar → emisi besar
    - Kendaraan hemat BBM lebih ramah lingkungan
    """)
