from pathlib import Path

import joblib
import streamlit as st

from src.predict import MODEL_PATH, predict_mass

import pandas as pd
from src.data import load_penguins, prepare_split

IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "penguin.png"

st.set_page_config(
    page_title="Penguen Kütlesi",
    page_icon="🐧",
    layout="wide",
)

image_column, text_column = st.columns([1, 2])

with image_column:
    st.image(
        IMAGE_PATH,
        width="stretch",
        alt="Antarktika manzarasında bir penguen illüstrasyonu",
    )

with text_column:
    st.title("Penguenin vücut kütlesi")
    st.write(
        "Türü ve üç ölçümü gir. Eğitilmiş doğrusal regresyon modeli "
        "tahmini ağırlığı gram cinsinden hesaplasın."
    )
    st.caption(
        "Penguen görseli açıklama amaçlıdır; modelin girdisi fotoğraf değildir."
    )

with st.form("penguin_measurements"):
    species = st.selectbox("Tür", ["Adelie", "Chinstrap", "Gentoo"])
    bill_length = st.number_input(
        "Gaga uzunluğu (mm)", min_value=0.1, value=39.5, step=0.1
    )
    bill_depth = st.number_input(
        "Gaga derinliği (mm)", min_value=0.1, value=17.4, step=0.1
    )
    flipper_length = st.number_input(
        "Yüzgeç uzunluğu (mm)", min_value=0.1, value=186.0, step=1.0
    )
    submitted = st.form_submit_button("Ağırlığı tahmin et")

if submitted:
    try:
        model = joblib.load(MODEL_PATH)
        predicted_g = predict_mass(
            model, species, bill_length, bill_depth, flipper_length
        )
    except FileNotFoundError:
        st.error("Kayıtlı model bulunamadı. Önce train.py dosyasını çalıştır.")
    except ValueError as error:
        st.error(f"Geçersiz girdi: {error}")
    else:
        st.metric("Tahmini vücut kütlesi", f"{predicted_g:.1f} g")
        st.caption(
            "Bu tek bir penguenin tahminidir. Gerçek ağırlığı bilinmeden "
            "bu penguenin tahmin hatası hesaplanamaz."
        )

st.divider()
st.subheader("Model nasıl hesaplıyor?")

st.latex(
    r"\hat m = b_0 + b_1 L_g + b_2 D_g + b_3 L_y + b_4 I_C + b_5 I_G"
)
st.caption(
    "L_g: gaga uzunluğu, D_g: gaga derinliği, L_y: yüzgeç uzunluğu. "
    "I_C ve I_G, tür Chinstrap veya Gentoo ise 1; değilse 0 olur. "
    "Adelie için ikisi de 0'dır."
)

st.subheader("Test verisinde ortalama hata")

_, train, test = prepare_split(load_penguins())
saved_model = joblib.load(MODEL_PATH)

actual = test["body_mass_g"]
baseline_g = train["body_mass_g"].mean()
predicted = pd.Series(
    saved_model.predict(test[list(saved_model.feature_names_in_)]),
    index=test.index,
)

baseline_mae = (actual - baseline_g).abs().mean()
model_mae = (actual - predicted).abs().mean()

error_chart = pd.DataFrame(
    {"MAE (g)": [baseline_mae, model_mae]},
    index=["Baseline", "Ölçümler + tür"],
)

st.bar_chart(error_chart, horizontal=True, height=220)

reduction_pct = (baseline_mae - model_mae) / baseline_mae * 100
st.metric(
    "Bu test ayrımında MAE azalması",
    f"%{reduction_pct:.1f}",
)

st.caption(
    f"Aynı {len(test)} test pengueninde baseline MAE {baseline_mae:.1f} g, "
    f"model MAE {model_mae:.1f} g. Bu bir güven aralığı değildir."
)