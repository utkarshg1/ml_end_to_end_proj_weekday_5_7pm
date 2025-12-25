import joblib
import streamlit as st
import pandas as pd


@st.cache_resource
def load_model(path="notebook/iris_model.joblib"):
    model = joblib.load(path)
    return model


def get_prediction(model, sep_len, sep_wid, pet_len, pet_wid):
    d = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid,
        }
    ]
    df = pd.DataFrame(d)
    pred = model.predict(df)[0]
    probs = model.predict_proba(df)
    classes = model.classes_
    probs_df = pd.DataFrame(probs, columns=classes)
    return pred, probs_df
