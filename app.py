import streamlit as st
from predictions import load_model, get_prediction

st.set_page_config(page_title="ml-end-to-end-project")

# Load the model object
model = load_model()

# Title
st.title("ML end to end project")
st.subheader("by Utkarsh Gaikwad")

# Get the inputs from user
sep_len = st.number_input("Sepal Length", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width", min_value=0.00, step=0.01)

# Submit button
submit = st.button("Predict")

if submit:
    preds, prob_df = get_prediction(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predicted Species : {preds}")
    st.subheader("Probability :")
    st.dataframe(prob_df)
    st.bar_chart(prob_df.T)
