import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Classification Model", layout="wide")
st.title("Classification Model Dashboard")

# Load scaler and models with caching for performance
@st.cache_resource
def load_artifacts():
  scaler = joblib.load("model/scaler.joblib")
  models = {
      "Logistic Regression": joblib.load( "model/logistic_regression_model.joblib" ),
      "Decision Tree": joblib.load("model/decision_tree_model.joblib"),
      "KNN": joblib.load("model/knn_model.joblib"),
      "Naive Bayes": joblib.load("model/naive_bayes_model.joblib"),
      "Support Vector Machine": joblib.load(
          "model/support_vector_machine_model.joblib"
      ),
  }
  return scaler, models
scaler, models = load_artifacts()
# Sidebar model selector
selected_model_name = st.sidebar.selectbox( "Select Classifier Model", list(models.keys()) )
model = models[selected_model_name]
st.subheader(f"Active Model: {selected_model_name}")
# File uploader for test data
uploaded_file = st.file_uploader("Upload your `test_data.csv` file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", test_df.head())
    if "target" in test_df.columns:
        y_true = test_df["target"]
        X_input = test_df.drop(columns=["target"])
    else:
        y_true = None
        X_input = test_df
    if st.button("Run Predictions"):
        if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_input_scaled = scaler.transform(X_input)
            preds = model.predict(X_input_scaled)
        else:
            preds = model.predict(X_input)
        test_df["Predicted_Class"] = preds
        st.success("Predictions generated successfully!")
        st.dataframe(test_df)
        if y_true is not None:
            st.subheader("Model Evaluation & Performance Report")
            st.write("### Classification Report")
            report = classification_report(y_true, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true, preds)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
else:
  st.info("Please upload the `test_data.csv` file to run predictions.")