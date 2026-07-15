
import pandas as pd
import streamlit as st

from pathlib import Path


st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)


BASE_DIRECTORY = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIRECTORY
    / "Bank_Customer_Churn_Outputs"
    / "Customer_Churn_Dashboard.csv"
)


@st.cache_data
def load_dashboard_data():
    return pd.read_csv(DATA_PATH)


st.title(
    "Bank Customer Churn Risk Dashboard"
)

st.caption(
    "Customer Churn Prediction, Risk Scoring "
    "and Retention Analytics"
)


if not DATA_PATH.exists():
    st.error(
        "Customer_Churn_Dashboard.csv was not found."
    )
    st.stop()


dashboard_data = load_dashboard_data()


total_customers = len(dashboard_data)

average_churn_probability = dashboard_data[
    "Churn Probability Percent"
].mean()

high_risk_customers = (
    dashboard_data["Risk Level"] == "High Risk"
).sum()

predicted_churners = dashboard_data[
    "Predicted Churn"
].sum()


column_one, column_two, column_three, column_four = (
    st.columns(4)
)


column_one.metric(
    "Total Customers",
    f"{total_customers:,}"
)

column_two.metric(
    "Average Churn Probability",
    f"{average_churn_probability:.2f}%"
)

column_three.metric(
    "High Risk Customers",
    f"{high_risk_customers:,}"
)

column_four.metric(
    "Predicted Churners",
    f"{predicted_churners:,}"
)


st.subheader(
    "Customer Risk Data Preview"
)

st.dataframe(
    dashboard_data.head(20),
    use_container_width=True,
    hide_index=True
)
