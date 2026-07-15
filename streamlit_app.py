from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


NAVY = "#0B1F3A"
BLUE = "#1F4E78"
TEAL = "#148F9C"
GOLD = "#D4A72C"
RED = "#C44747"
GREEN = "#2E7D5B"
LIGHT_BLUE = "#EAF1F8"
RISK_COLORS = {
    "Low Risk": GREEN,
    "Medium Risk": GOLD,
    "High Risk": RED,
}
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
GEOGRAPHY_COLORS = {
    "Germany": BLUE,
    "Spain": TEAL,
    "France": GOLD,
}
GEOGRAPHY_ORDER = ["Germany", "Spain", "France"]


st.markdown(
    f"""
    <style>
    .stApp {{ background: #F4F7FB; }}
    [data-testid="stHeader"] {{ background: rgba(244,247,251,0.92); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, #123B62 100%);
        border-right: 1px solid rgba(255,255,255,0.12);
    }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF; }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {{
        background-color: rgba(255,255,255,0.10);
        border-color: rgba(255,255,255,0.24);
    }}
    .block-container {{ max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }}
    h1, h2, h3, h4 {{ color: {NAVY} !important; letter-spacing: -0.02em; }}
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label {{ color: #4F6174; }}
    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(11,31,58,0.06);
    }}
    [data-testid="stMetricLabel"] {{ color: #5E6C7B; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 750; }}
    .bank-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 68%, {TEAL} 100%);
        box-shadow: 0 12px 32px rgba(11,31,58,0.18);
    }}
    .bank-mark {{
        width: 58px;
        height: 58px;
        display: grid;
        place-items: center;
        border: 2px solid rgba(255,255,255,0.70);
        border-radius: 50%;
        font-size: 30px;
        background: rgba(255,255,255,0.10);
    }}
    .bank-header h1 {{ color: #FFFFFF !important; margin: 0; font-size: 2rem; }}
    .bank-header p {{ color: #DCE9F5 !important; margin: 5px 0 0; font-size: 0.98rem; }}
    .section-note {{ color: #647487; margin-top: -8px; margin-bottom: 16px; }}
    .page-title-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 18px 0;
    }}
    .page-title-row h3 {{
        margin: 0;
        color: {NAVY} !important;
        font-size: 1.75rem;
        line-height: 1.2;
    }}
    .navigation-symbol {{
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        flex: 0 0 38px;
        border-radius: 11px;
        color: #FFFFFF;
        background: linear-gradient(135deg, {BLUE}, {TEAL});
        border: 1px solid rgba(255,255,255,0.55);
        box-shadow: 0 4px 12px rgba(11,31,58,0.16);
        font-size: 1.25rem;
        cursor: default;
    }}
    .insight-card {{
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-left: 5px solid {TEAL};
        border-radius: 12px;
        padding: 16px 18px;
        margin: 8px 0;
        color: #263849;
        box-shadow: 0 4px 14px rgba(11,31,58,0.05);
    }}
    .risk-low {{ border-left-color: {GREEN}; }}
    .risk-medium {{ border-left-color: {GOLD}; }}
    .risk-high {{ border-left-color: {RED}; }}
    .governance-notepad {{
        position: relative;
        max-width: 1120px;
        min-height: 340px;
        margin: 18px auto 28px auto;
        padding: 28px 38px 28px 72px;
        color: #263849;
        background:
            linear-gradient(90deg, transparent 0 48px, #E7A0A0 49px 51px, transparent 52px),
            repeating-linear-gradient(#FFFFFF 0 38px, #DCE7F0 39px 40px);
        border: 1px solid #C9D8E8;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(11,31,58,0.10);
    }}
    .governance-notepad h3 {{
        margin: 0 0 10px 0;
        line-height: 40px;
        color: {NAVY} !important;
    }}
    .governance-notepad ul {{ margin: 0; padding: 5px 0 0 22px; }}
    .governance-notepad li {{
        margin: 0;
        padding-left: 5px;
        color: #263849;
        line-height: 40px;
        min-height: 40px;
    }}
    .model-status-card {{
        min-height: 113px;
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(11,31,58,0.06);
    }}
    .model-status-label {{
        color: #5E6C7B;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .model-status-value {{
        color: {NAVY};
        font-size: 1.62rem;
        line-height: 1.15;
        font-weight: 750;
        white-space: nowrap;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid #DCE5EF;
        border-radius: 12px;
        overflow: hidden;
    }}
    [data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid #C9D8E8;
        border-radius: 14px;
        padding: 10px 12px 4px 12px;
        box-shadow: 0 6px 18px rgba(11,31,58,0.07);
        overflow: hidden;
    }}
    .stDownloadButton button, .stFormSubmitButton button {{
        background: {BLUE};
        color: white;
        border: 0;
        border-radius: 9px;
        font-weight: 650;
    }}
    .stDownloadButton button *,
    .stFormSubmitButton button * {{ color: #FFFFFF !important; }}
    .stDownloadButton button:hover, .stFormSubmitButton button:hover {{
        background: {NAVY};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "Bank_Customer_Churn_Outputs"
DATA_PATH = OUTPUT_DIRECTORY / "Customer_Churn_Dashboard.csv"
FEATURE_PATH = OUTPUT_DIRECTORY / "Feature_Importance_Summary.csv"
SHAP_PATH = OUTPUT_DIRECTORY / "SHAP_Importance_Summary.csv"
MODEL_PATH = OUTPUT_DIRECTORY / "Model_Comparison_Summary.csv"
CV_PATH = OUTPUT_DIRECTORY / "Cross_Validation_Summary.csv"
RISK_PATH = OUTPUT_DIRECTORY / "Risk_Level_Summary.csv"
THRESHOLD_PATH = OUTPUT_DIRECTORY / "Threshold_Analysis.csv"
BUNDLE_PATH = OUTPUT_DIRECTORY / "Gradient_Boosting_Churn_Model_Bundle.pkl"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model_bundle(path: Path):
    return joblib.load(path)


def create_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["Balance to Salary Ratio"] = result["Balance"] / (
        result["Estimated Salary"] + 1
    )
    result["Product Density"] = result["Number of Products"] / (
        result["Tenure"] + 1
    )
    result["Engagement Score"] = (
        result["Is Active Member"] * result["Number of Products"]
    )
    result["Age Tenure Interaction"] = result["Age"] * result["Tenure"]
    return result


def retention_action(number_of_products: int, is_active_member: int) -> str:
    if number_of_products >= 3:
        return "Review product suitability, fees, service quality, and complexity."
    if is_active_member == 0 and number_of_products == 1:
        return "Priority re-engagement and needs-based second-product review."
    if is_active_member == 0:
        return "Relationship-manager outreach and digital engagement support."
    if number_of_products == 1:
        return "Needs-based product review and personalized loyalty offer."
    return "Personalized retention call and customer-service review."


def header() -> None:
    st.markdown(
        """
        <div class="bank-header">
            <div class="bank-mark">🏦</div>
            <div>
                <h1>Bank Customer Churn Intelligence</h1>
                <p>Predictive risk scoring, retention prioritization and model intelligence</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str) -> None:
    st.markdown(
        f"""
        <div class="page-title-row">
            <div class="navigation-symbol" title="Use the page menu on the left">☰</div>
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(probability: float, low_limit: float, high_limit: float) -> str:
    if probability < low_limit:
        return "Low Risk"
    if probability < high_limit:
        return "Medium Risk"
    return "High Risk"


def style_figure(figure, legend_title: str = "Legend"):
    figure.update_layout(
        font=dict(family="Arial", color="#263849", size=13),
        title=dict(x=0.5, xanchor="center", font=dict(size=19, color=NAVY)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            title_text=legend_title,
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="#FFFFFF",
            bordercolor="#C9D8E8",
            borderwidth=1,
            font=dict(color=NAVY, size=12),
            title_font=dict(color=NAVY, size=12),
        ),
        xaxis=dict(
            tickfont=dict(color="#4F6174"),
            title_font=dict(color=NAVY),
            linecolor="#AFC0D1",
        ),
        yaxis=dict(
            tickfont=dict(color="#4F6174"),
            title_font=dict(color=NAVY),
            linecolor="#AFC0D1",
        ),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=NAVY),
    )
    return figure


def kpi_row(data: pd.DataFrame) -> None:
    total = len(data)
    average_probability = data["Churn Probability Percent"].mean() if total else 0
    high_risk = int((data["Risk Level"] == "High Risk").sum())
    medium_risk = int((data["Risk Level"] == "Medium Risk").sum())
    predicted = int(data["Predicted Churn"].sum())
    historical_churn_rate = data["Actual Churn"].mean() * 100 if total else 0
    columns = st.columns(6)
    columns[0].metric("Customers in View", f"{total:,}")
    columns[1].metric("Average Churn Risk", f"{average_probability:.2f}%")
    columns[2].metric("High Risk Customers", f"{high_risk:,}")
    columns[3].metric("Medium Risk Customers", f"{medium_risk:,}")
    columns[4].metric("Predicted Churners", f"{predicted:,}")
    columns[5].metric("Historical Churn Rate", f"{historical_churn_rate:.2f}%")


if not DATA_PATH.exists():
    st.error(
        "Dashboard data was not found. Keep streamlit_app.py beside the "
        "Bank_Customer_Churn_Outputs folder."
    )
    st.stop()


data = load_csv(DATA_PATH)
feature_importance = load_csv(FEATURE_PATH) if FEATURE_PATH.exists() else pd.DataFrame()
shap_importance = load_csv(SHAP_PATH) if SHAP_PATH.exists() else pd.DataFrame()
model_comparison = load_csv(MODEL_PATH) if MODEL_PATH.exists() else pd.DataFrame()
cross_validation = load_csv(CV_PATH) if CV_PATH.exists() else pd.DataFrame()
risk_summary = load_csv(RISK_PATH) if RISK_PATH.exists() else pd.DataFrame()
threshold_analysis = (
    load_csv(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else pd.DataFrame()
)


st.sidebar.markdown("## 🏦 Churn Intelligence")
st.sidebar.caption("European retail banking portfolio")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Risk Analytics",
        "Customer Explorer",
        "Scenario Simulator",
        "Model Intelligence",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Portfolio Filters")

geography_options = sorted(data["Geography"].dropna().unique().tolist())
gender_options = sorted(data["Gender"].dropna().unique().tolist())
selected_geography = st.sidebar.selectbox(
    "Geography", ["All Geographies"] + geography_options
)
selected_risk = st.sidebar.selectbox(
    "Risk Level", ["All Risk Levels"] + RISK_ORDER
)
selected_gender = st.sidebar.selectbox(
    "Gender", ["All Genders"] + gender_options
)
probability_range = st.sidebar.slider(
    "Churn Probability (%)", 0, 100, (0, 100), step=1
)
st.sidebar.markdown("---")
st.sidebar.caption("Champion model: Gradient Boosting")
st.sidebar.caption("Classification threshold: 35%")
st.sidebar.caption("Risk bands: Low <30% · High ≥70%")


filtered_data = data[
    data["Churn Probability Percent"].between(
        probability_range[0], probability_range[1]
    )
].copy()

if selected_geography != "All Geographies":
    filtered_data = filtered_data[
        filtered_data["Geography"] == selected_geography
    ]

if selected_risk != "All Risk Levels":
    filtered_data = filtered_data[
        filtered_data["Risk Level"] == selected_risk
    ]

if selected_gender != "All Genders":
    filtered_data = filtered_data[
        filtered_data["Gender"] == selected_gender
    ]


header()

page_title_labels = {
    "Executive Overview": "Executive Overview",
    "Risk Analytics": "Risk Analytics",
    "Customer Explorer": "Customer Explorer",
    "Scenario Simulator": "Scenario Simulator",
    "Model Intelligence": "Model Intelligence and Governance",
}
page_title(page_title_labels[page])


if filtered_data.empty:
    st.warning(
        "No customers match this filter combination. "
        "The complete portfolio is shown instead."
    )
    filtered_data = data.copy()


if page == "Executive Overview":
    kpi_row(filtered_data)
    st.markdown("### Portfolio Risk Overview")
    st.markdown(
        '<p class="section-note">Risk concentration and the customer segments requiring attention.</p>',
        unsafe_allow_html=True,
    )

    risk_counts = (
        filtered_data["Risk Level"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("Risk Level")
        .reset_index(name="Customers")
    )
    risk_counts["Customer Share"] = (
        risk_counts["Customers"] / risk_counts["Customers"].sum() * 100
    )
    geography_risk = (
        filtered_data.groupby("Geography", as_index=False)
        .agg(
            Customers=("Customer ID", "count"),
            Average_Risk=("Churn Probability Percent", "mean"),
        )
        .rename(columns={"Average_Risk": "Average Risk"})
        .sort_values("Average Risk", ascending=False)
    )

    left, right = st.columns([1, 1.35])
    with left:
        risk_figure = px.pie(
            risk_counts,
            names="Risk Level",
            values="Customers",
            hole=0.62,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        risk_figure.update_traces(
            textposition="outside",
            textinfo="label+percent+value",
            textfont_size=13,
            hovertemplate=(
                "<b>%{label}</b><br>Customers: %{value:,}"
                "<br>Portfolio Share: %{percent}<extra></extra>"
            ),
        )
        risk_figure.update_layout(
            title="Customer Risk Mix",
            height=470,
            margin=dict(l=30, r=30, t=65, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        style_figure(risk_figure, "Risk Level")
        st.plotly_chart(risk_figure, width="stretch")
    with right:
        geography_figure = px.bar(
            geography_risk,
            x="Geography",
            y="Average Risk",
            color="Geography",
            color_discrete_map=GEOGRAPHY_COLORS,
            category_orders={"Geography": GEOGRAPHY_ORDER},
            text_auto=".1f",
            custom_data=["Customers"],
        )
        geography_figure.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>Average Risk: %{y:.2f}%"
                "<br>Customers: %{customdata[0]:,}<extra></extra>"
            ),
        )
        geography_figure.update_layout(
            title="Average Churn Risk by Geography",
            height=470,
            yaxis_title="Average Churn Probability (%)",
            xaxis_title="Geography",
            margin=dict(l=20, r=20, t=65, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#E6ECF2", ticksuffix="%"),
        )
        style_figure(geography_figure, "Geography")
        st.plotly_chart(geography_figure, width="stretch")

    left, right = st.columns([1.35, 1])
    with left:
        probability_figure = px.histogram(
            filtered_data,
            x="Churn Probability Percent",
            nbins=25,
            color="Risk Level",
            category_orders={"Risk Level": RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            barmode="stack",
        )
        probability_figure.update_layout(
            title="Churn Probability Distribution",
            height=470,
            xaxis_title="Churn Probability (%)",
            yaxis_title="Customers",
            bargap=0.04,
            margin=dict(l=20, r=20, t=65, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#E6ECF2"),
        )
        probability_figure.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Probability Bin: %{x:.1f}%"
                "<br>Customers: %{y:,}<extra></extra>"
            )
        )
        style_figure(probability_figure, "Risk Level")
        st.plotly_chart(probability_figure, width="stretch")
    with right:
        high_risk = filtered_data[filtered_data["Risk Level"] == "High Risk"]
        inactive_high = int((high_risk["Is Active Member"] == 0).sum())
        complex_products = int((high_risk["Number of Products"] >= 3).sum())
        st.markdown("#### Executive Signals")
        st.markdown(
            f'<div class="insight-card risk-high"><b>{len(high_risk):,}</b> customers are currently classified as High Risk.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-card risk-medium"><b>{inactive_high:,}</b> High Risk customers are inactive and suitable for re-engagement.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-card"><b>{complex_products:,}</b> High Risk customers hold three or more products and require suitability review.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Risk Band Performance Summary")
    executive_risk_summary = (
        filtered_data.groupby("Risk Level", observed=False)
        .agg(
            Customers=("Customer ID", "count"),
            Actual_Churners=("Actual Churn", "sum"),
            Average_Risk=("Churn Probability Percent", "mean"),
            Historical_Churn_Rate=("Actual Churn", "mean"),
        )
        .reindex(RISK_ORDER)
        .reset_index()
        .rename(
            columns={
                "Actual_Churners": "Actual Churners",
                "Average_Risk": "Average Churn Probability",
                "Historical_Churn_Rate": "Historical Churn Rate",
            }
        )
    )
    executive_risk_summary["Customer Share"] = (
        executive_risk_summary["Customers"]
        / executive_risk_summary["Customers"].sum()
        * 100
    )
    executive_risk_summary["Historical Churn Rate"] *= 100
    executive_risk_summary["Churner Share"] = (
        executive_risk_summary["Actual Churners"]
        / executive_risk_summary["Actual Churners"].sum()
        * 100
    )

    left, right = st.columns(2)
    with left:
        calibration_data = executive_risk_summary.melt(
            id_vars="Risk Level",
            value_vars=["Average Churn Probability", "Historical Churn Rate"],
            var_name="Measure",
            value_name="Rate",
        )
        calibration_figure = px.bar(
            calibration_data,
            x="Risk Level",
            y="Rate",
            color="Measure",
            barmode="group",
            category_orders={"Risk Level": RISK_ORDER},
            color_discrete_sequence=[BLUE, GOLD],
            text_auto=".1f",
        )
        calibration_figure.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Risk Level: %{x}"
                "<br>Rate: %{y:.2f}%<extra></extra>"
            ),
        )
        calibration_figure.update_layout(
            title="Predicted Risk vs Historical Churn",
            height=470,
            xaxis_title="Risk Level",
            yaxis_title="Rate (%)",
            yaxis=dict(gridcolor="#E6ECF2", ticksuffix="%"),
            margin=dict(l=25, r=25, t=70, b=95),
        )
        style_figure(calibration_figure, "Measure")
        st.plotly_chart(calibration_figure, width="stretch")

    with right:
        churner_figure = px.bar(
            executive_risk_summary,
            x="Risk Level",
            y="Actual Churners",
            color="Risk Level",
            category_orders={"Risk Level": RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            text="Churner Share",
        )
        churner_figure.update_traces(
            texttemplate="%{y:,}<br>(%{text:.1f}%)",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>Actual Churners: %{y:,}"
                "<br>Share of All Churners: %{text:.2f}%<extra></extra>"
            ),
        )
        churner_figure.update_layout(
            title="Actual Churner Concentration",
            height=470,
            xaxis_title="Risk Level",
            yaxis_title="Actual Churners",
            yaxis=dict(gridcolor="#E6ECF2"),
            margin=dict(l=25, r=25, t=70, b=95),
        )
        style_figure(churner_figure, "Risk Level")
        st.plotly_chart(churner_figure, width="stretch")

    st.markdown("### Priority Customer Queue")
    priority_columns = [
        "Customer ID",
        "Surname",
        "Geography",
        "Age",
        "Number of Products",
        "Is Active Member",
        "Churn Probability Percent",
        "Risk Level",
        "Recommended Retention Action",
    ]
    st.dataframe(
        filtered_data.sort_values("Churn Probability Percent", ascending=False)[
            priority_columns
        ].head(20),
        width="stretch",
        hide_index=True,
        column_config={
            "Churn Probability Percent": st.column_config.ProgressColumn(
                "Churn Probability", format="%.2f%%", min_value=0, max_value=100
            ),
            "Is Active Member": st.column_config.CheckboxColumn("Active Member"),
        },
    )


elif page == "Risk Analytics":
    kpi_row(filtered_data)
    distribution_tab, segment_tab, action_tab = st.tabs(
        ["Risk Distribution", "Segment Analysis", "Retention Priorities"]
    )

    with distribution_tab:
        risk_table = (
            filtered_data.groupby("Risk Level", observed=False)
            .agg(
                Customers=("Customer ID", "count"),
                Average_Risk=("Churn Probability Percent", "mean"),
                Historical_Churn_Rate=("Actual Churn", "mean"),
            )
            .reindex(RISK_ORDER)
            .reset_index()
            .rename(
                columns={
                    "Average_Risk": "Average Churn Probability",
                    "Historical_Churn_Rate": "Historical Churn Rate",
                }
            )
        )
        risk_table["Historical Churn Rate"] *= 100
        st.markdown("#### Risk Band Validation Table")
        st.dataframe(
            risk_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Average Churn Probability": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "Historical Churn Rate": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        risk_bar = px.bar(
            risk_table,
            x="Risk Level",
            y=["Average Churn Probability", "Historical Churn Rate"],
            barmode="group",
            category_orders={"Risk Level": RISK_ORDER},
            color_discrete_sequence=[BLUE, GOLD],
            text_auto=".1f",
        )
        risk_bar.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Risk Band: %{x}"
                "<br>Rate: %{y:.2f}%<extra></extra>"
            ),
        )
        risk_bar.update_layout(
            title="Predicted Risk versus Historical Churn",
            height=500,
            yaxis_title="Rate (%)",
            xaxis_title="Risk Level",
            margin=dict(l=20, r=20, t=65, b=85),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#E6ECF2", ticksuffix="%"),
        )
        style_figure(risk_bar, "Measure")
        st.plotly_chart(risk_bar, width="stretch")

    with segment_tab:
        left, right = st.columns(2)
        geography_segment = (
            filtered_data.groupby("Geography", as_index=False)
            .agg(
                Customers=("Customer ID", "count"),
                Average_Risk=("Churn Probability Percent", "mean"),
            )
            .rename(columns={"Average_Risk": "Average Risk"})
        )
        product_segment = (
            filtered_data.groupby("Number of Products", as_index=False)
            .agg(
                Customers=("Customer ID", "count"),
                Average_Risk=("Churn Probability Percent", "mean"),
            )
            .rename(columns={"Average_Risk": "Average Risk"})
        )
        with left:
            figure = px.bar(
                geography_segment,
                x="Geography",
                y="Average Risk",
                text_auto=".1f",
                color="Geography",
                color_discrete_map=GEOGRAPHY_COLORS,
                category_orders={"Geography": GEOGRAPHY_ORDER},
                custom_data=["Customers"],
            )
            figure.update_traces(
                texttemplate="%{y:.1f}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>Average Risk: %{y:.2f}%"
                    "<br>Customers: %{customdata[0]:,}<extra></extra>"
                ),
            )
            figure.update_layout(
                title="Average Risk by Geography",
                height=470,
                xaxis_title="Geography",
                yaxis_title="Average Churn Probability (%)",
                margin=dict(l=20, r=20, t=65, b=85),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#E6ECF2", ticksuffix="%"),
            )
            style_figure(figure, "Geography")
            st.plotly_chart(figure, width="stretch")
        with right:
            figure = px.bar(
                product_segment,
                x="Number of Products",
                y="Average Risk",
                text_auto=".1f",
                color="Number of Products",
                color_continuous_scale=["#B7D7E8", GOLD, RED],
                custom_data=["Customers"],
            )
            figure.update_traces(
                texttemplate="%{y:.1f}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x} Products</b><br>Average Risk: %{y:.2f}%"
                    "<br>Customers: %{customdata[0]:,}<extra></extra>"
                ),
            )
            figure.update_layout(
                title="Average Risk by Product Count",
                height=470,
                xaxis_title="Number of Products",
                yaxis_title="Average Churn Probability (%)",
                coloraxis_colorbar_title="Risk %",
                margin=dict(l=20, r=70, t=65, b=85),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#E6ECF2", ticksuffix="%"),
            )
            style_figure(figure, "Product Count")
            st.plotly_chart(figure, width="stretch")

    with action_tab:
        action_summary = (
            filtered_data.groupby("Recommended Retention Action", as_index=False)
            .agg(
                Customers=("Customer ID", "count"),
                Average_Risk=("Churn Probability Percent", "mean"),
            )
            .rename(columns={"Average_Risk": "Average Risk"})
            .sort_values("Customers", ascending=False)
        )
        action_figure = px.bar(
            action_summary,
            x="Customers",
            y="Recommended Retention Action",
            orientation="h",
            color="Average Risk",
            color_continuous_scale=["#B7D7E8", GOLD, RED],
            text="Customers",
        )
        action_figure.update_traces(
            texttemplate="%{x:,}",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Customers: %{x:,}"
                "<br>Average Risk: %{marker.color:.2f}%<extra></extra>"
            ),
        )
        action_figure.update_layout(
            title="Retention Workload by Recommended Action",
            height=500,
            yaxis_title="",
            xaxis_title="Customers Requiring Action",
            coloraxis_colorbar_title="Risk %",
            margin=dict(l=20, r=90, t=65, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        style_figure(action_figure, "Average Risk")
        st.plotly_chart(action_figure, width="stretch")
        st.markdown("#### Retention Action Summary Table")
        st.dataframe(action_summary, width="stretch", hide_index=True)


elif page == "Customer Explorer":
    st.markdown(
        '<p class="section-note">Search the filtered portfolio and export an operational retention list.</p>',
        unsafe_allow_html=True,
    )
    search_term = st.text_input(
        "Search by Customer ID or Surname", placeholder="Enter an ID or surname"
    ).strip()
    explorer_data = filtered_data.copy()
    if search_term:
        explorer_data = explorer_data[
            explorer_data["Customer ID"].astype(str).str.contains(
                search_term, case=False, na=False
            )
            | explorer_data["Surname"].astype(str).str.contains(
                search_term, case=False, na=False
            )
        ]
    st.caption(f"{len(explorer_data):,} matching customers")
    explorer_columns = [
        "Customer ID",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "Credit Score",
        "Balance",
        "Number of Products",
        "Has Credit Card",
        "Is Active Member",
        "Churn Probability Percent",
        "Risk Level",
        "Predicted Churn",
        "Recommended Retention Action",
    ]
    st.markdown("#### Customer-Level Risk Register")
    st.dataframe(
        explorer_data[explorer_columns],
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Balance": st.column_config.NumberColumn(format="%,.2f"),
            "Churn Probability Percent": st.column_config.ProgressColumn(
                "Churn Probability", format="%.2f%%", min_value=0, max_value=100
            ),
            "Has Credit Card": st.column_config.CheckboxColumn("Credit Card"),
            "Is Active Member": st.column_config.CheckboxColumn("Active Member"),
        },
    )
    st.download_button(
        "Download Filtered Customer List",
        data=explorer_data[explorer_columns].to_csv(index=False).encode("utf-8"),
        file_name="Filtered_Customer_Retention_List.csv",
        mime="text/csv",
    )


elif page == "Scenario Simulator":
    st.markdown(
        '<p class="section-note">Adjust a customer profile and observe the model-generated churn risk through the what-if simulator.</p>',
        unsafe_allow_html=True,
    )
    if not BUNDLE_PATH.exists():
        st.error("The saved model bundle was not found.")
        st.stop()
    bundle = load_model_bundle(BUNDLE_PATH)
    with st.form("scenario_form"):
        first, second, third = st.columns(3)
        with first:
            credit_score = st.slider("Credit Score", 350, 850, 650)
            age = st.slider("Age", 18, 92, 39)
            tenure = st.slider("Tenure", 0, 10, 5)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        with second:
            balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=5000.0)
            estimated_salary = st.number_input(
                "Estimated Salary", min_value=0.0, value=100000.0, step=5000.0
            )
            number_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            gender = st.selectbox("Gender", ["Female", "Male"])
        with third:
            has_credit_card = st.selectbox(
                "Has Credit Card", [0, 1], format_func=lambda value: "Yes" if value else "No"
            )
            is_active_member = st.selectbox(
                "Is Active Member", [0, 1], format_func=lambda value: "Yes" if value else "No"
            )
            st.info(
                "The simulator supports retention planning. Demographic variables should not be used for discriminatory treatment."
            )
        submitted = st.form_submit_button("Calculate Churn Risk", width="stretch")

    if submitted:
        scenario = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Geography": [geography],
                "Gender": [gender],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "Number of Products": [number_of_products],
                "Has Credit Card": [has_credit_card],
                "Is Active Member": [is_active_member],
                "Estimated Salary": [estimated_salary],
            }
        )
        scenario = create_engineered_features(scenario)
        transformed = bundle["Preprocessor"].transform(scenario)
        transformed = pd.DataFrame(
            transformed, columns=bundle["Processed Feature Names"]
        )
        probability = float(bundle["Model"].predict_proba(transformed)[0, 1])
        prediction = int(probability >= bundle["Selected Threshold"])
        level = risk_label(
            probability,
            bundle["Low Risk Upper Limit"],
            bundle["High Risk Lower Limit"],
        )
        action = retention_action(number_of_products, is_active_member)

        left, right = st.columns([1, 1.15])
        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"color": NAVY}},
                    title={"text": "Predicted Churn Risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RISK_COLORS[level]},
                        "steps": [
                            {"range": [0, 30], "color": "#DCEFE6"},
                            {"range": [30, 70], "color": "#F7EAC0"},
                            {"range": [70, 100], "color": "#F3D6D6"},
                        ],
                        "threshold": {
                            "line": {"color": NAVY, "width": 4},
                            "value": bundle["Selected Threshold"] * 100,
                        },
                    },
                )
            )
            gauge.update_layout(
                height=330,
                margin=dict(l=30, r=30, t=70, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge, width="stretch")
        with right:
            risk_class = {
                "Low Risk": "risk-low",
                "Medium Risk": "risk-medium",
                "High Risk": "risk-high",
            }[level]
            st.markdown("#### Scenario Result")
            st.markdown(
                f'<div class="insight-card {risk_class}"><b>Risk Level:</b> {level}<br><b>Binary Churn Flag:</b> {prediction}<br><b>Decision Threshold:</b> {bundle["Selected Threshold"]:.0%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="insight-card"><b>Recommended Action</b><br>{action}</div>',
                unsafe_allow_html=True,
            )


elif page == "Model Intelligence":
    model_status_columns = st.columns(4)
    with model_status_columns[0]:
        st.markdown(
            """
            <div class="model-status-card">
                <div class="model-status-label">Champion Model</div>
                <div class="model-status-value">Gradient Boosting</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    model_status_columns[1].metric("Test ROC-AUC", "87.22%")
    model_status_columns[2].metric("Five-Fold CV ROC-AUC", "86.43%")
    model_status_columns[3].metric("Selected Threshold", "35%")
    performance_tab, importance_tab, governance_tab = st.tabs(
        ["Model Performance", "Feature Importance", "Governance Notes"]
    )
    with performance_tab:
        if model_comparison.empty:
            st.warning("Model comparison file is unavailable.")
        else:
            st.markdown("#### Test-Set Metric Table")
            st.dataframe(model_comparison, width="stretch", hide_index=True)
            metric_columns = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC",
            ]
            chart_data = model_comparison.melt(
                id_vars="Model",
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score",
            )
            performance_figure = px.bar(
                chart_data,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#A9C4DA", GOLD, TEAL, NAVY],
                text_auto=".3f",
            )
            performance_figure.update_traces(
                texttemplate="%{y:.1%}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Metric: %{x}"
                    "<br>Score: %{y:.2%}<extra></extra>"
                ),
            )
            performance_figure.update_layout(
                title="Test-Set Model Comparison",
                height=520,
                xaxis_title="Evaluation Metric",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#E6ECF2"),
                margin=dict(l=20, r=20, t=65, b=90),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            style_figure(performance_figure, "Model")
            st.plotly_chart(performance_figure, width="stretch")
            if not cross_validation.empty:
                st.markdown("#### Five-Fold Cross-Validation Table")
                cv_columns = [
                    "Model",
                    "Mean Accuracy",
                    "Mean Precision",
                    "Mean Recall",
                    "Mean F1-Score",
                    "Mean ROC-AUC",
                    "ROC-AUC Standard Deviation",
                ]
                st.dataframe(
                    cross_validation[cv_columns],
                    width="stretch",
                    hide_index=True,
                )
            if not threshold_analysis.empty:
                st.markdown("#### Classification Threshold Trade-Off")
                threshold_chart_data = threshold_analysis.melt(
                    id_vars="Threshold",
                    value_vars=["Precision", "Recall", "F1-Score"],
                    var_name="Metric",
                    value_name="Score",
                )
                threshold_figure = px.line(
                    threshold_chart_data,
                    x="Threshold",
                    y="Score",
                    color="Metric",
                    markers=True,
                    color_discrete_sequence=[TEAL, RED, NAVY],
                )
                threshold_figure.add_vline(
                    x=0.35,
                    line_width=2,
                    line_dash="dash",
                    line_color=GOLD,
                    annotation_text="Selected 35%",
                    annotation_position="top right",
                )
                threshold_figure.update_traces(
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Threshold: %{x:.0%}"
                        "<br>Score: %{y:.2%}<extra></extra>"
                    )
                )
                threshold_figure.update_layout(
                    title="Precision, Recall and F1 by Classification Threshold",
                    height=520,
                    xaxis_title="Classification Threshold",
                    yaxis_title="Metric Score",
                    xaxis=dict(tickformat=".0%", gridcolor="#E6ECF2"),
                    yaxis=dict(tickformat=".0%", gridcolor="#E6ECF2"),
                    margin=dict(l=20, r=20, t=70, b=90),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                style_figure(threshold_figure, "Metric")
                st.plotly_chart(threshold_figure, width="stretch")
                st.markdown("#### Threshold Metrics Table")
                st.dataframe(
                    threshold_analysis,
                    width="stretch",
                    hide_index=True,
                )

    with importance_tab:
        source = shap_importance if not shap_importance.empty else feature_importance
        if source.empty:
            st.warning("Feature-importance files are unavailable.")
        else:
            value_column = (
                "Mean Absolute SHAP Value"
                if "Mean Absolute SHAP Value" in source.columns
                else "Importance"
            )
            top_features = source.head(10).sort_values(value_column)
            importance_figure = px.bar(
                top_features,
                x=value_column,
                y="Feature",
                orientation="h",
                color=value_column,
                color_continuous_scale=["#B7D7E8", TEAL, NAVY],
                text=value_column,
            )
            importance_figure.update_traces(
                texttemplate="%{x:.4f}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                ),
            )
            importance_figure.update_layout(
                title="Top Churn Drivers",
                height=540,
                yaxis_title="",
                xaxis_title=(
                    "Mean Absolute SHAP Value"
                    if value_column == "Mean Absolute SHAP Value"
                    else "Feature Importance"
                ),
                coloraxis_colorbar_title="Importance",
                margin=dict(l=20, r=95, t=65, b=50),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            style_figure(importance_figure, "Importance")
            st.plotly_chart(importance_figure, width="stretch")
            st.markdown("#### Explainability Ranking Table")
            st.dataframe(source, width="stretch", hide_index=True)

    with governance_tab:
        st.markdown(
            """
            <div class="governance-notepad">
                <h3>Model Governance Notes</h3>
                <ul>
                    <li><b>Human oversight:</b> Predictions support retention prioritization and do not replace employee review.</li>
                    <li><b>Validated performance:</b> Formal performance claims use the held-out test set and stratified five-fold cross-validation.</li>
                    <li><b>Decision threshold:</b> The selected 35% threshold maximizes the evaluated F1-score; operational risk bands remain separate.</li>
                    <li><b>Fair treatment:</b> Demographic variables must not be used for discriminatory pricing, access, or service decisions.</li>
                    <li><b>Ongoing monitoring:</b> Track drift, calibration, segment performance, false alarms, and missed churners after deployment.</li>
                    <li><b>Controlled retraining:</b> Retrain only with validated current data and retain versioned model documentation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
