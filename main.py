import streamlit as st
import pandas as pd
import plotly.express as px
from PIL.Image import linear_gradient
from starlette import background

st.set_page_config(page_title="Ferry Ticket Demand Forecasting",
                   page_icon="🚢",
                   layout="wide")


page_bg = """
<style>
.stApp {
    background: linear_gradient(to right, #E3F2FD, #F3E5F5);
    background-attachment: fixed;
}
</style>
"""

st.markdown ("""
<style>
[data-testid="stAppViewContainer"] {
background: lightblue;
}
</style>)
""", unsafe_allow_html=True)



st.title("🚢 Short-Term Ferry Ticket Demand Forecasting")
st.write("Predictive Decision Support System for Ferry Operations")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\hp\PycharmProjects\Ferry_Demand_Repository\data\ferryticketdata.csv")
    return data

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")

if "Route" in df.columns:
    route = st.sidebar.selectbox("Select Route", df["Route"].unique())
    filtered_df = df[df["Route"] == route]
else:
    filtered_df = df


# KPI cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tickets", f"{filtered_df['Redemption Count'].sum():,}")

    with col2:
        st.metric("Average Daily Demand", f"{filtered_df['Redemption Count'].mean():.0f}")

        with col3:
            st.metric("Maximum Demand", f"{filtered_df['Sales Count'].max():,}")



# Demand trend chart
st.subheader("Ticket Demand Trend")

if "Date" in filtered_df.columns:
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])

    fig = px.line(filtered_df, x="Date", y="Ticket Count", title="Daily Ferry Ticket Demand")
    st.plotly_chart(fig, use_container_width=True)


# Data Table
st.subheader("Ferry Ticket Dataset")
st.dataframe(filtered_df)

# Forecast section
st.subheader("Predicted Demand")

predicted_demand = filtered_df["Redemption Count"].mean() * 1.10
st.success(f"Predicted ticket demand for the next period: {predicted_demand:.0f} tickets")
st.info("Recommendation: Increase ferry capacity and staff availability during high-demand periods.")



