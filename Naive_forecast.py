import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Page Configuration
st.set_page_config(
    page_title="Baseline Forecast Models",
    page_icon="📈",
    layout="wide"
)


page_bg = """
<style>
.stApp {
    background: linear_gradient(135deg, #e0f7fa, #e1hee7);
    background-size: cover;
    background-attachment: fixed;
}
.block-container {
    background-color:rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
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


st.title("📈 Ferry Ticket Demand Forecasting")
st.write("Comparison of Naïve Forecast, Moving Average, and Linear Regression")


# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\hp\PycharmProjects\Ferry_Demand_Repository\data\ferryticketdata.csv"
    )

df = load_data()

# Target Columns
ticket_column = "Redemption Count"

if ticket_column not in df.columns:
    st.error(f"'{ticket_column}' column not found.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Convert demand column to numeric
df[ticket_column] = pd.to_numeric(df[ticket_column], errors="coerce")

# Remove missing values
df = df.dropna(subset=[ticket_column]).copy()

# Create record sequence
df["Interval"] = range(1, len(df) + 1)

# ---------------------------------------------------
# 1. Naïve Forecast
# ---------------------------------------------------
df["Naive_Forecast"] = df[ticket_column].shift(1)

# ---------------------------------------------------
# 2. Moving Average Forecast
# ---------------------------------------------------
df["Moving_Average_Forecast"] = (
    df[ticket_column]
    .rolling(window=3)
    .mean()
    .shift(1)
)

# ---------------------------------------------------
# 3. Linear Regression with Lag Features
# ---------------------------------------------------
# Previous 1, 2 and 3 records are used as features
df["Lag_1"] = df[ticket_column].shift(1)
df["Lag_2"] = df[ticket_column].shift(2)
df["Lag_3"] = df[ticket_column].shift(3)

# Rows with missing lag values remove
lr_df = df.dropna(subset=["Lag_1", "Lag_2", "Lag_3"]).copy()

# Features and target
X = lr_df[["Lag_1", "Lag_2", "Lag_3"]]
y = lr_df[ticket_column]

# Time-based train-test split: 80% train, 20% test
split_index = int(len(lr_df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Prediction
lr_predictions = linear_model.predict(X_test)

# Store predictions in dataframe
test_df = lr_df.iloc[split_index:].copy()
test_df["Linear_Regression_Prediction"] = lr_predictions

# ---------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------
st.subheader("Demand Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Redemption Demand", f"{df[ticket_column].sum():,.0f}")

with col2:
    st.metric("Average Redemption Demand", f"{df[ticket_column].mean():.0f}")

with col3:
    st.metric("Maximum Redemption Demand", f"{df[ticket_column].max():,.0f}")

# ---------------------------------------------------
# Naïve Forecast Graph
# ---------------------------------------------------
st.subheader("1. Naïve Forecast Baseline Model")

naive_df = df.dropna(subset=["Naive_Forecast"]).copy()
naive_plot_df = naive_df.tail(200)

fig_naive = go.Figure()

fig_naive.add_trace(
    go.Scatter(
        x=naive_plot_df["Interval"],
        y=naive_plot_df[ticket_column],
        mode="lines",
        name="Actual Redemption Demand"
    )
)

fig_naive.add_trace(
    go.Scatter(
        x=naive_plot_df["Interval"],
        y=naive_plot_df["Naive_Forecast"],
        mode="lines",
        name="Naïve Forecast"
    )
)

fig_naive.update_layout(
    title="Actual vs Naïve Forecast",
    xaxis_title="Record / Time Interval Sequence",
    yaxis_title="Redemption Count",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_naive, use_container_width=True)

# ---------------------------------------------------
# Moving Average Graph
# ---------------------------------------------------
st.subheader("2. Moving Average Forecast Baseline Model")

moving_df = df.dropna(subset=["Moving_Average_Forecast"]).copy()
moving_plot_df = moving_df.tail(200)

fig_moving = go.Figure()

fig_moving.add_trace(
    go.Scatter(
        x=moving_plot_df["Interval"],
        y=moving_plot_df[ticket_column],
        mode="lines",
        name="Actual Redemption Demand"
    )
)

fig_moving.add_trace(
    go.Scatter(
        x=moving_plot_df["Interval"],
        y=moving_plot_df["Moving_Average_Forecast"],
        mode="lines",
        name="Moving Average Forecast"
    )
)

fig_moving.update_layout(
    title="Actual vs Moving Average Forecast",
    xaxis_title="Record / Time Interval Sequence",
    yaxis_title="Redemption Count",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_moving, use_container_width=True)

# ---------------------------------------------------
# Linear Regression Graph
# ---------------------------------------------------
st.subheader("3. Linear Regression with Lag Features")

# Last 200 test records only
lr_plot_df = test_df.tail(200)

fig_lr = go.Figure()

fig_lr.add_trace(
    go.Scatter(
        x=lr_plot_df["Interval"],
        y=lr_plot_df[ticket_column],
        mode="lines",
        name="Actual Redemption Demand"
    )
)

fig_lr.add_trace(
    go.Scatter(
        x=lr_plot_df["Interval"],
        y=lr_plot_df["Linear_Regression_Prediction"],
        mode="lines",
        name="Linear Regression Prediction"
    )
)

fig_lr.update_layout(
    title="Actual vs Linear Regression Prediction",
    xaxis_title="Record / Time Interval Sequence",
    yaxis_title="Redemption Count",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_lr, use_container_width=True)

# ---------------------------------------------------
# Model Evaluation
# ---------------------------------------------------
st.subheader("Model Performance Comparison")

# Same test data par baseline errors calculate karna
evaluation_df = test_df.copy()

naive_mae = mean_absolute_error(
    evaluation_df[ticket_column],
    evaluation_df["Naive_Forecast"]
)

moving_mae = mean_absolute_error(
    evaluation_df[ticket_column],
    evaluation_df["Moving_Average_Forecast"]
)

linear_regression_mae = mean_absolute_error(
    evaluation_df[ticket_column],
    evaluation_df["Linear_Regression_Prediction"]
)

performance_df = pd.DataFrame({
    "Model": [
        "Naïve Forecast",
        "Moving Average Forecast",
        "Linear Regression with Lag Features"
    ],
    "MAE": [
        round(naive_mae, 2),
        round(moving_mae, 2),
        round(linear_regression_mae, 2)
    ]
})

st.dataframe(performance_df, use_container_width=True)

best_model = performance_df.loc[performance_df["MAE"].idxmin(), "Model"]

st.success(f"Best model based on MAE: {best_model}")

# ---------------------------------------------------
# Forecast Data Table
# ---------------------------------------------------
st.subheader("Forecast Comparison Data")

st.dataframe(
    test_df[
        [
            "Interval",
            ticket_column,
            "Naive_Forecast",
            "Moving_Average_Forecast",
            "Linear_Regression_Prediction"
        ]
    ].tail(200),
    use_container_width=True
)