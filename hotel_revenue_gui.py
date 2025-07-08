import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from PIL import Image

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hotel Revenue Monte Carlo", layout="wide")
st.title("Monte Carlo Hotel Revenue Simulator")

# --- Add Background Image or Logo ---
image = Image.open("/mnt/data/dadb6196-cfa8-481d-9385-e329d178a47f.png")
st.image(image, caption="Podere di Sasso", use_column_width=True)

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
n_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=100, value=11)
n_simulations = st.sidebar.slider("Number of Simulations", 1000, 500000, 10000, step=1000)

st.sidebar.header("Occupancy Settings")
occupancy_min = st.sidebar.slider("Min Occupancy", 0.0, 1.0, 0.30)
occupancy_mode = st.sidebar.slider("Mode Occupancy", 0.0, 1.0, 0.60)
occupancy_max = st.sidebar.slider("Max Occupancy", 0.0, 1.0, 0.85)

st.sidebar.header("ADR Settings")
adr_std_dev_percentage = st.sidebar.slider("ADR Std Dev (%)", 0.0, 0.3, 0.07)

st.sidebar.header("Seasonality Coefficients")
def get_seasonality():
    default_seasonality = [0.111, 0.167, 0.167, 0.444, 0.444, 0.556, 0.833, 1.0, 0.833, 0.444, 0.333, 0.222]
    seasonality = {}
    for i, coef in enumerate(default_seasonality):
        month = i + 1
        seasonality[month] = st.sidebar.slider(f"Reduction Coef. Month {month:02d}", 0.0, 1.0, coef)
    return seasonality

seasonality_coefficients = get_seasonality()

st.sidebar.header("Additional Guest Spending")
guest_spending_per_night_per_room = st.sidebar.number_input("Avg. Guest Spending per Night", value=50)
spending_std_dev = st.sidebar.number_input("Spending Std Dev", value=15)

st.sidebar.header("ADR-Occupancy Elasticity")
adr_occupancy_correlation = st.sidebar.slider("Correlation (-1 to 0)", -1.0, 0.0, -0.5)

# Base ADR input
base_adr = st.sidebar.number_input("Base ADR (High Season, e.g. July)", value=127)

# --- Simulation Core Logic ---
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

monthly_adr_base = {month: base_adr * seasonality_coefficients[month] for month in range(1, 13)}
results_monthly = {}

for month in range(1, 13):
    adr_base = monthly_adr_base[month]

    mean = [0, 0]
    cov = [[1, adr_occupancy_correlation], [adr_occupancy_correlation, 1]]
    z = np.random.multivariate_normal(mean, cov, n_simulations)
    z1, z2 = z[:, 0], z[:, 1]

    occupancy_sim = np.clip(np.interp(norm.cdf(z1), [0, 1], [occupancy_min, occupancy_max]), 0, 1)

    adr_sim = np.random.normal(adr_base, adr_base * adr_std_dev_percentage, n_simulations)
    adr_sim *= (1 - 0.2 * z2)
    adr_sim = np.clip(adr_sim, 0, None)

    spend_sim = np.random.normal(guest_spending_per_night_per_room, spending_std_dev, n_simulations)
    spend_sim[spend_sim < 0] = 0

    revenue_per_night = (adr_sim + spend_sim) * occupancy_sim * n_rooms
    monthly_revenue = revenue_per_night * days_in_month[month]
    results_monthly[month] = monthly_revenue

# --- Results ---
yearly_revenue_sim = np.array(list(results_monthly.values())).sum(axis=0)

st.subheader("Annual Revenue Summary")
st.metric("Mean Annual Revenue", f"€{np.mean(yearly_revenue_sim):,.2f}")
st.metric("P5 (Conservative)", f"€{np.percentile(yearly_revenue_sim, 5):,.2f}")
st.metric("P95 (Optimistic)", f"€{np.percentile(yearly_revenue_sim, 95):,.2f}")
st.metric("Confidence Interval (90%)", f"€{np.percentile(yearly_revenue_sim, 5):,.2f} - €{np.percentile(yearly_revenue_sim, 95):,.2f}")

# --- Monthly Charts ---
st.subheader("Monthly Revenue Distribution")
fig, axs = plt.subplots(3, 4, figsize=(20, 10))
for i, (month, revenue) in enumerate(results_monthly.items()):
    ax = axs[i // 4, i % 4]
    sns.histplot(revenue, bins=50, kde=True, ax=ax, color='skyblue')
    ax.axvline(np.mean(revenue), color='red', linestyle='--', label='Mean')
    ax.axvline(np.percentile(revenue, 5), color='orange', linestyle='--', label='P5')
    ax.axvline(np.percentile(revenue, 95), color='green', linestyle='--', label='P95')
    ax.set_title(f"Month {month:02d}")
    ax.legend()
fig.tight_layout()
st.pyplot(fig)

# --- Export ---
st.subheader("Export Results")
if st.button("Export to Excel"):
    df_export = pd.DataFrame({f"Month {m:02d}": results_monthly[m] for m in results_monthly})
    df_export["Annual"] = yearly_revenue_sim
    df_export.to_excel("hotel_revenue_simulation.xlsx", index=False)
    st.success("Exported hotel_revenue_simulation.xlsx")
