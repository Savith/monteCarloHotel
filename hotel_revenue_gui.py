import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from PIL import Image
import os
import json

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hotel Revenue Monte Carlo", layout="wide")
st.title("Monte Carlo Hotel Revenue Simulator")

# --- Add Background Image or Logo ---
try:
    image_path = "assets/podere.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Podere di Sasso", use_container_width=True)
    else:
        st.warning("Image not found. Please ensure 'assets/podere.png' exists in your project folder.")
except:
    st.warning("Error loading image.")

# --- Tabs for better organization ---
tabs = st.tabs(["Simulation", "Assumptions", "Save/Load Scenario"])

with tabs[1]:
    st.header("Simulation Assumptions")
    st.markdown("Configure core assumptions like inflation, growth, reforms, legal fees, etc.")
    inflation_rate = st.slider("Annual Inflation Rate (%)", 0.0, 10.0, 2.5)
    demand_growth = st.slider("Demand Growth Rate (%)", 0.0, 10.0, 3.0)
    unexpected_costs_mean = st.number_input("Unexpected Costs (Mean €)", value=15000)
    unexpected_costs_std = st.number_input("Unexpected Costs (Std Dev €)", value=5000)

with tabs[2]:
    st.header("Save or Load Scenarios")

    # Saving Scenario
    if st.button("Save Current Scenario"):
        scenario_data = {
            "scenario_name": st.session_state.get("scenario_name", "Unnamed Scenario"),
            "n_rooms": st.session_state.get("n_rooms"),
            "n_simulations": st.session_state.get("n_simulations"),
            "occupancy_min": st.session_state.get("occupancy_min"),
            "occupancy_mode": st.session_state.get("occupancy_mode"),
            "occupancy_max": st.session_state.get("occupancy_max"),
            "adr_std_dev_percentage": st.session_state.get("adr_std_dev_percentage"),
            "seasonality_coefficients": st.session_state.get("seasonality_coefficients"),
            "guest_spending_per_night_per_room": st.session_state.get("guest_spending_per_night_per_room"),
            "spending_std_dev": st.session_state.get("spending_std_dev"),
            "adr_occupancy_correlation": st.session_state.get("adr_occupancy_correlation"),
            "base_adr": st.session_state.get("base_adr"),
            "inflation_rate": inflation_rate,
            "demand_growth": demand_growth,
            "unexpected_costs_mean": unexpected_costs_mean,
            "unexpected_costs_std": unexpected_costs_std
        }
        with open("saved_scenario.json", "w") as f:
            json.dump(scenario_data, f)
        st.success("Scenario saved successfully!")

    # Loading Scenario
    if st.button("Load Last Scenario"):
        try:
            with open("saved_scenario.json", "r") as f:
                loaded = json.load(f)
            st.session_state.update(loaded)
            st.success("Scenario loaded. Please refresh the app to see values.")
        except:
            st.error("No saved scenario found.")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
scenario_name = st.sidebar.text_input("Scenario Name", value="Default Scenario", key="scenario_name")
n_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=100, value=11, key="n_rooms")
n_simulations = st.sidebar.slider("Number of Simulations", 1000, 500000, 10000, step=1000, key="n_simulations")

st.sidebar.header("Occupancy Settings")
occupancy_min = st.sidebar.slider("Min Occupancy", 0.0, 1.0, 0.30, key="occupancy_min")
occupancy_mode = st.sidebar.slider("Mode Occupancy", 0.0, 1.0, 0.60, key="occupancy_mode")
occupancy_max = st.sidebar.slider("Max Occupancy", 0.0, 1.0, 0.85, key="occupancy_max")

st.sidebar.header("ADR Settings")
adr_std_dev_percentage = st.sidebar.slider("ADR Std Dev (%)", 0.0, 0.3, 0.07, key="adr_std_dev_percentage")

st.sidebar.header("Seasonality Coefficients")
def get_seasonality():
    default_seasonality = [0.111, 0.167, 0.167, 0.444, 0.444, 0.556, 0.833, 1.0, 0.833, 0.444, 0.333, 0.222]
    seasonality = {}
    for i, coef in enumerate(default_seasonality):
        month = i + 1
        seasonality[month] = st.sidebar.slider(f"Reduction Coef. Month {month:02d}", 0.0, 1.0, coef, key=f"seasonality_{month}")
    return seasonality

seasonality_coefficients = get_seasonality()
st.session_state["seasonality_coefficients"] = seasonality_coefficients

st.sidebar.header("Additional Guest Spending")
guest_spending_per_night_per_room = st.sidebar.number_input("Avg. Guest Spending per Night", value=50, key="guest_spending_per_night_per_room")
spending_std_dev = st.sidebar.number_input("Spending Std Dev", value=15, key="spending_std_dev")

st.sidebar.header("ADR-Occupancy Elasticity")
adr_occupancy_correlation = st.sidebar.slider("Correlation (-1 to 0)", -1.0, 0.0, -0.5, key="adr_occupancy_correlation")

base_adr = st.sidebar.number_input("Base ADR (High Season, e.g. July)", value=127, key="base_adr")

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
st.metric("Scenario", scenario_name)
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
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --- Export ---
st.subheader("Export Results")
if st.button("Export to Excel"):
    df_export = pd.DataFrame({f"Month {m:02d}": results_monthly[m] for m in results_monthly})
    df_export["Annual"] = yearly_revenue_sim
    df_export.to_excel("hotel_revenue_simulation.xlsx", index=False)
    st.success("Exported hotel_revenue_simulation.xlsx")

# --- Ideal Price Simulation ---
st.subheader("Ideal ADR Price Simulation")
adrs_to_test = np.arange(80, 180, 5)
expected_revenues = []
for test_adr in adrs_to_test:
    simulated = []
    for month in range(1, 13):
        adr_base = test_adr * seasonality_coefficients[month]
        adr_sim = np.random.normal(adr_base, adr_base * adr_std_dev_percentage, n_simulations)
        occupancy_sim = np.random.triangular(occupancy_min, occupancy_mode, occupancy_max, n_simulations)
        revenue = adr_sim * occupancy_sim * n_rooms * days_in_month[month]
        simulated.append(np.mean(revenue))
    expected_revenues.append(np.sum(simulated))

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(adrs_to_test, expected_revenues, marker='o')
ax2.set_title("Expected Annual Revenue by ADR")
ax2.set_xlabel("ADR (€)")
ax2.set_ylabel("Expected Revenue (€)")
ax2.grid(True)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)
