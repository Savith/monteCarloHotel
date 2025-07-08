# Monte Carlo Hotel Revenue Simulator - Versão Melhorada com Tabs Organizacionais

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from PIL import Image
import os
import json

# --- Configuração da Página ---
st.set_page_config(page_title="Hotel Revenue Monte Carlo", layout="wide")
st.title("Monte Carlo Hotel Revenue Simulator")

# --- Imagem ---
try:
    image_path = "assets/podere.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Podere di Sasso", use_container_width=True)
    else:
        st.warning("Image not found. Please ensure 'assets/podere.png' exists in your project folder.")
except:
    st.warning("Error loading image.")

# --- Tabs Principais ---
tab_inputs, tab_simulation, tab_results, tab_export = st.tabs(["Inputs", "Simulation", "Results", "Export"])

# --- Tab de Inputs ---
with tab_inputs:
    st.header("Simulation Configuration")

    col1, col2 = st.columns(2)
    with col1:
        scenario_name = st.text_input("Scenario Name", value="Default Scenario", key="scenario_name")
        n_rooms = st.number_input("Number of Rooms", min_value=1, max_value=100, value=11, key="n_rooms")
        n_simulations = st.slider("Number of Simulations", 1000, 500000, 10000, step=1000, key="n_simulations")
        base_adr = st.number_input("Base ADR (High Season)", value=127, key="base_adr")

    with col2:
        inflation_rate = st.slider("Annual Inflation Rate (%)", 0.0, 10.0, 2.5)
        demand_growth = st.slider("Demand Growth Rate (%)", 0.0, 10.0, 3.0)
        unexpected_costs_mean = st.number_input("Unexpected Costs (Mean €)", value=15000)
        unexpected_costs_std = st.number_input("Unexpected Costs (Std Dev €)", value=5000)

    st.subheader("Occupancy Settings")
    col3, col4, col5 = st.columns(3)
    occupancy_min = col3.slider("Min Occupancy", 0.0, 1.0, 0.30, key="occupancy_min")
    occupancy_mode = col4.slider("Mode Occupancy", 0.0, 1.0, 0.60, key="occupancy_mode")
    occupancy_max = col5.slider("Max Occupancy", 0.0, 1.0, 0.85, key="occupancy_max")

    st.subheader("ADR Settings")
    adr_std_dev_percentage = st.slider("ADR Std Dev (%)", 0.0, 0.3, 0.07, key="adr_std_dev_percentage")
    adr_occupancy_correlation = st.slider("ADR-Occupancy Correlation (-1 to 0)", -1.0, 0.0, -0.5, key="adr_occupancy_correlation")

    st.subheader("Seasonality Coefficients")
    default_seasonality = [0.111, 0.167, 0.167, 0.444, 0.444, 0.556, 0.833, 1.0, 0.833, 0.444, 0.333, 0.222]
    seasonality_coefficients = {}
    for i, coef in enumerate(default_seasonality):
        month = i + 1
        seasonality_coefficients[month] = st.slider(f"Reduction Coef. Month {month:02d}", 0.0, 1.0, coef, key=f"seasonality_{month}")

    st.subheader("Guest Spending")
    guest_spending_per_night_per_room = st.number_input("Avg. Guest Spending per Night", value=50, key="guest_spending_per_night_per_room")
    spending_std_dev = st.number_input("Spending Std Dev", value=15, key="spending_std_dev")

# --- Simulação ---
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

yearly_revenue_sim = np.array(list(results_monthly.values())).sum(axis=0)

# --- Resultados ---
with tab_results:
    st.header("Annual Revenue Summary")
    st.metric("Scenario", scenario_name)
    st.metric("Mean Annual Revenue", f"€{np.mean(yearly_revenue_sim):,.2f}")
    st.metric("P5 (Conservative)", f"€{np.percentile(yearly_revenue_sim, 5):,.2f}")
    st.metric("P95 (Optimistic)", f"€{np.percentile(yearly_revenue_sim, 95):,.2f}")
    st.metric("Confidence Interval (90%)", f"€{np.percentile(yearly_revenue_sim, 5):,.2f} - €{np.percentile(yearly_revenue_sim, 95):,.2f}")

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

# --- Exportação ---
with tab_export:
    st.header("Export Results")
    if st.button("Export to Excel"):
        df_export = pd.DataFrame({f"Month {m:02d}": results_monthly[m] for m in results_monthly})
        df_export["Annual"] = yearly_revenue_sim
        df_export.to_excel("hotel_revenue_simulation.xlsx", index=False)
        st.success("Exported hotel_revenue_simulation.xlsx")
