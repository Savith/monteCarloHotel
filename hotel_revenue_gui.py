import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hotel Revenue Monte Carlo", layout="wide")
st.title("Monte Carlo Hotel Revenue Simulator")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
n_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=100, value=11)
n_simulations = st.sidebar.slider("Number of Simulations", 1000, 500000, 10000, step=1000)

st.sidebar.header("Occupancy Settings")
occupancy_min = st.sidebar.slider("Min Occupancy", 0.0, 1.0, 0.30)
occupancy_mode = st.sidebar.slider("Mode Occupancy", 0.0, 1.0, 0.60)
occupancy_max = st.sidebar.slider("Max Occupancy", 0.0, 1.0, 0.85)

st.sidebar.header("ADR Settings")
base_price_july = st.sidebar.number_input("Base ADR for July (High Season)", value=127)
adr_std_dev_percentage = st.sidebar.slider("ADR Std Dev (%)", 0.0, 0.3, 0.07)

st.sidebar.header("Additional Guest Spending")
guest_spending_per_night_per_room = st.sidebar.number_input("Avg. Guest Spending per Night", value=50)
spending_std_dev = st.sidebar.number_input("Spending Std Dev", value=15)

st.sidebar.header("ADR-Occupancy Elasticity")
adr_occupancy_correlation = st.sidebar.slider("Correlation (-1 to 0)", -1.0, 0.0, -0.5)

st.sidebar.header("Seasonality Factors (Monthly)")
seasonality_factors = {}
for month in range(1, 13):
    default_factor = {
        1: 0.9, 2: 0.85, 3: 0.85, 4: 0.9,
        5: 0.95, 6: 1.0, 7: 1.0,
        8: 0.95, 9: 0.9, 10: 0.9,
        11: 0.9, 12: 0.95
    }[month]
    seasonality_factors[month] = st.sidebar.number_input(
        f"Seasonality Factor Month {month}",
        min_value=0.0, max_value=2.0, value=default_factor, step=0.01, format="%.2f"
    )

# --- Calculate monthly ADR base ---
monthly_adr_base = {m: base_price_july * seasonality_factors[m] for m in range(1, 13)}

# --- Simulation Core Logic ---
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

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

# --- Aggregate annual revenue ---
yearly_revenue_sim = np.array(list(results_monthly.values())).sum(axis=0)

# --- Display Annual Summary ---
st.subheader("Annual Revenue Summary")
st.metric("Mean Annual Revenue", f"\u20ac{np.mean(yearly_revenue_sim):,.2f}")
st.metric("P5 (Conservative)", f"\u20ac{np.percentile(yearly_revenue_sim, 5):,.2f}")
st.metric("P95 (Optimistic)", f"\u20ac{np.percentile(yearly_revenue_sim, 95):,.2f}")

# --- Annual Revenue Distribution Plot ---
st.subheader("Annual Revenue Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(yearly_revenue_sim, bins=50, kde=True, color='purple', ax=ax)
ax.axvline(np.mean(yearly_revenue_sim), color='red', linestyle='--', label='Mean')
ax.axvline(np.percentile(yearly_revenue_sim, 5), color='orange', linestyle='--', label='P5')
ax.axvline(np.percentile(yearly_revenue_sim, 95), color='green', linestyle='--', label='P95')
ax.legend()
st.pyplot(fig)

# --- Monthly Detailed Results ---
st.subheader("Monthly Revenue Summary and Distribution")

monthly_summary = []
for month in range(1, 13):
    data = results_monthly[month]
    mean = np.mean(data)
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    std = np.std(data)
    monthly_summary.append({
        "Month": month,
        "Mean (€)": mean,
        "P5 (€)": p5,
        "P95 (€)": p95,
        "Std Dev (€)": std
    })
    st.markdown(f"### Month {month} - Mean: €{mean:,.2f} | P5: €{p5:,.2f} | P95: €{p95:,.2f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(data, bins=40, kde=True, color='skyblue', ax=ax)
    ax.axvline(mean, color='red', linestyle='--', label='Mean')
    ax.axvline(p5, color='orange', linestyle='--', label='P5')
    ax.axvline(p95, color='green', linestyle='--', label='P95')
    ax.legend()
    st.pyplot(fig)

# --- Export to Excel ---
st.subheader("Export Results")
if st.button("Export to Excel"):
    df_monthly = pd.DataFrame(monthly_summary)
    df_sim = pd.DataFrame({f"Month_{m:02d}": results_monthly[m] for m in results_monthly})
    df_sim["Annual"] = yearly_revenue_sim

    # Create Excel with 2 sheets: summary + simulations
    with pd.ExcelWriter("hotel_revenue_simulation.xlsx") as writer:
        df_monthly.to_excel(writer, sheet_name="Monthly Summary", index=False)
        df_sim.to_excel(writer, sheet_name="Simulations", index=False)

    st.success("Exported hotel_revenue_simulation.xlsx")
